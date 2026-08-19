import json
import logging
import torchaudio
from torch.utils.data import Dataset
import datasets
import torch
import trl 
from transformers import DataCollatorForLanguageModeling
from typing import Any, Literal, Optional, Union
import warnings
import numpy as np 
def _handle_wav(wav_path, target_rate=16000):
    """
    handle one wav file.
    Return:
        waveform: numpy narray(1d)
    """
    waveform, sample_rate = torchaudio.load(wav_path)
    # print(sample_rate)
    if sample_rate != 16000:
        # print(target_rate)
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_rate)(waveform)
    audio = waveform[0]
    return audio

class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    """
    Data collator used for completion tasks. It ensures that all the tokens of the labels are set to an 'ignore_index'
    when they do not come from the assistant. This ensure that the loss is only calculated on the completion made by
    the assistant.

    Args:
        response_template (`Union[str, list[int]]`):
            the template form that indicates the start of the response, typically something like '### Response:\n'. It
            can also be passed as tokenized ids, which can be useful when using a tokenizer that encodes the response
            differently if it does not have proper context.
        instruction_template (`Union[str, list[int]]`):
            the template form that indicates the start of the human instruction, typically something like '###
            Human:\n'. Useful for assistant-style conversation datasets. It can also be passed as tokenized ids.
        mlm (`bool`, *optional*, defaults to `False`): Whether to use masked language modeling in the underlying
            `DataCollatorForLanguageModeling` class. Note that this option currently has no effect but is present
             for flexibility and backwards-compatibility.
        ignore_index (`int`, *optional*, defaults to `-100`):
            The index to use to ignore the initial tokens with
    """

    def __init__(
        self,
        response_template: Union[str, list[int]],
        instruction_template: Optional[Union[str, list[int]]] = None,
        *args,
        mlm: bool = False,
        ignore_index: int = -100,
        padding_free: bool = False,
        **kwargs,
    ):
        super().__init__(*args, mlm=mlm, **kwargs)
        warnings.warn(
            "This class is deprecated and will be removed in version 0.20.0. To train on completion only, please use "
            "the parameter `completion_only_loss` of `SFTConfig` instead.",
            DeprecationWarning,
        )

        self.instruction_template = instruction_template
        if isinstance(instruction_template, str):
            # The user provides a string, must tokenize
            self.instruction_token_ids = self.tokenizer.encode(self.instruction_template, add_special_tokens=False)
        else:
            # The user already provides the token ids
            self.instruction_token_ids = instruction_template

        self.response_template = response_template
        if isinstance(response_template, str):
            # The user provides a string, must tokenize
            self.response_token_ids = self.tokenizer.encode(self.response_template, add_special_tokens=False)
        else:
            # The user already provides the token ids
            self.response_token_ids = response_template

        if not self.mlm and self.instruction_template and self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            warnings.warn(
                "The pad_token_id and eos_token_id values of this tokenizer are identical. "
                "If you are planning for multi-turn training, "
                "it can result in the model continuously generating questions and answers without eos token. "
                "To avoid this, set the pad_token_id to a different value.",
                UserWarning,
            )

        self.ignore_index = ignore_index
        self.padding_free = padding_free

    def torch_call(self, examples: list[Union[list[int], Any, dict[str, Any]]]) -> dict[str, Any]:
        batch = super().torch_call(examples)

        if self.instruction_template is None:
            for i in range(len(examples)):
                response_token_ids_start_idx = None

                for idx in np.where(batch["labels"][i] == self.response_token_ids[0])[0]:
                    # `response_token_ids` is `'### Response:\n'`, here we are just making sure that the token IDs match
                    if (
                        self.response_token_ids
                        == batch["labels"][i][idx : idx + len(self.response_token_ids)].tolist()
                    ):
                        response_token_ids_start_idx = idx

                if response_token_ids_start_idx is None:
                    warnings.warn(
                        f"Could not find response key `{self.response_template}` in the following instance: "
                        f"{self.tokenizer.decode(batch['input_ids'][i])}. This instance will be ignored in loss "
                        "calculation. Note, if this happens often, consider increasing the `max_length`.",
                        UserWarning,
                    )
                    batch["labels"][i, :] = self.ignore_index
                else:
                    response_token_ids_end_idx = response_token_ids_start_idx + len(self.response_token_ids)

                    # Make pytorch loss function ignore all tokens up through the end of the response key
                    batch["labels"][i, :response_token_ids_end_idx] = self.ignore_index

        else:
            for i in range(len(examples)):
                response_token_ids_idxs = []
                human_token_ids_idxs = []

                for assistant_idx in np.where(batch["labels"][i] == self.response_token_ids[0])[0]:
                    # find the indexes of the start of a response.
                    if (
                        self.response_token_ids
                        == batch["labels"][i][assistant_idx : assistant_idx + len(self.response_token_ids)].tolist()
                    ):
                        response_token_ids_idxs.append(assistant_idx + len(self.response_token_ids))

                if len(response_token_ids_idxs) == 0:
                    warnings.warn(
                        f"Could not find response key `{self.response_template}` in the following instance: "
                        f"{self.tokenizer.decode(batch['input_ids'][i])}. This instance will be ignored in loss "
                        "calculation. Note, if this happens often, consider increasing the `max_length`.",
                        UserWarning,
                    )
                    batch["labels"][i, :] = self.ignore_index

                human_token_ids = self.instruction_token_ids
                for human_idx in np.where(batch["labels"][i] == human_token_ids[0])[0]:
                    # find the indexes of the start of a human answer.
                    if human_token_ids == batch["labels"][i][human_idx : human_idx + len(human_token_ids)].tolist():
                        human_token_ids_idxs.append(human_idx)

                if len(human_token_ids_idxs) == 0:
                    warnings.warn(
                        f"Could not find instruction key `{self.instruction_template}` in the following instance: "
                        f"{self.tokenizer.decode(batch['input_ids'][i])}. This instance will be ignored in loss "
                        "calculation. Note, if this happens often, consider increasing the `max_length`.",
                        UserWarning,
                    )
                    batch["labels"][i, :] = self.ignore_index

                if (
                    len(human_token_ids_idxs) > 0
                    and len(response_token_ids_idxs) > 0
                    and human_token_ids_idxs[0] > response_token_ids_idxs[0]
                ):
                    human_token_ids_idxs = [0] + human_token_ids_idxs

                for idx, (start, end) in enumerate(zip(human_token_ids_idxs, response_token_ids_idxs)):
                    # Make pytorch loss function ignore all non response tokens
                    if idx != 0:
                        batch["labels"][i, start:end] = self.ignore_index
                    else:
                        batch["labels"][i, :end] = self.ignore_index

                if len(response_token_ids_idxs) < len(human_token_ids_idxs):
                    batch["labels"][i, human_token_ids_idxs[-1] :] = self.ignore_index

        if self.padding_free:
            # remove padding, `attention_mask` and add `position_ids`
            attn_mask = batch.pop("attention_mask")
            batch["input_ids"] = batch["input_ids"][attn_mask.bool()].unsqueeze(0)
            batch["position_ids"] = attn_mask.cumsum(1)[attn_mask.bool()].unsqueeze(0) - 1
            batch["labels"] = batch["labels"][attn_mask.bool()].unsqueeze(0)
            batch["labels"][batch["position_ids"] == 0] = self.ignore_index

            # Calculate cumulative sequence lengths for queries and keys to prevent graph breaks during further computations.
            flattened_position_ids = batch["position_ids"].flatten()
            indices_q = torch.arange(
                flattened_position_ids.size(0), device=flattened_position_ids.device, dtype=torch.int32
            )
            batch["cu_seq_lens_q"] = torch.cat(
                (
                    indices_q[flattened_position_ids == 0],
                    torch.tensor(
                        flattened_position_ids.size(), device=flattened_position_ids.device, dtype=torch.int32
                    ),
                )
            ).unsqueeze(0)
            batch["cu_seq_lens_k"] = batch["cu_seq_lens_q"]

            # Determine maximum sequence lengths to prevent graph breaks during further computations.
            batch["max_length_k"] = torch.tensor([flattened_position_ids.max().item() + 1])
            batch["max_length_q"] = batch["max_length_k"]

        return batch

class CustomDataCollatorForSFT(DataCollatorForCompletionOnlyLM):
    def __call__(self, features):
        # Separate 'source' from the features
        # Assuming 'features' is a list of dicts, one dict per sample
        for feature in features:
            feature.pop("prompt")
            feature.pop("response")
            feature.pop("audio")
            # feature.pop("input_features")
            # feature.pop("feature_attention_mask")
        if "feature_attention_mask" in features[0]:
            mask_name = "feature_attention_mask"
        elif "input_features_mask" in features[0]:
            mask_name = "input_features_mask"

        dataset_names = [feature.pop("dataset_name") for feature in features]
        input_features = torch.tensor(  [feature.pop("input_features") for feature in features])

        feature_attention_mask =  torch.tensor( [feature.pop(mask_name) for feature in features])
        flag=False
        if "wave" in features[0].keys():
            flag=True
            waves =   [ torch.tensor(feature.pop("wave")) for feature in features]
        # print(input_features.shape)
        # print(feature_attention_mask.shape)
        batch = super().__call__(features)            
        batch["dataset_name"] = dataset_names
        # Put them back into the batch
        batch["input_features"] = input_features
        batch[mask_name] = feature_attention_mask
        torch.set_printoptions(threshold=float("inf"))
        # print(batch["input_ids"])
        # print(batch["labels"])
        # print(mask_name)
        # print(batch["input_features"].shape)
        # print(batch["input_features_mask"].shape)
        # print("Total input_ids length:", batch["input_ids"].shape[-1])
        if flag:
            batch ["wave"] = waves
        # print()
        return batch



class AudioDataset(Dataset):
    def __init__(self, data_file, reward_format,safe_data_file , safety_mixture, sample_rate=16000, is_perturb=False, data_num= 1000):
        super().__init__()
        # self.lists = []
        if "gsm8k" in data_file:
            self.obj_list =datasets.load_dataset(data_file)["train"]
        else:
            with open(data_file, 'r') as f:
                json_content = f.read()
            self.obj_list = json.loads(json_content)
        self.sample_rate = sample_rate
        self.is_perturb = is_perturb
        logging.info(f"{data_file}, len:{len(self.obj_list)}, rate:{sample_rate}")
        self.data_num=data_num
        self.reward_format= reward_format
        self.safety_mixture= safety_mixture
        self.total_count = data_num*2
        self.safety_count = data_num
        self.safety_dataset =datasets.load_dataset(safe_data_file)["train"]
        
    def __len__(self):
        return self.total_count


    # def _handle_avqa(self, obj_avqa):
    #     choice_str = f"Please choose the answer from the following options: {obj_avqa['multi_choice']}."
    #     if self.reward_format!="cot":
    #         question_template = f"{obj_avqa['question_text'].replace('video', 'audio')} {choice_str} Output the final answer in <answer> </answer>."
    #         # If you want to improve the thinking process, uncomment the next line and design your strategy.
    #     else:
    #         question_template = f"{obj_avqa['question_text'].replace('video', 'audio')} {choice_str} Please think about this question as if you were a human pondering deeply. It's encouraged to include self-reflection or verification in the reasoning process. Output the thinking process in <think> </think> and final answer in <answer> </answer>."
    #     obj_avqa["prompt"] = [{"role": "user", "content": [{"type": "audio", "audio_url": None}, {"type": "text", "text": question_template}]}]
    #     answer_str = obj_avqa["multi_choice"][obj_avqa["answer"]]
    #     obj_avqa["solution"] = f"<answer>{answer_str}</answer>"
    #     # print(obj_avqa)
    #     return obj_avqa


    def handle_json_line(self, obj, sample_rate=16000):
        if self.reward_format!="cot":
            question_template = f"Based on the given audio, answer the speaker's question. Output the final answer in <answer> </answer>."
        else:
            question_template = f"Based on the given audio, answer the speaker's question. Please think about this question as if you were a human pondering deeply. It's encouraged to include self-reflection or verification in the reasoning process. Output the thinking process in <think> </think> and final answer in <answer> </answer>."
        obj["prompt"] = [{"role": "user", "content": [{"type": "audio", "audio_url": None}, {"type": "text", "text": question_template}]}]
        waveform = torch.tensor(obj["audio"]["array"])
        obj["audio"] = waveform.numpy()
        return obj


    def handle_safety_data(self, data_obj):
        choice_str = f"Based on the given audio, answer the speaker's question."
        if self.reward_format!="cot":
            question_template = f"{choice_str} Output the final answer in <answer> </answer>."
            # If you want to improve the thinking process, uncomment the next line and design your strategy.
        else:
            question_template = f"Based on the given audio, answer the speaker's question. Please think about this question as if you were a human pondering deeply. It's encouraged to include self-reflection or verification in the reasoning process. Output the thinking process in <think> </think> and final answer in <answer> </answer>."
        data_obj["prompt"]=  [{"role": "user", "content": [{"type": "audio", "audio_url": None}, {"type": "text", "text": question_template}]}]
        data_obj["solution"] =  ""
        # process audio
        waveform = torch.tensor(data_obj["audio"]["array"])
        data_obj["audio"] = waveform.numpy()
        return data_obj

    def __getitem__(self, index):
        # data = handle_json_line(self.obj_list[index])
        if index>=self.safety_count:
            return self.handle_json_line (self.obj_list[index-self.safety_count])
        else:
            return self.handle_safety_data (self.safety_dataset[index])