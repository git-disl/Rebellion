
from trl import SFTTrainer, SFTConfig
import contextlib
import dataclasses
import os
import warnings
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union, Dict, List, Optional, Tuple
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
import torchaudio
import torch
import torch.nn as nn
import transformers
from accelerate import PartialState
from datasets import Dataset, IterableDataset,DatasetDict
from packaging import version
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BaseImageProcessor,
    DataCollator,
    DataCollatorWithFlattening,
    FeatureExtractionMixin,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
    TrainingArguments,
    is_wandb_available,
)
import pyarrow
import pyarrow.compute as pc
from typing import Any, TypeVar
DatasetType = TypeVar("DatasetType", Dataset, DatasetDict)

def truncate_dataset(dataset: DatasetType, max_length: int, map_kwargs: dict[str, Any] | None = None) -> DatasetType:
    r"""
    Truncate sequences in a dataset to a specified `max_length`.

    Args:
        dataset ([`~datasets.Dataset`] or [`~datasets.DatasetDict`]):
            Dataset to truncate.
        max_length (`int`):
            Maximum sequence length to truncate to.
        map_kwargs (`dict`, *optional*):
            Additional keyword arguments to pass to the dataset's map method when truncating examples.

    Returns:
        [`~datasets.Dataset`] or [`~datasets.DatasetDict`]: The dataset with truncated sequences.

    Example:
    ```python
    >>> from datasets import Dataset

    >>> examples = {
    ...     "input_ids": [[1, 2, 3], [4, 5, 6, 7], [8]],
    ...     "attention_mask": [[0, 1, 1], [0, 0, 1, 1], [1]],
    ... }
    >>> dataset = Dataset.from_dict(examples)
    >>> truncated_dataset = truncate_dataset(dataset, max_length=2)
    >>> truncated_dataset[:]
    {'input_ids': [[1, 2], [4, 5], [8]],
     'attention_mask': [[0, 1], [0, 0], [1]]}
    ```
    """
    if map_kwargs is None:
        map_kwargs = {}
    if isinstance(dataset, Dataset):
        # Fast truncation with pyarrow
        def truncate(examples):
            truncated_columns = []
            for column in examples.columns:
                if pyarrow.types.is_list(column.type) or pyarrow.types.is_large_list(column.type):
                    column = pc.list_slice(column, 0, max_length)
                truncated_columns.append(column)
            return pyarrow.Table.from_arrays(truncated_columns, names=examples.column_names)

        dataset = dataset.with_format("arrow")
        dataset = dataset.map(truncate, batched=True, **map_kwargs)
        dataset = dataset.with_format(None)
    else:

        def truncate(examples):
            truncated_examples = {}
            for key, column in examples.items():
                if column and isinstance(column[0], list):
                    column = [val[:max_length] for val in column]
                truncated_examples[key] = column
            return truncated_examples

        dataset = dataset.map(
            truncate,
            batched=True,
            **map_kwargs,
        )
    return dataset

def _get_audio(example):
    audio = example["audio"]
    waveform = torch.tensor(audio["array"])
    sample_rate = audio["sampling_rate"]
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
    # print(torch.norm(torch.tensor(waveform)))
    # waveform = torch.zeros_like(waveform)
    return waveform.numpy()

def _get_message( example, for_generation=False):
    question_template = f"Based on the given audio, answer the speaker's question. Please think about this question as if you were a human pondering deeply. It's encouraged to include self-reflection or verification in the reasoning process. Output the thinking process in <think> </think> and final answer in <answer> </answer>."
    # print(question_template)
    MAX_SAMPLES = 16000 * 30
    if "response" in example.keys() and not for_generation:
        if  example["dataset_name"]=="safety":
            message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question_template},
                        {"type": "audio", "audio": example["audio"]["array"][:MAX_SAMPLES],"sampling_rate": example["audio"]["sampling_rate"]},
                        
                    ],
                },
                {   "role": "assistant", 
                    "content": [{"type": "text", "text": example["response"]}]
                }
                # {"role": "assistant", 
                # "content": "sorry, "}
            ]
        else:
            message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question_template},
                        {"type": "audio", "audio": example["audio"]["array"][:MAX_SAMPLES],"sampling_rate": example["audio"]["sampling_rate"]},
                    ],
                },
                {"role": "assistant", 
                "content": [{"type": "text", "text": example["response"]}],
                }
                # {"role": "assistant", 
                # "content":"sure, "}
                # {"role": "assistant", 
                # "content": "sorry, "}
            ]
    # else:
        # for tsne visualization purpose. This code should be deleted in production.
    # print("hi")
    # message = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {"type": "audio", "audio_url": None},
    #             {"type": "text", "text": question_template},
    #         ],
    #     },
    #     {"role": "assistant", 
    #     "content": "<think> </think> <answer> "}
    #     ]
    # logging.info(message)
    return message

def tokenize(example, processing_class, dataset_text_field, add_special_tokens,for_generation):
    # print(example["prompt"])
    text = _get_message(example, for_generation =for_generation )

    # chat_text = processing_class.apply_chat_template(text , add_generation_prompt=for_generation, tokenize=False)

    # processed = processing_class(
    #     text=chat_text, 
    #     audio=_get_audio(example),
    #     sampling_rate=16000,
    #     return_tensors="pt",
    #     padding=True
    # )

    processed = processing_class.apply_chat_template(
        text,
        tokenize=True,
        add_generation_prompt=for_generation,
        return_dict=True,
        output_labels=True,
    )

    if "feature_attention_mask" in processed:
        mask_name = "feature_attention_mask"
    elif "input_features_mask" in processed:
        mask_name = "input_features_mask"

    processed["input_ids"] = processed["input_ids"][0]
    processed["attention_mask"] = processed["attention_mask"][0]
    processed["input_features"] = processed["input_features"][0]
    processed[mask_name] = processed[mask_name][0]
    processed["dataset_name"] = example["dataset_name"]
    return processed

def add_eos(example, eos_token):
    if "text" in example and not example["text"].endswith(eos_token):  # language modeling case
        example["text"] = example["text"] + eos_token
    elif "completion" in example and not example["completion"].endswith(eos_token):
        example["completion"] = example["completion"] + eos_token
    return example

class AudioSFTTrainer(SFTTrainer):

    def __init__(self, original_dataset, processing_class, safety_mixture, for_generation=False,  *args, **kwargs):
        self.for_generation=for_generation
        super().__init__(processing_class=processing_class, *args, **kwargs)
        self.safety_mixture= safety_mixture
        # self.original_dataset =original_dataset
        # for param in self.model.audio_tower.parameters(): 
        #     param.requires_grad = False
        # for param in self.model.multi_modal_projector.parameters(): 
        #     # Assuming 'audio_encoder' is the correct name
        #     param.requires_grad = False
        original_dataset = self._prepare_dataset(original_dataset,processing_class, self.args, None, None, "gsm8k")
        self.original_dataloader = self.get_dataloader(original_dataset)
        self.original_data_iter = iter(self.original_dataloader)
        self.round = 0
        self.sum_similarity=0
        
    def get_dataloader(self,original_dataset) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
     
        from transformers.trainer_utils import (
            seed_worker
        )
        from transformers.trainer_pt_utils import (
        LengthGroupedSampler,
        )
        from torch.utils.data import DataLoader, RandomSampler
        data_collator = self.data_collator
  
        sampler = RandomSampler(original_dataset)
        
        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }
        # generator = torch.Generator()
        # generator.manual_seed(0)  # or use a fixed int
        # dataloader_params["generator"] = generator
        
        if not isinstance(original_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = sampler
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker

        return self.accelerator.prepare(DataLoader(original_dataset, **dataloader_params))

    def sample_from_original(self):
        # Get a  batch
        try:
            batch = next(self.original_data_iter)
        except (StopIteration):
            # If the iterator is exhausted, create a new iterator
            self.original_data_iter = iter(self.original_dataloader)
            batch = next(self.original_data_iter)
        return batch

    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]],num_items_in_batch
    ) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)
        def step(step_inputs):
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, step_inputs)
            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            self.accelerator.backward(loss)
                # print("gere2")
            return loss 

        loss = step(inputs)
        original_inputs = self.sample_from_original() 
        # step on gsm8k dataset
        loss2 = step(original_inputs)
        return (loss.detach()+loss2.detach() ) / self.args.gradient_accumulation_steps



    def _prepare_dataset(
        self,
        dataset: Union[Dataset, IterableDataset],
        processing_class: Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin],
        args: SFTConfig,
        packing: bool,
        formatting_func: Optional[Callable[[dict], str]],
        dataset_name: str
    ) -> Union[Dataset, IterableDataset]:
        # Convert the dataset to an IterableDataset if it is a ConstantLengthDataset
    
        # If the dataset is already preprocessed (tokenized), skip the processing steps.
        column_names = list(next(iter(dataset)).keys())
        is_processed = "input_ids" in column_names

        # Build the kwargs for the `map` function
        map_kwargs = {}
        if isinstance(dataset, Dataset):  # IterableDataset does not support num_proc
            map_kwargs["num_proc"] = args.dataset_num_proc
            print("num data processor {}".format(args.dataset_num_proc))
        with PartialState().main_process_first():
            # Apply the formatting function if any
            if formatting_func is not None and is_processed:
                warnings.warn(
                    "You passed a dataset that is already processed (contains an `input_ids` field) together with a "
                    "formatting function. Therefore `formatting_func` will be ignored. Either remove the "
                    "`formatting_func` or pass a dataset that is not already processed.",
                    UserWarning,
                )

            if formatting_func is not None and not is_processed:
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Applying formatting function to {dataset_name} dataset"

                def _func(example):
                    return {"text": formatting_func(example)}

                try:
                    dataset = dataset.map(_func, batched=False, **map_kwargs)
                except Exception as e:
                    warnings.warn(
                        f"Failed to apply the formatting function due to the following error: {e}. This may be "
                        "because the function is designed for batched input. Please update it to process one example "
                        "at a time (i.e., accept and return a single example). For now, we will attempt to apply the "
                        "function in batched mode, but note that batched formatting is deprecated and will be removed "
                        "in version 0.21.",
                        DeprecationWarning,
                    )
                    dataset = dataset.map(_func, batched=True, **map_kwargs)

            if not is_processed:
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Adding EOS to {dataset_name} dataset"

                dataset = dataset.map(
                    add_eos,
                    # keep_in_memory=True,
                    fn_kwargs={"eos_token": processing_class.eos_token_id},
                    remove_columns="messages" if "messages" in column_names else None,  # renamed to "text"
                    **map_kwargs,
                )
                # Subsequent tokenization will add special tokens (mostly for bos).
                # See https://huggingface.co/blog/qgallouedec/gotchas-in-tokenizer-behavior#7-chat-template-and-tokenization-dont-compose-due-to-special-tokens
                add_special_tokens = True

                # Tokenize the dataset
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Tokenizing {dataset_name} dataset"

                dataset = dataset.map(
                    tokenize,
                    keep_in_memory=True,
                    fn_kwargs={
                        "processing_class": processing_class,
                        "dataset_text_field": args.dataset_text_field,
                        "add_special_tokens": add_special_tokens,
                        "for_generation": self.for_generation
                    },
                    **map_kwargs,
                )

            # Pack or truncate
            if packing:
                if args.max_length is None:
                    raise ValueError("When packing is enabled, `max_length` can't be `None`.")
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Packing {dataset_name} dataset"
                dataset = dataset.select_columns("input_ids")
                dataset = pack_dataset(dataset, args.max_length, map_kwargs)
            # elif args.max_length is not None:
            #     if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
            #         map_kwargs["desc"] = f"Truncating {dataset_name} dataset"
            #     dataset = truncate_dataset(dataset, args.max_length, map_kwargs)
            # For Liger kernel, ensure only input_ids is present
            if args.use_liger_kernel:
                dataset = dataset.select_columns("input_ids")

        return dataset
    
    def loss_func(self, model_output, labels, num_items_in_batch,weighted=True):
        # Shift so that tokens < n predict n
        logits = model_output["logits"] if isinstance(model_output, dict) else model_output[0]
        
        logits  = logits[..., :-1, :].contiguous()
        labels  = labels[..., 1:].contiguous()
        ignore_index=-100
        epsilon= 0
        log_probs = -nn.functional.log_softmax(logits , dim=-1)
  
        if labels.dim() == log_probs.dim() - 1:
            labels = labels.unsqueeze(-1)
        # loss_fct = torch.nn.CrossEntropyLoss(reduction="none") # Important: reduction='none'
        padding_mask = labels.eq(ignore_index)
        # In case the ignore_index is -100, the gather will fail, so we replace labels by 0. The padding_mask
        # will ignore them in any case.
        labels = torch.clamp(labels, min=0)
        nll_loss = log_probs.gather(dim=-1, index=labels)
        # works for fp16 input tensor too, by internally upcasting it to fp32
        smoothed_loss = log_probs.sum(dim=-1, keepdim=True, dtype=torch.float32)

        # print(nll_loss.shape)
        nll_loss.masked_fill_(padding_mask, 0.0)
        smoothed_loss.masked_fill_(padding_mask, 0.0)
        
        # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
        num_active_elements = padding_mask.numel() - padding_mask.long().sum()
       
        nll_loss = nll_loss.sum() / num_active_elements
        smoothed_loss = smoothed_loss.sum() / (num_active_elements * log_probs.shape[-1])
        loss = (1 - epsilon) * nll_loss + epsilon * smoothed_loss
        return loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, weighted=True, return_hidden_state=False):
        """
        Compute training loss and additionally compute token accuracies
        """
        mode = "train" if self.model.training else "eval"
        (loss, outputs) = self.compute_weighted_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch,
             weighted=weighted, return_hidden_state=return_hidden_state
        )
        # for key in inputs:
        #     print(key)
        if mode == "train":
            # When using padding-free, the attention_mask is not present in the inputs, instead we have cu_seq_lens_q,
            # cu_seq_lens_k, and max_length_k, max_length_q and position_ids.
            if "attention_mask" in inputs:
                num_tokens_in_batch = self.accelerator.gather_for_metrics(inputs["attention_mask"].sum()).sum().item()
            elif "position_ids" in inputs:
                local_num_tokens = torch.tensor(inputs["position_ids"].size(1), device=inputs["position_ids"].device)
                num_tokens_in_batch = self.accelerator.gather_for_metrics(local_num_tokens).sum().item()
            else:
                raise ValueError("Expected 'attention_mask' or 'position_ids' in inputs.")
            self._total_train_tokens += num_tokens_in_batch
        self._metrics[mode]["num_tokens"] = [self._total_train_tokens]

        # Compute token accuracy if we have labels and if the model is not using Liger (no logits)
        # ignore_index: int = -100
        if "labels" in inputs and not self.args.use_liger_kernel:
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = inputs["labels"][..., 1:].contiguous()

            # Get predictions
            predictions = shift_logits.argmax(dim=-1)

            # Create mask for non-padding tokens (assuming ignore_index is -100)
            mask = shift_labels != -100

            # Calculate accuracy only on non-padding tokens
            correct_predictions = (predictions == shift_labels) & mask
            total_tokens = mask.sum()
            correct_tokens = correct_predictions.sum()

            # Gather the correct_tokens and total_tokens across all processes
            correct_tokens = self.accelerator.gather_for_metrics(correct_tokens)
            total_tokens = self.accelerator.gather_for_metrics(total_tokens)

            # Compute the mean token accuracy and log it
            total_sum = total_tokens.sum()
            accuracy = (correct_tokens.sum() / total_sum).item() if total_sum > 0 else 0.0
            self._metrics[mode]["mean_token_accuracy"].append(accuracy)

        return (loss, outputs) if return_outputs else loss

    def compute_weighted_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, weighted=True,return_hidden_state=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        # for key in inputs:
        #     print(key)
        if "wave" in inputs.keys():
            inputs.pop("wave")
        # labels = inputs.pop("labels")
        # torch.set_printoptions(threshold=10000, linewidth=200, edgeitems=50) # Adjust values as needed
        # print(inputs["input_ids"], flush=True)
        # dataset_name = inputs.pop("dataset_name")
        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        # if self.args.past_index >= 0:
        #     self._past = outputs[self.args.past_index]

        # if labels is not None:
        unwrapped_model = self.accelerator.unwrap_model(model)
        # if _is_peft_model(unwrapped_model):
        #     model_name = unwrapped_model.base_model.model._get_name()
        # else:
        model_name = unwrapped_model._get_name()
        # User-defined compute_loss function
        loss = self.loss_func(outputs, inputs["labels"], num_items_in_batch=num_items_in_batch, weighted=weighted)


        if (
            self.args.average_tokens_across_devices
            and (self.model_accepts_loss_kwargs or self.compute_loss_func)
            and num_items_in_batch is not None
        ):
            loss *= self.accelerator.num_processes
        
        return (loss, outputs) if return_outputs else loss