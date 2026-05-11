import json
import logging
import torchaudio
from torch.utils.data import Dataset
import datasets
import torch
import trl 

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

class CustomDataCollatorForSFT(trl.DataCollatorForCompletionOnlyLM):
    def __call__(self, features):
        # Separate 'source' from the features
        # Assuming 'features' is a list of dicts, one dict per sample
        for feature in features:
            feature.pop("prompt")
            feature.pop("response")
            feature.pop("audio")
            # feature.pop("input_features")
            # feature.pop("feature_attention_mask")

        dataset_names = [feature.pop("dataset_name") for feature in features]
        
        input_features = torch.tensor(  [feature.pop("input_features") for feature in features])
        feature_attention_mask =  torch.tensor( [feature.pop("feature_attention_mask") for feature in features])
        flag=False
        if "wave" in features[0].keys():
            flag=True
            waves =   [ torch.tensor(feature.pop("wave")) for feature in features]
        
        batch = super().__call__(features)            
        # print(len(batch))
        batch["dataset_name"] = dataset_names
        batch["input_features"] = input_features
        batch["feature_attention_mask"] = feature_attention_mask
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


    def _handle_avqa(self, obj_avqa):
        choice_str = f"Please choose the answer from the following options: {obj_avqa['multi_choice']}."
        if self.reward_format!="cot":
            question_template = f"{obj_avqa['question_text'].replace('video', 'audio')} {choice_str} Output the final answer in <answer> </answer>."
            # If you want to improve the thinking process, uncomment the next line and design your strategy.
        else:
            question_template = f"{obj_avqa['question_text'].replace('video', 'audio')} {choice_str} Please think about this question as if you were a human pondering deeply. It's encouraged to include self-reflection or verification in the reasoning process. Output the thinking process in <think> </think> and final answer in <answer> </answer>."
        obj_avqa["prompt"] = [{"role": "user", "content": [{"type": "audio", "audio_url": None}, {"type": "text", "text": question_template}]}]
        answer_str = obj_avqa["multi_choice"][obj_avqa["answer"]]
        obj_avqa["solution"] = f"<answer>{answer_str}</answer>"
        # print(obj_avqa)
        return obj_avqa


    def handle_json_line(self, obj, sample_rate=16000):
        if obj["dataset_name"] == "gsm8k":
            if self.reward_format!="cot":
                question_template = f"Based on the given audio, answer the speaker's question. Output the final answer in <answer> </answer>."
            else:
                question_template = f"Based on the given audio, answer the speaker's question. Please think about this question as if you were a human pondering deeply. It's encouraged to include self-reflection or verification in the reasoning process. Output the thinking process in <think> </think> and final answer in <answer> </answer>."
            obj["prompt"] = [{"role": "user", "content": [{"type": "audio", "audio_url": None}, {"type": "text", "text": question_template}]}]
            waveform = torch.tensor(obj["audio"]["array"])
            obj["audio"] = waveform.numpy()
            return obj
        else:
            waveform = _handle_wav(obj["audio_path"], sample_rate)
            obj["audio"] = waveform.numpy()    
            return self._handle_avqa(obj)

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