import logging
from dataclasses import dataclass, field
from typing import Optional

import transformers
from transformers import AutoProcessor, HfArgumentParser, Qwen2AudioForConditionalGeneration
from trl import GRPOConfig

from trainer import GRPOTrainer
from rewards import accuracy_reward, format_reward,safety_reward,cot_format_reward
# from dataset import AudioDataset
# from sft_dataset import AudioDataset
import random
import numpy as np
import torch 
from tqdm import tqdm
from absl import app
from absl import flags
from absl import logging
import datasets
from datasets import Dataset, Audio

_DATA_DIR = flags.DEFINE_string('data_file', None, 'data directory')
_MODEL_DIR = flags.DEFINE_string('model_name_or_path', None, 'data directory')


def _get_audio(example):
    audio = example["audio"]
    waveform = torch.tensor(audio["array"])
    sample_rate = audio["sampling_rate"]
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
    return waveform

def _get_message(obj_dict):
    question_template = f"Based on the given audio, answer the speaker's question. Please think about this question as if you were a human pondering deeply. It's encouraged to include self-reflection or verification in the reasoning process. Output the thinking process in <think> </think> and final answer in <answer> </answer>."
    # print(question_template)
    message = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": None},
                {"type": "text", "text": question_template},
            ],
        }
    ]
    # logging.info(message)
    return message

def main(_):
    from huggingface_hub import login
    # You might need to log in first if you haven't already
    login(token="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")
    seed=0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # For multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # parser = HfArgumentParser(DataTrainingArguments)
    # data_args = parser.parse_args_into_dataclasses()[0]
    # logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # transformers.logging.set_verbosity_info()
    # logging.info(data_args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"CUDA available: {torch.cuda.is_available()}")
    audio_processor = AutoProcessor.from_pretrained(_MODEL_DIR.value)
    # logging.info("hihiaa")
    audio_model = Qwen2AudioForConditionalGeneration.from_pretrained(_MODEL_DIR.value).to(device)
    dataset =datasets.load_dataset(_DATA_DIR.value)
    # logging.info("sssss")
    index=0
    datas = []
    for example in dataset["train"]:
        instance = {}
        instance["instruction"] = example["prompt"]
        instance["audio"] = example["audio"]
        datas += [instance]

    batch_size = 1
    # for i in tqdm(range(0, len(datas), batch_size)):
    for i in tqdm(range(0, len(datas), batch_size)):
        batch_data = datas[i : i + batch_size]
        batch_messages = []
        batch_audios = []
        for bd in batch_data:
            batch_messages.append(_get_message(bd))
            batch_audios.append(_get_audio(bd).numpy())

        text = [
            audio_processor.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
            for msg in batch_messages
        ]
        inputs = audio_processor(
            text=text, audios=batch_audios, sampling_rate=16000, return_tensors="pt", padding=True
        ).to(audio_model.device)
        # print(inputs)
        generated_ids = audio_model.generate(**inputs, max_new_tokens=1024)
        batch_response = audio_processor.batch_decode(
            generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        
        for j in range(len(batch_response)):
            batch_data[j]["text"] = batch_response[j]

    dataset = Dataset.from_list(datas).cast_column("audio", Audio())
    distill_dataset_name = _MODEL_DIR.value.rsplit('/', 1)[-1]
    dataset.push_to_hub(distill_dataset_name)
   


if __name__ == "__main__":
    # from huggingface_hub import login
    # login(token="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")
    app.run(main)