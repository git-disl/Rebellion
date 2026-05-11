import os
from absl import app
from absl import flags
from absl import logging
import torch
from transformers import AutoProcessor, HfArgumentParser, Qwen2AudioForConditionalGeneration
import json
from dataclasses import dataclass, field
import torchaudio
from tqdm import tqdm
import datasets
import random
import numpy as np

_OUTPUT_DIR = flags.DEFINE_string('output_dir', "", 'output dir')
_DATA_DIR = flags.DEFINE_string('data_dir', None, 'data directory')
_CACHE_DIR = flags.DEFINE_string('cache_dir', None, 'cache directory')
_MODEL_DIR = flags.DEFINE_string('model_dir', None, 'data directory')
_BATCH_SIZE = flags.DEFINE_integer('batch_size', 128, 'batch size')
_WRITE_METRICS = flags.DEFINE_boolean(
    'write_metrics', False, 'Use tensorboard to visualize metrics'
)
_FAIL_ON_CPU = flags.DEFINE_boolean(
    'fail_on_cpu', False, 'fail if not run on GPU'
)
_REWARD_FORMAT = flags.DEFINE_string(
    'reward_format', "cot", ''
)

def _get_audio(example):
    audio = example["audio"]
    waveform = torch.tensor(audio["array"])
    sample_rate = audio["sampling_rate"]
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
    return waveform


def _get_message(obj_dict):

    if _REWARD_FORMAT.value!= "cot":
        question_template = f"Based on the given audio, answer the speaker's question. Output the final answer in <answer> </answer>."
    else:
    # If you want to improve the thinking process, uncomment the next line and design your strategy.
        question_template = f"Based on the given audio, answer the speaker's question. Please think about this question as if you were a human pondering deeply. It's encouraged to include self-reflection or verification in the reasoning process. Output the thinking process in <think> </think> and final answer in <answer> </answer>."
    
    # question_template = f"{choice_str} Output the final answer in <answer> </answer>."

    # print(question_template)
    message = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": obj_dict['audio']["path"]},
                {"type": "text", "text": question_template},
            ],
        }
    ]
    # logging.info(message)
    return message

def main(_):
    seed=1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # For multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    if _CACHE_DIR.value!=None:
        os.environ["HF_DATASETS_CACHE"] = _CACHE_DIR.value
        os.environ["HF_DATASETS_OFFLINE"] = "1"
    output_folder = os.path.dirname(_OUTPUT_DIR.value)
    os.makedirs(output_folder, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"CUDA available: {torch.cuda.is_available()}")
    audio_processor = AutoProcessor.from_pretrained(_MODEL_DIR.value)
    # logging.info("hihiaa")
    audio_model = Qwen2AudioForConditionalGeneration.from_pretrained(_MODEL_DIR.value).to(device)
    all_outputs = []
    # logging.info("sssssas")
    if _CACHE_DIR.value!=None:
        dataset =datasets.load_dataset(_DATA_DIR.value,cache_dir=_CACHE_DIR.value)
    else:
        dataset =datasets.load_dataset(_DATA_DIR.value)
    # logging.info("sssss")
    index=0
    datas = []
    for example in dataset["train"]:
        instance = {}
        instance["instruction"] = example["prompt"]
        instance["audio"] = example["audio"]
        datas += [instance]

    logging.info("hihiaa")
    # datas= [{"audio_id": "beavertail_audio_{}.wav".format(i) }  for i in range(1000)]

    batch_size = 20
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
        generated_ids = audio_model.generate(**inputs, max_new_tokens=100000)
        generated_ids = generated_ids[:, inputs.input_ids.size(1) :]
        batch_response = audio_processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        for j in range(len(batch_response)):
            dictionary={}
            dictionary['instruction']= batch_data[j]['instruction']
            dictionary['full_output'] = batch_response[j]
              # --- START OF CHANGES ---
            full_text_response = batch_response[j]
            answer_tag = "<answer>"
            answer_start_index = full_text_response.find(answer_tag)

            if answer_start_index != -1:
                # If <answer> is found, extract everything after it
                extracted_answer = full_text_response[answer_start_index + len(answer_tag):].strip()
                dictionary['output'] = extracted_answer
            else:
                # If <answer> is not found, you can decide what to do:
                # Option 1: Store the entire response
                dictionary['output'] = full_text_response
                # Option 2: Store an empty string or a placeholder
                # dictionary['output'] = ""

            all_outputs+= [ dictionary]
        logging.info(batch_response)
    with open(_OUTPUT_DIR.value, 'w') as f:
        json.dump(all_outputs, f, indent=4)


if __name__ == '__main__':
  app.run(main)
