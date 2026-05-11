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
from vllm import LLM, SamplingParams

_OUTPUT_DIR = flags.DEFINE_string('output_dir', "", 'output dir')
_DATA_DIR = flags.DEFINE_string('data_dir', None, 'data directory')
_CACHE_DIR = flags.DEFINE_string('cache_dir', None, 'cache directory')
_MODEL_DIR = flags.DEFINE_string('model_dir', None, 'data directory')
_BATCH_SIZE = flags.DEFINE_integer('batch_size', 128, 'batch size')
_SYSTEM_PROMPT = flags.DEFINE_string('system_prompt', "Based on the given audio, answer the speaker's question. Please think about this question as if you were a human pondering deeply. It's encouraged to include self-reflection or verification in the reasoning process. Output the thinking process in <think> </think> and final answer in <answer> </answer>", 'system prompt')
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
    waveform = torch.tensor(audio["array"], dtype=torch.float32)
    sample_rate = audio["sampling_rate"]
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)

    # Define the target duration in seconds
    max_duration_s = 30
    
    # Calculate the number of samples for the target duration
    # We use the final sample rate (16000) for this calculation
    max_samples = int(max_duration_s * 16000)
    
    # Truncate the waveform if its length exceeds the maximum number of samples
    if waveform.shape[-1] > max_samples:
        waveform = waveform[..., :max_samples]
    return waveform


def _get_message(obj_dict):
    message = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": obj_dict['audio']["path"]},
                {"type": "text", "text": _SYSTEM_PROMPT.value},
            ],
        }
    ]
    # logging.info(message)
    return message



def query(model,audio_processor,sampling_params,  inputs):
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            sampling_params,
        )
        
    res_list = []
    # print(outputs)
    for request in outputs:
        res = request.outputs[0].text.strip()
        # print(res)
        # res = audio_processor.batch_decode(
        #     request.outputs[0].token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        # )
        # print(res)
        res_list+=[res]
    return res_list


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
    
    llm = LLM(
        model=_MODEL_DIR.value, trust_remote_code=True, gpu_memory_utilization=0.98,
        enforce_eager=True,     # Disable CUDA graph, force call forward in every decode step.
        limit_mm_per_prompt={"audio": 5},
    )
    sampling_params = SamplingParams(
        temperature=0.7, top_p=0.01, top_k=1, repetition_penalty=1.1, max_tokens=2048,
        stop_token_ids=[],
    )



    all_outputs = []
    # logging.info("sssssas")
    if _CACHE_DIR.value!=None:
        if "JALMBench" in _DATA_DIR.value: 
            dataset =datasets.load_dataset("AnonymousUser000/JALMBench","AdvWave", cache_dir=_CACHE_DIR.value)
        else:
            dataset =datasets.load_dataset(_DATA_DIR.value,cache_dir=_CACHE_DIR.value)
    else:
        if "JALMBench" in _DATA_DIR.value: 
            dataset =datasets.load_dataset("AnonymousUser000/JALMBench","AdvWave")
        else:
            dataset =datasets.load_dataset(_DATA_DIR.value)
    # logging.info("sssss")
    index=0
    datas = []
    for example in dataset["train"]:
        # if index<100:
        instance = {}
        if "original_text" in example.keys(): 
            instance["prompt"] = example["original_text"]
        else:
            instance["prompt"] = example["prompt"]
        instance["audio"] = example["audio"]
        if "response" in example.keys():
            instance["ground_truth"] = example["response"]
        datas += [instance]
        index+=1

    # datas= [{"audio_id": "beavertail_audio_{}.wav".format(i) }  for i in range(1000)]
    
    batch_size = 5000
    
    for i in tqdm(range(0, len(datas), batch_size)):
        input_list = []
        batch_data = datas[i : i + batch_size]
        batch_messages = []
        batch_audios = []
        for bd in batch_data:
            batch_messages.append(_get_message(bd))
            batch_audios.append((_get_audio(bd).numpy(),16000))
    
        text = [
            audio_processor.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
            for msg in batch_messages
        ]
        
        inputs = [
            {
                'prompt': text[i],
                'multi_modal_data': {
                    'audio': batch_audios[i]
                }
            } for i in range(len(text))
        ]

        input_list+= inputs

        response = query(llm, audio_processor, sampling_params, input_list)
            
        for j in range(len(response)):
            dictionary={}
            dictionary['prompt']= batch_data[j]['prompt']
            dictionary['full_output'] = response[j]
            if "ground_truth" in batch_data[j].keys():
                dictionary["ground_truth"] = batch_data[j]['ground_truth']
                
            full_text_response = response[j]
            answer_tag = "<answer>"
            answer_start_index = full_text_response.find(answer_tag)

            if answer_start_index != -1:
                # If <answer> is found, extract everything after it
                extracted_answer = full_text_response[answer_start_index + len(answer_tag):].strip()
                dictionary['output'] = extracted_answer
            else:
                dictionary['output'] = full_text_response

            all_outputs+= [ dictionary]
    with open(_OUTPUT_DIR.value, 'w') as f:
        json.dump(all_outputs, f, indent=4)


if __name__ == '__main__':
  app.run(main)
