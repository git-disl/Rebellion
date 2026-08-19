import os.path
from io import BytesIO
from urllib.request import urlopen
from typing import List, Optional, Union
import librosa
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor,AudioFlamingo3ForConditionalGeneration
import torch
import numpy
import torch.nn as nn
from tqdm import tqdm
import torchaudio
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from transformers import HfArgumentParser
from datasets import Dataset, Audio
import torch.nn.functional as F
from transformers.feature_extraction_utils import BatchFeature
import logging
from torch.optim import lr_scheduler

MAX_SAMPLES = 16000 * 30

def get_input_embeds(model, input_ids, input_features, input_features_mask, attention_mask, labels):
    # 1. Extract the input embeddings
    def get_audio_features(
        input_features: torch.FloatTensor,
        input_features_mask: torch.Tensor,
    ):
        audio_output = model.model.audio_tower(input_features, input_features_mask=input_features_mask, return_dict=True)
        audio_embeds = model.model.multi_modal_projector(audio_output.last_hidden_state)
        input_lengths = input_features_mask.sum(-1).to(torch.long)
        _, post_lengths = model.model.audio_tower._get_feat_extract_output_lengths(input_lengths)
        valid_mask = torch.arange(audio_embeds.shape[1], device=post_lengths.device)[None, :] < post_lengths[:, None]
        audio_output.pooler_output = audio_embeds[valid_mask.to(audio_embeds.device)]

        return audio_output


    inputs_embeds = model.get_input_embeddings()(input_ids)
    if input_features is not None and input_ids is not None:
        input_features = input_features.to(
            dtype=model.model.audio_tower.conv1.weight.dtype, device=model.model.audio_tower.conv1.weight.device
        )
        input_features_mask = input_features_mask.to(
            dtype=model.model.audio_tower.conv1.weight.dtype, device=model.model.audio_tower.conv1.weight.device
        )
        audio_embeds = get_audio_features(input_features, input_features_mask).pooler_output

        # replace text-audio token placeholders with audio embeddings
        special_audio_mask = model.model.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, audio_features=audio_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(
            special_audio_mask, audio_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        )

    
    return inputs_embeds

def qwen_eval_gen(audio_list, processor, model):
    if len(audio_list) > 1:
        raise ValueError("Only consider single audio clip here!")
    audio_url = "file:" + audio_list[0]
    conversation = [
        {"role": "user", "content": [
            {"type": "audio", "audio_url": audio_url},
        ]},
    ]
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    print(text)
    audios = []
    for message in conversation:
        if isinstance(message["content"], list):
            for ele in message["content"]:
                if ele["type"] == "audio":
                    audios.append(librosa.load(
                        BytesIO(urlopen(ele['audio_url']).read()),
                        sr=processor.feature_extractor.sampling_rate)[0]
                                  )

    for idx in range(len(audios)):
        audios[idx] = torch.tensor(audios[idx], device="cuda")
    audios = torch.cat(audios).unsqueeze_(0)

    # Inference
    inputs = processor(text=text, audio=audios, return_tensors="pt" , sampling_rate=16000)
    inputs = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    generate_ids = model.generate(**inputs, max_length=1024, do_sample=False, temperature=0.0, top_p=0, top_k=0)
    generate_ids = generate_ids[:, inputs['input_ids'].size(1):]
    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return response

def get_adv_targets(text, targets, model, processor, audios):
    target_ind = -1
    best_loss = 1e10
    for idx in range(len(audios)):
        audios[idx] = torch.tensor(audios[idx], device="cuda")
    audios = torch.cat(audios).unsqueeze_(0)

    for ind, target_text in enumerate(targets):
        target_ids = processor(text=target_text, return_tensors="pt")["input_ids"].to("cuda")
        inputs = processor(text=text + target_text, audios=audios, return_tensors="pt",
                           sampling_rate=16000)
        inputs["input_ids"] = inputs["input_ids"].to("cuda")
        inputs["attention_mask"] = inputs["attention_mask"].to("cuda")
        output = model(model_inputs, inputs_embeds=inputs_embeds)
        logits = output.logits
        # Shift logits so token n-1 predicts token n
        shift = inputs_embeds.shape[1] - target_ids.shape[1]
        shift_logits = logits[..., shift - 1:-1, :].contiguous()  # (1, num_target_ids, vocab_size)
        shift_labels = target_ids
        loss = torch.nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        if loss < best_loss:
            target_ind = ind
            best_loss = loss.item()
        del output
        del loss
        torch.cuda.empty_cache()
    return targets[target_ind]


def differentiableWhisper(raw_speech, feature_extractor, truncation: bool = True,
    pad_to_multiple_of: Optional[int] = None,
    return_tensors: Optional[str] = 'pt',
    return_attention_mask: Optional[bool] = None,
    padding: Optional[str] = "max_length",
    max_length: Optional[int] = None,
    sampling_rate: Optional[int] = None,
    do_normalize: Optional[bool] = None,
    device: Optional[str] = "cpu",
    return_token_timestamps: Optional[bool] = None, ):

        # Padding
        padding = 480000 - raw_speech.size(1)
        raw_speech = F.pad(raw_speech, (0, padding), "constant", 0)
        padded_inputs = BatchFeature({"input_features": raw_speech.unsqueeze_(0)})

        padded_inputs["attention_mask"] = torch.zeros([1, 480000], dtype=torch.int, device="cuda")
        padded_inputs["attention_mask"][:,:480000 - padding] = 1

        # Zero-mean and unit-variance normalization
        if do_normalize:
            padded_inputs["input_features"] = feature_extractor.zero_mean_unit_var_norm(
                padded_inputs["input_features"],
                attention_mask=padded_inputs.get("attention_mask"),
                padding_value=feature_extractor.padding_value,
            )
            # Already tensors, no need to stack

        # Prepare input features
        input_features = padded_inputs["input_features"]  # Already a tensor

        def _torch_extract_fbank_features( waveform: torch.Tensor, extractor,  device: str = "cpu"):
            """
            Compute the log-mel spectrogram of the audio using PyTorch's GPU-accelerated STFT implementation with batching,
            yielding results similar to cpu computing with 1e-5 tolerance.
            """
            # waveform = torch.from_numpy(waveform).type(torch.float32)
            window = torch.hann_window(extractor.n_fft).to("cuda")
            stft = torch.stft(waveform, extractor.n_fft, feature_extractor.hop_length, window=window, return_complex=True,pad_mode="constant")
            magnitudes = stft[..., :-1].abs() ** 2
            mel_filters = torch.from_numpy(extractor.mel_filters).type(torch.float32).to("cuda")
            mel_spec = mel_filters.T @ magnitudes
            log_spec = torch.clamp(mel_spec, min=1e-10).log10()
            if waveform.dim() == 2:
                max_val = log_spec.max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0]
                log_spec = torch.maximum(log_spec, max_val - 8.0)
            else:
                log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
            log_spec = (log_spec + 4.0) / 4.0
            return log_spec

        extract_fbank_features = (
                _torch_extract_fbank_features
            )
        input_features = extract_fbank_features(input_features[0], feature_extractor, device)




        if isinstance(input_features[0], List):
            padded_inputs["input_features"] = [np.asarray(feature, dtype=np.float32) for feature in input_features]

        else:
            padded_inputs["input_features"] = input_features

        if return_attention_mask:
            # rescale from sample (48000) to feature (3000)
            padded_inputs["attention_mask"] = padded_inputs["attention_mask"][:, :: feature_extractor.hop_length]

        if return_token_timestamps is not None:
            padded_inputs["num_frames"] = [len(raw_speech_i) // feature_extractor.hop_length for raw_speech_i in raw_speech]

        # if return_tensors is not None:
        #     padded_inputs = padded_inputs.convert_to_tensors(return_tensors)

        return padded_inputs


def qwen_jailbreak_gen(index, audio_list, processor, model, audio_save_path, num_token_suffix, num_epochs, advnoise_control=False, control_obj="", model_judge=None, noise_initial=None, targets=None):
    if len(audio_list) > 1:
        raise ValueError("Only consider single audio clip here!")
    audio_url = "file:" + audio_list[0]
    question_template = f"Based on the given audio, answer the speaker's question. Please think about this question as if you were a human pondering deeply. It's encouraged to include self-reflection or verification in the reasoning process. Output the thinking process in <think> </think> and final answer in <answer> </answer>."
    conversation = [
        {"role": "user", "content": [
            {"type": "audio", "audio_url": audio_url},
            {"type": "text", "text": question_template}
        ]},
    ]
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

    
    audios = []
    for message in conversation:
        if isinstance(message["content"], list):
            for ele in message["content"]:
                if ele["type"] == "audio":
                    audios.append(librosa.load(
                        BytesIO(urlopen(ele['audio_url']).read()),
                        sr=processor.feature_extractor.sampling_rate)[0][:MAX_SAMPLES]
                                  )
                    # print(processor.feature_extractor.sampling_rate)
    # Load the CSV file
    file_path = 'harmful_behaviors.csv'  # Replace with the path to your CSV file
    data = pd.read_csv(file_path)
    for i, row in data.iterrows():
        row = row.to_dict()
        if i == index:
            target_text =  "<think> </think> <answer> "+ row['target']

    # print(f'Target response: {target_text}')

    target_ids = processor(text=target_text, return_tensors="pt")["input_ids"].to("cuda")


    adv_length = num_token_suffix
    std = 0.01
    # if not os.path.exists(noise_initial):
    xx = torch.randn([adv_length], device="cuda")
    adv_audio_suffix = xx * std
    # else:
    #     adv_audio_suffix = torchaudio.load(noise_initial)[0][0,:].to("cuda")
        # print(adv_audio_suffix.shape)
        # exit()
    adv_audio_suffix.requires_grad_(True)

    for idx in range(len(audios)):
        audios[idx] = torch.tensor(audios[idx], device="cuda", requires_grad=True)

    audios_original = audios.copy()

    audios.append(adv_audio_suffix)

    # audios = torch.cat(audios).unsqueeze_(0)
    audios = torch.cat(audios)
    audios.requires_grad_(True)

    pbar = tqdm(range(num_epochs))
    optimizer = torch.optim.Adam([adv_audio_suffix], lr=5e-4)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=num_epochs, gamma=0.01)
    losses = []


    for i in pbar:
        optimizer.zero_grad()
        audios_here = audios_original.copy()
        audios_here.append(adv_audio_suffix)
        # audios_here = torch.cat(audios_here).unsqueeze_(0)
        audios_here = torch.cat(audios_here).unsqueeze_(0)
        # print(audios_here)
        audios_here.requires_grad_(True)
        # print(audios_here.shape)
        inputs = processor(text=text + target_text, audio=audios_here[0].detach().cpu().numpy(), return_tensors="pt", sampling_rate=16000)
        # print(inputs)
        inputs["input_ids"] = inputs["input_ids"].to("cuda")
        # print(inputs["input_ids"])
        inputs["attention_mask"] = inputs["attention_mask"].to("cuda")
        inputs["input_features_mask"] = inputs["input_features_mask"].to("cuda")

        # print(torch.norm(inputs["input_features"]), flush=True)
        # print(inputs["input_features"].shape, flush=True)


        
        # window_size = int(16000* processor.feature_extractor.chunk_length)
      

        # per_sample_windows: list[int] = []
        # flat_chunks: list[np.ndarray] = []
        # for audio_el in audios_here:
        #     n_samples = int(audio_el.shape[0])
        #     n_win = max(1, (n_samples + window_size - 1) // window_size)
        #     per_sample_windows.append(n_win)

        #     time_cap = min(n_samples, n_win * window_size)
        #     for i in range(n_win):
        #         start = i * window_size
        #         end = min((i + 1) * window_size, time_cap)
        #         flat_chunks.append(audio_el[start:end])

        # # print(processor.feature_extractor)
        # print(flat_chunks)
        differentiable_inputs = differentiableWhisper(audios_here, processor.feature_extractor)
        # print(audio_inputs)
        inputs["input_features"] = differentiable_inputs ["input_features"]
        inputs["input_features"] = inputs["input_features"].to("cuda")
        print(torch.norm(inputs["input_features"]), flush=True)
        print(inputs["input_features"].shape, flush=True)
        with torch.no_grad():
            inputs_embeds = get_input_embeds(model.to("cuda"), inputs["input_ids"], inputs["input_features"],
                                            inputs["input_features_mask"], inputs["attention_mask"], None)
        model = model.to("cuda")
        inputs = {
            k: v.to(device=model.device, dtype=model.dtype) if torch.is_floating_point(v) 
            else v.to(device=model.device)
            for k, v in inputs.items()
        }
        output = model(**inputs,output_attentions=True)
        averaged_tensor= 0
        for i in range(len(output.attentions)):
        # for i in range(10):
            averaged_tensor += torch.mean(output.attentions[i], dim=1)
        # averaged_tensor/=10
        averaged_tensor/=len(output.attentions)
        tensor_2d = torch.squeeze(averaged_tensor, dim=0)
        # print(torch.norm(tensor_2d[:,:19]))
        logits = output.logits
        # Shift logits so token n-1 predicts token n
        shift = inputs_embeds.shape[1] - target_ids.shape[1]
        shift_logits = logits[..., shift - 1:-1, :].contiguous()  # (1, num_target_ids, vocab_size)
        shift_labels = target_ids
        loss = torch.nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


        try:
            loss.backward()
        except RuntimeError as e:
            print(f"Caught a RuntimeError: {e}")
            import traceback
            traceback.print_exc() # Print the full traceback

        # adv_audio_suffix.grad = grad1
        loss2 = None
        optimizer.step()
        norm_val = torch.norm(adv_audio_suffix) # Calculates the L2 norm
        max_norm = 10
        if norm_val >= max_norm:
            adv_audio_suffix = adv_audio_suffix / norm_val * max_norm

        if loss2 == None:
            loss2 = 0
        else:
            loss2 = loss2.detach().cpu().item()

        loss = loss.detach().cpu().item()

        pbar.set_description(f'Loss: {loss+loss2}')
        losses.append(loss + loss2)
        # if losses[-1] < 0.1:
        #     break



    if len(losses)>0:
        print("final loss {}".format(losses[-1]))
    audios_here = audios_original.copy()
    audios_here.append(adv_audio_suffix)
    audios_here = torch.cat(audios_here).unsqueeze_(0)
    output_dir = os.path.dirname(audio_save_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True) # exist_ok=True prevents an error if the directory already exists
    # print(audios_here.shape)
    torchaudio.save(audio_save_path, audios_here.detach().cpu(), 16000)
    torch.cuda.empty_cache()
    return None, None
    # return response, record

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    model_name_or_path : Optional[str] = field(default="", metadata={"help": "model name or path"})
    num_epochs: Optional[int] = field(default=500, metadata={"help": "adv train epoch"})
    num_token_suffix: Optional[int] = field(default=200000, metadata={"help": "num token suffix"})


def train_adv(model_name,data_args, begin_index, end_index):

    model = AudioFlamingo3ForConditionalGeneration.from_pretrained(data_args.model_name_or_path,torch_dtype=torch.bfloat16,attn_implementation="eager")
    
    
    processor = AutoProcessor.from_pretrained(data_args.model_name_or_path)
    # processor.pad_token_id = processor.tokenizer.pad_token_id
    processor.eos_token_id = processor.tokenizer.eos_token_id

    audio_list = ["../../data/advbench/original/"+str(i) +".wav" for i in range(520)]
    
    for id, audio_clip in tqdm(enumerate(audio_list)):
        if begin_index<=id<end_index:
            response, record= qwen_jailbreak_gen(id, [audio_list[id]], processor, model, "../../data/advbench/advwave/"+model_name+"_"+str(data_args.num_token_suffix)+"_"+str(data_args.num_epochs)+"/"+ str(id)+".wav", data_args.num_token_suffix, data_args.num_epochs)
            # response= qwen_eval_gen([audio_list[id]], processor, model)
            # print(response)
        
def hf_save_adv(model_name, data_args, begin_index, end_index):
    from datasets import load_dataset
    from huggingface_hub import login
    # You might need to log in first if you haven't already
    login(token="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") # Uncomment this and follow the prompts if needed
    index=0
    data= []
    file_path = 'harmful_behaviors.csv'  # Replace with the path to your CSV file
    original_data = pd.read_csv(file_path)
    instructions = []
    for i, row in original_data.iterrows():
        row = row.to_dict()
        target_text = row['goal']
        instructions += [target_text]
    for i in range(begin_index, end_index):
        audio_path = "../../data/advbench/advwave/"+model_name+"_"+str(data_args.num_token_suffix)+"_"+str(data_args.num_epochs)+"/"+ str(i)+".wav"
        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != 16000:
            # print(target_rate)
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)[0].numpy()
        else:
            waveform = waveform.squeeze(0).numpy()
        data.append({"prompt": instructions[i], "audio": {"array": waveform[:MAX_SAMPLES], "sampling_rate": 16000},"dataset_name": "advwave"})

    dataset = Dataset.from_list(data).cast_column("audio", Audio())
    dataset.push_to_hub(model_name + "_advwave_"+str(data_args.num_token_suffix)+"_"+str(data_args.num_epochs))

def hf_save_original():
    from datasets import load_dataset
    from huggingface_hub import login
    # You might need to log in first if you haven't already
    login(token="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") # Uncomment this and follow the prompts if needed
    index=0
    data= []
    file_path = 'harmful_behaviors.csv'  # Replace with the path to your CSV file
    original_data = pd.read_csv(file_path)
    instructions = []
    for i, row in original_data.iterrows():
        row = row.to_dict()
        target_text = row['goal']
        instructions += [target_text]
    for i in range(begin_index, end_index):
        audio_path = "../../data/advbench/original/"+ str(i)+".wav"
        waveform, sample_rate = torchaudio.load(audio_path)
        # print(sample_rate)
        # print(waveform)
        # waveform = waveform.numpy()
        # print(waveform.shape)
        if sample_rate != 16000:
            # print(target_rate)
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)[0].numpy()
        else:
            waveform = waveform.squeeze(0).numpy()
        data.append({"prompt": instructions[i], "audio": {"array": waveform, "sampling_rate": 16000},"dataset_name": "advwave"})

    dataset = Dataset.from_list(data).cast_column("audio", Audio())
    dataset.push_to_hub("advbench")

if __name__ == "__main__":
    parser = HfArgumentParser(DataTrainingArguments)
    data_args = parser.parse_args_into_dataclasses()[0]
    last_slash_index = data_args.model_name_or_path.rfind('/')
    if last_slash_index != -1:
        model_name = data_args.model_name_or_path[last_slash_index + 1:]
    begin_index = 0
    end_index = 80
    import random
    import numpy as np 
    seed=0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # For multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    train_adv(model_name, data_args,  begin_index,end_index)
    hf_save_adv(model_name, data_args, begin_index, end_index)

    # hf_save_original()
    

    