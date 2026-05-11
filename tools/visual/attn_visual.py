import logging
from dataclasses import dataclass, field
from typing import Optional

import transformers
from transformers import HfArgumentParser
import trl
from trl import GRPOConfig, SFTTrainer, SFTConfig
from sft_trainer import AudioSFTTrainer
from trainer import GRPOTrainer
from rewards import accuracy_reward, format_reward,safety_reward,cot_format_reward
# from dataset import AudioDataset
import random
import numpy as np
import torch 
from torch.utils.data import Dataset, DataLoader
from transformers import (
    Qwen2AudioForConditionalGeneration,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
import datasets
from p_sft_trainer import NoiseTrainer
from vaccine_trainer import VaccineTrainer
from trl.trainer.utils import (
    pad
)
from dataset import CustomDataCollatorForSFT
from transformers.models.qwen2_audio.modeling_qwen2_audio import Qwen2AudioAttention
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention,apply_rotary_pos_emb,eager_attention_forward,repeat_kv
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from tsne_visual import construct_data_loader
from typing import Any, Callable, Optional, Union, Dict, List, Optional, Tuple

def qwen2_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        sliding_window = None
        if (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            sliding_window = self.config.sliding_window

        # attention_interface: Callable = eager_attention_forward
        # if self.config._attn_implementation != "eager":
        #     if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
        #         print(
        #             "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
        #             'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
        #         )
        #     else:
        #         print(self.config._attn_implementation)
        #         attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        # attn_output, attn_weights = attention_interface(
        #     self,
        #     query_states,
        #     key_states,
        #     value_states,
        #     attention_mask,
        #     dropout=0.0 if not self.training else self.attention_dropout,
        #     scaling=self.scaling,
        #     sliding_window=sliding_window,  # main diff with Llama
        #     **kwargs,
        # )


        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # scale up the attention sink
        attn_weights[:,:,:,:19] = attn_weights[:,:,:,:19] +5

        # print(attn_weights.shape)
        # print(torch.norm(attn_weights[:,:19]))

        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # print(torch.norm(attn_weights[:,:,:,:]))
        attn_weights = torch.nn.functional.dropout(attn_weights, p=0, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()


        # # print("attention {}".format(attn_weights))
        # # record attention
        # # print("attention shape {}".format(attn_weights.shape))
        # # self.layerwise_attention += [attn_weights.to("cpu")]

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        
        
        return attn_output, attn_weights



@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    # model_name_or_path : Optional[str] = field(default="ckpt/Qwen2-Audio-7B-Instruct_noise2_10_0.5", metadata={"help": "model name or path"})
    model_name_or_path : Optional[str] = field(default="ckpt/Qwen2-Audio-7B-Instruct_sft_mixture_0.5", metadata={"help": "model name or path"})
    data_file: Optional[str] = field(default="anonymous4486/Qwen2-Audio-7B-Instruct_sft_mixture_0.5_advwave_150000", metadata={"help": "train data file"})
    # data_file: Optional[str] = field(default="anonymous4486/gsm8k_audio", metadata={"help": "train data file"})
    # data_file: Optional[str] = field(default="anonymous4486/Qwen2-Audio-7B-Instruct_sft_mixture_0.5_advwave_150000_0", metadata={"help": "train data file"})

    # data_file: Optional[str] = field(default="anonymous4486/Qwen2-Audio-7B-Instruct_noise2_10_0.5_advwave_150000", metadata={"help": "train data file"})
    # data_file: Optional[str] = field(default="anonymous4486/gsm8k_audio", metadata={"help": "train data file"})
    


    

def main():
    seed=0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # For multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    parser = HfArgumentParser(DataTrainingArguments)
    data_args = parser.parse_args_into_dataclasses()[0]
    # Qwen2AudioAttention.forward = atten_aug_forward_eval
    Qwen2Attention.forward = qwen2_forward
    model = Qwen2AudioForConditionalGeneration.from_pretrained(data_args.model_name_or_path,torch_dtype=torch.bfloat16,attn_implementation="eager")
    
    processor = AutoProcessor.from_pretrained(data_args.model_name_or_path)
    # processor.pad_token_id = processor.tokenizer.pad_token_id
    processor.eos_token_id = processor.tokenizer.eos_token_id
    
    dataset1 =datasets.load_dataset(data_args.data_file)["train"].select(range(2))
    
    data_loader_1 = construct_data_loader(dataset1, model, processor,for_generation=True)
    tensor_2d=0
    for index, batch in enumerate(data_loader_1):
        # if index>2:
        #     # model.layerwise_attention = []
        #     print(batch["input_ids"])
        #     with torch.no_grad():
        #         outputs = model(**batch,output_attentions=True)
        #     print(len(batch["input_ids"][0]))
        #     print(len(outputs.attentions))
        #     print(outputs.attentions[1].shape)
        #     averaged_tensor= 0
        #     for i in range(len(outputs.attentions)):
        #     # for i in range(10):
        #         averaged_tensor += torch.mean(outputs.attentions[i], dim=1)
        #     # averaged_tensor/=10
        #     averaged_tensor/=len(outputs.attentions)
            
        #     # Squeeze the tensor to remove the single-dimensional entry at index 0,
        #     # resulting in a 2D tensor of shape (430, 430)
        #     tensor_2d += torch.squeeze(averaged_tensor, dim=0)
        #     break
        generated_ids = model.generate(**batch, max_new_tokens=2048)
        # generated_ids = generated_ids[:, batch["input_ids"].size(1) :]
        batch_response = processor.batch_decode(
            generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        print(batch_response)
        break
        



    # import matplotlib.pyplot as plt
    # # Convert the 2D tensor to a NumPy array for visualization with Matplotlib
    # # image_array = tensor_2d.cpu().float().numpy()/len(data_loader_1)
    # print(torch.norm(tensor_2d[:,:19]))
    # image_array = tensor_2d[:,:19].cpu().float().numpy()
    # # Create a Matplotlib figure and axis
    # plt.figure(figsize=(8, 8))
    # # plt.ylim(top=40)
    # # Display the 2D NumPy array as an image
    # plt.imshow(image_array, cmap='viridis', vmax=0.05)

    # # Add a colorbar to show the scale of values
    # plt.colorbar(label='Averaged Tensor Value')
    # plt.savefig("attention_jailbreak_rebellion", dpi=600)



    # plt.savefig("attention_random_noise_rebellion", dpi=600)
    # plt.savefig("attention_jailbreak", dpi=600)

if __name__ == "__main__":
    main()

    