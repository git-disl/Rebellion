import logging
from dataclasses import dataclass, field
from typing import Optional

import transformers
from transformers import HfArgumentParser
from trl import GRPOConfig

from trainer import GRPOTrainer
from p_trainer import NoiseTrainer
from rewards import accuracy_reward, format_reward,safety_reward,cot_format_reward
from dataset import AudioDataset
import random
import numpy as np
import torch 

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    config_path: Optional[str] = field(default=None, metadata={"help": "config path"})
    model_name_or_path : Optional[str] = field(default="", metadata={"help": "model name or path"})
    out_dir: Optional[str] = field(default="ckpt/r1_aqa", metadata={"help": "output dir for model"})
    data_file: Optional[str] = field(default="data/AVQA/AVQA_dataset/new_train_qa.json", metadata={"help": "train data file"})
    safe_data_file: Optional[str] = field(default="anonymous4486/audio_beavertail_30k_train", metadata={"help": "train data file"})
    use_wandb: Optional[str] = field(default="flase", metadata={"help": "whether use wandb to report logs"})
    safety_mixture: Optional[float] = field(default=0, metadata={"help":"mixture of safety data"})
    reward_format: Optional[str] = field(default="cot", metadata={"help":"mixture of safety data"})
    dataset_num: Optional[int] = field(default=1000, metadata={"help":"data number"})
    max_steps: Optional[int] = field(default=500, metadata={"help":"max step"})
    # noisey training related
    perturb_train: Optional[str] = field(default="False", metadata={"help":"perturb train or not"})
    noise_lr: Optional[float] = field(default=0, metadata={"help":"noise lr"})
    rho: Optional[float] = field(default=0, metadata={"help":"noise level"})
    original_data_file: Optional[str] = field(default="", metadata={"help":"original data path"})
    # def __post_init__(self):
    #     if self.config_path is None:
    #         raise ValueError("config path should not none")


def main():
    seed=0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # For multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    parser = HfArgumentParser(DataTrainingArguments)
    data_args = parser.parse_args_into_dataclasses()[0]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # transformers.logging.set_verbosity_info()
    logging.info(data_args)

    reward_funcs_registry = {"accuracy": accuracy_reward, "format": format_reward, "safety": safety_reward, "cot_format": cot_format_reward}
    if data_args.reward_format=="cot":
        reward_funcs = [reward_funcs_registry["accuracy"], reward_funcs_registry["cot_format"], reward_funcs_registry["safety"]]
    else:
        reward_funcs = [reward_funcs_registry["accuracy"], reward_funcs_registry["format"], reward_funcs_registry["safety"]]
    train_dataset = AudioDataset(data_args.data_file, data_args.reward_format,data_args.safe_data_file, data_args.safety_mixture, data_num= data_args.dataset_num )
    
    training_args = GRPOConfig(
        seed=0,
        data_seed=0,
        output_dir=data_args.out_dir, 
        deepspeed=data_args.config_path, 
        max_prompt_length=1024, 
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1, 
        logging_steps=1, 
        bf16=True,
        report_to="wandb" if data_args.use_wandb == "true" else [],
        gradient_checkpointing=False,
        num_train_epochs=2,
        run_name="AQA-GRPO",
        save_strategy="no",
        warmup_ratio=0.1,
        save_only_model=True,
        temperature=1.0,
        beta=0,
        learning_rate=5e-7,
        # max_grad_norm=0.1,
        model_init_kwargs= {"torch_dtype": torch.bfloat16},
        num_generations=8)
        
    if data_args.perturb_train == "False":
        trainer = GRPOTrainer(
            model=data_args.model_name_or_path,
            reward_funcs=reward_funcs,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=None,
            mixture_ratio=data_args.safety_mixture)
    else:
        original_dataset = AudioDataset(data_args.original_data_file, data_args.reward_format,data_args.safe_data_file, 1, data_num= data_args.dataset_num )
        
        trainer = NoiseTrainer(
            # noise training  things
            original_model_path = data_args.original_model_name_or_path,
            original_dataset= original_dataset,
            rho = data_args.rho,
            noise_lr=data_args.noise_lr, 
            # GRPO things
            model=data_args.model_name_or_path,
            reward_funcs=reward_funcs,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=None,
            mixture_ratio=data_args.safety_mixture)
    trainer.train()
    trainer.save_model(data_args.out_dir)


if __name__ == "__main__":
    main()