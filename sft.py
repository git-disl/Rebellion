import logging
from dataclasses import dataclass, field
from typing import Optional
import transformers
from transformers import HfArgumentParser
import trl
from trl import SFTTrainer, SFTConfig
# from dataset import AudioDataset
import random
import numpy as np
import torch 
from torch.utils.data import Dataset, DataLoader
from transformers import (
    Qwen2AudioForConditionalGeneration,
    AudioFlamingo3ForConditionalGeneration,
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
from trainer.sft_trainer import AudioSFTTrainer
from trainer.vaccine_trainer import VaccineTrainer
from trl.trainer.utils import (
    pad
)
import os 


from dataset import CustomDataCollatorForSFT
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
    method: Optional[str] = field(default="sft", metadata={"help":"training method"})
    suffix_length: Optional[int] = field(default=40, metadata={"help":"suffix length"})
    noise_lr: Optional[float] = field(default=0, metadata={"help":"noise lr"})
    rho: Optional[float] = field(default=0.1, metadata={"help":"noise level"})
    lamb: Optional[float] = field(default=0.1, metadata={"help":"balance level"})
    learning_rate: Optional[float] = field(default=1e-5, metadata={"help":"learning rate"})
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
    torch.use_deterministic_algorithms(True)
    # import deepspeed
    # deepspeed.init_distributed()
    parser = HfArgumentParser(DataTrainingArguments)
    data_args = parser.parse_args_into_dataclasses()[0]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # transformers.logging.set_verbosity_info()
    logging.info(data_args)

    # train_dataset = AudioDataset(data_args.data_file, data_args.reward_format,data_args.safe_data_file, data_args.safety_mixture, data_num= data_args.dataset_num )
    
    training_args = SFTConfig(
    seed=0,
    data_seed=0,
    output_dir=data_args.out_dir,
    # num_train_epochs=3,
    num_train_epochs=10,
    #max_steps=1000,
    # per_device_eval_batch_size = 0,
    per_device_train_batch_size =1,
    gradient_accumulation_steps = 1,
    learning_rate = data_args.learning_rate,
    logging_steps = 1,
    #evaluation_strategy="steps",
    #eval_steps=75,
    #save_steps=100,
    max_grad_norm=10.0,
    bf16 = True,
    optim = "adamw_torch",
    weight_decay = 0.1,
    #seed = 3407,
    save_strategy="no",
    #save_strategy="epoch",
    # use_liger_loss=False,
    lr_scheduler_type = "cosine",
    #load_best_model_at_end=True,
    report_to=[],
    remove_unused_columns=False,
    packing=False,
    dataset_num_proc= 8
    )
    # if data_args.noise_train =="noise_train":
    #     model = Qwen2AudioForConditionalGeneration.from_pretrained(data_args.model_name_or_path,torch_dtype=torch.bfloat16,attn_implementation="eager")
    # else:
    
    if "Qwen" in data_args.model_name_or_path:
        model = Qwen2AudioForConditionalGeneration.from_pretrained(data_args.model_name_or_path,torch_dtype=torch.bfloat16)
    else:
        model = AudioFlamingo3ForConditionalGeneration.from_pretrained(data_args.model_name_or_path, torch_dtype=torch.bfloat16)
    # print(model)
    processor = AutoProcessor.from_pretrained(data_args.model_name_or_path)
    # processor.pad_token_id = processor.tokenizer.pad_token_id
    processor.eos_token_id = processor.tokenizer.eos_token_id
    
    # # Use a token that is never used
    # processor.tokenizer.pad_token = "<|fim_pad|>"

    # Only compute loss over assistant responses
    # Verified that it precisely starts where the thinking tokens start and ends with the first pad token
    # via labels being set to -100

    instruction_template = "<|im_start|>user"
    response_template = "<|im_start|>assistant\n"
    # Use a token that is never used
    # tokenizer.pad_token = "<|fim_pad|>"

    # Only compute loss over assistant responses
    # Verified that it precisely starts where the thinking tokens start and ends with the first pad token
    # via labels being set to -100

    collator = CustomDataCollatorForSFT(
        instruction_template=instruction_template,
        response_template=response_template,
        tokenizer=processor.tokenizer,
        mlm=False
    )
    
    # train_dataset =datasets.load_dataset(data_args.data_file)["train"].select(range(2))
    original_dataset =datasets.load_dataset(data_args.data_file)["train"].select(range(500))
    alpaca_dataset = datasets.load_dataset("anonymous4486/audio_alpaca_train")["train"].select(range(500))
    safe_dataset =datasets.load_dataset(data_args.safe_data_file)["train"].select(range(500))
    # safe_dataset =datasets.load_dataset(data_args.safe_data_file)["train"].select(range(2))
    # Merge the two datasets
    merged_dataset = datasets.concatenate_datasets([original_dataset, alpaca_dataset])

    # harmful_dataset  =datasets.load_dataset(data_args.harmful_data_file)["train"].select(range(1))
    # print(train_dataset)
    
    # print(train_dataset)
    training_args.loss_type = "nll"
    if data_args.method == "sft": 
        trainer = AudioSFTTrainer(
            # audio sft things
            safety_mixture=data_args.safety_mixture,
            original_dataset=merged_dataset,
            model=model,
            # tokenizer=processor.tokenizer,
            # tokenizer=processor.tokenizer,
            args=training_args,
            train_dataset=safe_dataset,
            # eval_dataset=test_dataset,
            processing_class= processor,
            data_collator=collator
        )
    # elif data_args.method =="balance":
    #     training_args.rho=data_args.rho
    #     training_args.lamb = data_args.lamb
    #     training_args.rebellion_enable=data_args.rebellion_enable
    #     trainer = BalanceTrainer(
    #         original_dataset = merged_dataset,
    #         processing_class= processor,
    #         # rho = data_args.rho,
    #         # noise_lr=data_args.noise_lr, 
    #         # processor= processor,
    #         # suffix_length= data_args.suffix_length, 
    #         # # SFT things
    #         safety_mixture=data_args.safety_mixture,
    #         model=model,
    #         # tokenizer=processor.tokenizer,
    #         # tokenizer=processor.tokenizer,
    #         args=training_args,
    #         train_dataset=safe_dataset,
    #         # eval_dataset=test_dataset,
    #         data_collator=collator
    #     )
    else:
        training_args.rho=data_args.rho
        trainer = VaccineTrainer(
            original_dataset = merged_dataset,
            processing_class= processor,
            # # SFT things
            safety_mixture=data_args.safety_mixture,
            model=model,
            args=training_args,
            train_dataset=safe_dataset,
            data_collator=collator
        )
    
        # noise trainer modify the processor and therefore need to be save 
        

    trainer.train()
    trainer.save_model(data_args.out_dir)
    # if data_args.noise_train =="noise_train":
    #     torch.save(trainer.adv_audio_suffix, data_args.out_dir+ "/noise.pt")

if __name__ == "__main__":
    main()