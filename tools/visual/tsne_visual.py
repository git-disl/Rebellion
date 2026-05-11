import torch 
import transformers
from transformers import TrainerCallback
from torch.utils.data import Dataset
import sys
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
from transformers.models.llama.modeling_llama import LlamaAttention,LlamaMLP
from transformers.models.opt.modeling_opt import OPTAttention
from trl import GRPOConfig, SFTTrainer, SFTConfig
import numpy as np 
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
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
from sft_trainer import AudioSFTTrainer
import trl



def output_embed(model, dataloader ):
    hidden_embedding_all = []
    # Your custom logic to accumulate embeddings and labels
    from transformers.models.qwen2_audio.modeling_qwen2_audio import Qwen2AudioAttention
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention


    def get_leaf_modules_with_grad(module):
        module_list= []
        for name, module in module.named_modules():
        #     if "lora_B" in name and "v_proj" in name and len(list(module.children())) == 0:
        #         module_list+= [module]
        # or isinstance(module, LlamaMLP)
            if isinstance(module,Qwen2Attention) or isinstance(module, OPTAttention):
                module_list+= [module]
        # # print(module_list)
        return module_list
    
    def track_embedding_hook(original_embedding):
        def hook(module, input, output):
            mean_output = torch.mean(output[0].detach().to("cpu"), dim=1)
            # print(output[0].shape)
            last_output = output[0][:,-1,:].detach().to("cpu")
            # print(last_output.shape)
            # print(mean_output.shape)
            original_embedding.append(last_output.view(-1))
            torch.cuda.empty_cache()
            return output
        return hook


    def apply_track_drift_hooks_recursive(module, hook_fn, hooks):
        hook = module.register_forward_hook(hook_fn)
        hooks.append(hook)
    track_i = 0
    track_sure = 0
    # print(len(dataloader))
    for index, batch in enumerate(dataloader):
        original_embedding = []
        hooks = []
        leaf_modules_with_grad = get_leaf_modules_with_grad(model)
        for layer in leaf_modules_with_grad:
            apply_track_drift_hooks_recursive(layer, track_embedding_hook(original_embedding), hooks)
        # print(batch["input_ids"])
        # inputs = torch.cat([batch["input_ids"][0,:-2],torch.tensor([19,17,522,9217]).to("cuda:0")]).unsqueeze(0)
        inputs = batch["input_ids"]
        # print(inputs.shape)
        labels = batch["labels"][0]
         # The mask identifies tokens where labels are -1 (or -100)

        # question_mask = (labels == -100) # Use -100 as default  
        # batch["input_ids"] = batch["input_ids"][0][question_mask].unsqueeze(0)
        # batch["attention_mask"] = batch["attention_mask"][0][question_mask].unsqueeze(0)
        # batch["labels"] = batch["labels"][0][question_mask].unsqueeze(0)

        # print(inputs)
        # print(inputs.shape)
        # print(batch["input_ids"] )
        batch["input_ids"] = batch["input_ids"][0,:-3].unsqueeze(0)
        batch["attention_mask"] = batch["attention_mask"][0,:-3].unsqueeze(0)
        batch["labels"] = batch["labels"][0,:-3].unsqueeze(0)
        # # inputs = batch["input_ids"][0,:-3].unsqueeze(0)
        # # print(inputs)
        # print(batch["input_ids"])
        # suffix = torch.tensor([13708,766,29,690,26865,29,366,9217,29]).to("cuda:0")
        # batch["labels"] = torch.cat([batch["labels"][0],suffix]).unsqueeze(0)
        # batch["input_ids"] = torch.cat([batch["input_ids"][0],suffix.to("cuda:0")]).unsqueeze(0)
        # batch["attention_mask"] = torch.cat([batch["attention_mask"][0],torch.ones_like(suffix).to("cuda:0")]).unsqueeze(0)
        
        # print(inputs)
        # print(inputs.shape)
        # print(inputs.shape)
        # here need to track hidden embedding 
        with torch.no_grad():
            outputs = model(**batch)
            logits = outputs.logits # This is on GPU
            max_logit_token_index = torch.argmax(logits[:,-1,:], dim=-1)
            print("chosen index {}".format(max_logit_token_index))
            if max_logit_token_index== torch.tensor(22555):
                track_sure+=1
            if max_logit_token_index== torch.tensor(358):
                track_i+=1
            # Calculate the mean *on the GPU* first
            # This reduces the dimensionality from (B, S, V) to (B, V)
            # before any CPU transfer
            mean_logits_on_gpu = torch.mean(logits, dim=1)
        
       
        original_embedding = torch.cat(original_embedding)
        # original_embedding = original_embedding[-1]
        import copy
        # original_embedding = mean_logits_on_gpu.to("cpu")
        # original_embedding=None
        
        # print(original_embedding.shape)
        hidden_embedding_all+=[original_embedding]
        # hidden_embedding_all = torch.cat(hidden_embedding_all)
        for hook in hooks:
            hook.remove()
        del outputs
        del logits
        del mean_logits_on_gpu
        torch.cuda.empty_cache()
    print("appearance ratio of i: {}".format(track_i/50))
    print("appearance ratio of sure: {}".format(track_sure/50))
    hidden_embedding_all= torch.stack(hidden_embedding_all)
    print(hidden_embedding_all.shape)
    return hidden_embedding_all


def construct_data_loader(dataset, model, processor,for_generation=False):

    dumb_dataset = dataset.select(range(1))
    class CustomDataCollatorForSFT(trl.DataCollatorForCompletionOnlyLM):
        def __call__(self, features):
            # Separate 'source' from the features
            # Assuming 'features' is a list of dicts, one dict per sample
            for feature in features:
                if "prompt" in feature.keys():
                    feature.pop("prompt")
                if "response" in feature.keys():
                    feature.pop("response")
                if "audio" in feature.keys():
                    feature.pop("audio")
                if "original_text" in feature.keys():
                    feature.pop("original_text")
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
            # batch["dataset_name"] = dataset_names
            batch["input_features"] = input_features
            batch["feature_attention_mask"] = feature_attention_mask
            if flag:
                batch ["wave"] = waves
            # print()
            return batch
    instruction_template = "<|im_start|>user"
    response_template = "<|im_start|>assistant\n"
    collator = CustomDataCollatorForSFT(
        instruction_template=instruction_template,
        response_template=response_template,
        tokenizer=processor.tokenizer,
        mlm=False
    )

    training_args = SFTConfig(
    seed=0,
    data_seed=0,
    num_train_epochs=16,
    per_device_train_batch_size =1,
    gradient_accumulation_steps = 1,
    warmup_ratio = 0.1,
    logging_dir='./logs',
    logging_steps = 1,
    max_grad_norm=10.0,
    bf16 = True,
    optim = "adamw_torch",
    weight_decay = 0.1,
    save_strategy="no",
    lr_scheduler_type = "cosine",
    report_to=[],
    remove_unused_columns=False,
    packing=False,
    dataset_num_proc= 1
    )

    # need to reuse code from trainer to prepare audio dataset 
    trainer = AudioSFTTrainer(
            # audio sft things
            safety_mixture=None,
            original_dataset=dumb_dataset,
            model=model,
            # tokenizer=processor.tokenizer,
            # tokenizer=processor.tokenizer,
            args=training_args,
            train_dataset=dumb_dataset,
            # eval_dataset=test_dataset,
            processing_class= processor,
            data_collator=collator,
            for_generation=for_generation
        )

    dataset = trainer._prepare_dataset(dataset, processor, training_args, None, None, "dataset1")
    data_loader = trainer.get_dataloader(dataset)
    return data_loader


def visual_test_tsne(embedded_features,labels_num, each_label_size, label_name, output_name):
    merged_labels  = np.concatenate([ [0 for i in range(50) ] ,  [1 for i in range(50) ]], axis=0)
    # print(embedded_features)
    # print(embedded_features.shape)
    
    import matplotlib.pyplot as plt
    # Add jitter to the points
    jitter_strength = 0  # Adjust the strength of the jitter
    jitter = np.random.normal(scale=jitter_strength, size=embedded_features.shape)
    jittered_features = embedded_features + jitter
    # Plot the t-SNE visualization
    plt.figure(figsize=(8, 6))
    # print(merged_labels[30:60])
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'black', 'magenta', 'yellow', 'lime', 'cyan', 'teal']
    for i in range(labels_num):
        plt.scatter(jittered_features[(i)*each_label_size:(i+1)*each_label_size, 0], jittered_features[(i)*each_label_size:(i+1)*each_label_size, 1], c=colors[i], label=label_name[i], alpha=0.5)
    plt.legend(fontsize =20, framealpha=0.5)
    # if "sft" in args.lora_folder:
    #     plt.title('t-SNE Visualization of Hidden Embedding (SFT)', fontsize =15)
    # else:
    #     plt.title('t-SNE Visualization of Hidden Embedding (Vaccine)', fontsize =15)
    # Reduce margins
    plt.tight_layout()
    # Hide axis numbers
    plt.xticks([])  # Remove x-axis numbers
    plt.yticks([])  # Remove y-axis numbers
    # plt.show()
    # filename = args.lora_folder.split('/')[-1]
    # # plt.colorbar()
    plt.savefig(output_name, dpi=600)


@dataclass
class DataArguments:
    data_path1: str = field(default=None, metadata={"help": "Path to the training data."})
    data_path2: str = field(default=None, metadata={"help": "Path to the training data."})
    data_path3: str = field(default=None, metadata={"help": "Path to the training data."})
    data_path4:str = field(default=None, metadata={"help": "Path to the training data."})
    data_path5:str = field(default=None, metadata={"help": "Path to the training data."})
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")

def main():
    parser = transformers.HfArgumentParser(DataArguments)
    data_args = parser.parse_args_into_dataclasses()[0]



    model = Qwen2AudioForConditionalGeneration.from_pretrained("ckpt/Qwen2-Audio-7B-Instruct_sft_mixture_1",torch_dtype=torch.bfloat16)
    # print(model)
    processor = AutoProcessor.from_pretrained(data_args.model_name_or_path)
    # processor.pad_token_id = processor.tokenizer.pad_token_id
    processor.eos_token_id = processor.tokenizer.eos_token_id

    dataset1 =datasets.load_dataset(data_args.data_path1)["train"].select(range(50))
    dataset2 =datasets.load_dataset(data_args.data_path2)["train"].select(range(50))
    dataset3 =datasets.load_dataset(data_args.data_path3)["train"].select(range(50))
    dataset4 =datasets.load_dataset(data_args.data_path4)["train"].select(range(50))
    dataset5 =datasets.load_dataset(data_args.data_path5)["train"].select(range(50))


    data_loader_1 = construct_data_loader(dataset1, model, processor)
    data_loader_2 = construct_data_loader(dataset2, model,processor)
    data_loader_3 = construct_data_loader(dataset3, model,processor)
    data_loader_4 = construct_data_loader(dataset4, model,processor)
    data_loader_5 = construct_data_loader(dataset5, model,processor)

    # model = Qwen2AudioForConditionalGeneration.from_pretrained("ckpt/Qwen2-Audio-7B-Instruct_sft_mixture_0",torch_dtype=torch.bfloat16)
    # model = model.to("cuda:0")
    # embedded_features_0 = output_embed(model, data_loader_1)

    # model = Qwen2AudioForConditionalGeneration.from_pretrained("ckpt/Qwen2-Audio-7B-Instruct_sft_mixture_0",torch_dtype=torch.bfloat16)
    # model = model.to("cuda:0")
    # embedded_features_1 = output_embed(model, data_loader_2)
    # # embedded_features_2 = output_embed(model, data_loader_1)


    # model = Qwen2AudioForConditionalGeneration.from_pretrained("ckpt/Qwen2-Audio-7B-Instruct_sft_mixture_0.1",torch_dtype=torch.bfloat16)
    # model = model.to("cuda:0")
    # embedded_features_2 = output_embed(model, data_loader_2)


    # model = Qwen2AudioForConditionalGeneration.from_pretrained("ckpt/Qwen2-Audio-7B-Instruct_sft_mixture_0.5",torch_dtype=torch.bfloat16)
    # model = model.to("cuda:0")
    # embedded_features_3 = output_embed(model, data_loader_2)

    # model = Qwen2AudioForConditionalGeneration.from_pretrained("ckpt/Qwen2-Audio-7B-Instruct_sft_mixture_0.9",torch_dtype=torch.bfloat16)
    # model = model.to("cuda:0")
    # embedded_features_4 = output_embed(model, data_loader_2)


    # model = Qwen2AudioForConditionalGeneration.from_pretrained("ckpt/Qwen2-Audio-7B-Instruct_noise2_0_0.1",torch_dtype=torch.bfloat16)
    # model = model.to("cuda:0")
    # embedded_features_5 = output_embed(model, data_loader_2)

    # model = Qwen2AudioForConditionalGeneration.from_pretrained("ckpt/Qwen2-Audio-7B-Instruct_noise2_1_0.1",torch_dtype=torch.bfloat16)
    # model = model.to("cuda:0")
    # embedded_features_6 = output_embed(model, data_loader_2)

    # model = Qwen2AudioForConditionalGeneration.from_pretrained("ckpt/Qwen2-Audio-7B-Instruct_noise2_10_0.5",torch_dtype=torch.bfloat16)


    model = Qwen2AudioForConditionalGeneration.from_pretrained("ckpt/Qwen2-Audio-7B-Instruct_sft_mixture_0",torch_dtype=torch.bfloat16)
    model = model.to("cuda:0")
    embedded_features_1 = output_embed(model, data_loader_1)

    embedded_features_2 = output_embed(model, data_loader_2)

    print("drift alpha =0 {}".format( sum([ torch.norm(embedded_features_1[i]-embedded_features_2[i])  for i in range(len(embedded_features_1))] )))

    model = Qwen2AudioForConditionalGeneration.from_pretrained("ckpt/Qwen2-Audio-7B-Instruct_sft_mixture_0.1",torch_dtype=torch.bfloat16)
    model = model.to("cuda:0")
    embedded_features_3 = output_embed(model, data_loader_1)

    embedded_features_4 = output_embed(model, data_loader_3)


    print("drift alpha =0.1  {}".format(sum([ torch.norm(embedded_features_3[i]-embedded_features_4[i])  for i in range(len(embedded_features_3))] )))

    model = Qwen2AudioForConditionalGeneration.from_pretrained("ckpt/Qwen2-Audio-7B-Instruct_sft_mixture_0.5",torch_dtype=torch.bfloat16)
    model = model.to("cuda:0")
    embedded_features_5 = output_embed(model, data_loader_1)

    embedded_features_6 = output_embed(model, data_loader_4)


    print("drift alpha =0.5  {}".format(sum([ torch.norm(embedded_features_5[i]-embedded_features_6[i])  for i in range(len(embedded_features_5))] )))

    model = Qwen2AudioForConditionalGeneration.from_pretrained("ckpt/Qwen2-Audio-7B-Instruct_noise2_10_0.5",torch_dtype=torch.bfloat16)
    model = model.to("cuda:0")

    embedded_features_7 = output_embed(model, data_loader_1)

    embedded_features_8 = output_embed(model, data_loader_5)

    print("drift rebellion {}".format(sum([ torch.norm(embedded_features_7[i]-embedded_features_8[i])  for i in range(len(embedded_features_7))] )))

    # dsd
  
    merged_features = torch.cat([ embedded_features_1, embedded_features_2,embedded_features_3,embedded_features_4,embedded_features_5,embedded_features_6,embedded_features_7,embedded_features_8], dim=0).float().numpy()
    tsne = TSNE(n_components=2, random_state=40)
    embedded_features = tsne.fit_transform(merged_features)
    torch.set_printoptions(profile="full")
    print(embedded_features)
    label_names = [ "harmful prompt (SFT (\u03B1=0) )", "jailbreak prompt (SFT (\u03B1=0) )", "harmful prompt (SFT (\u03B1=0.1) )","jailbreak prompt (SFT (\u03B1=0.1))","harmful prompt (SFT (\u03B1=0.5) )","jailbreak prompt (SFT (\u03B1=0.5))", "harmful prompt (Rebellion)", "jailbreak prompt (Rebellion)", "rebellion (\u03B1=0.5) evaluated on harmful prompts","rebellion (\u03B1=0.9) evaluated on harmful prompts","rebellion (\u03B1=0.9) evaluated on harmful prompts", "Mixed (\u03B1=1) evaluated on harmful prompts", "Mixed (\u03B1=0.9) evaluated on harmful prompts", "Mixed (\u03B1=1) evaluated on harmful prompts","Mixed (\u03B1=1) evaluated on benign prompts"]


    labels_num=2
    each_label_size=50
    visual_test_tsne(embedded_features[:100], labels_num, each_label_size, label_names[:2],"sft_harmful_alpha0.png")
    visual_test_tsne(embedded_features[100:200], labels_num, each_label_size, label_names[2:4],"sft_harmful_alpha0.1.png")
    visual_test_tsne(embedded_features[200:300], labels_num, each_label_size, label_names[4:6],"sft_harmful_alpha0.5.png")
    visual_test_tsne(embedded_features[300:400], labels_num, each_label_size, label_names[6:8],"rebellion_harmful.png")

if __name__ == "__main__":
    main()
