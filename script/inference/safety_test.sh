#!/bin/bash
#SBATCH -J audio                 # Job name
#SBATCH -N1 --gres=gpu:H200:1
#SBATCH -t 480                                    # Duration of the job (Ex: 15 mins)
#SBATCH --mem-per-cpu=80G
#SBATCH -o safety_test-%j.out                         # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences


module load anaconda3/2023.03
module load gcc/12.3.0
module load ffmpeg
source activate hts

# model_path=${1:-../../ckpt/Qwen2-Audio-7B-Instruct_mixture_1_mixture_0}
# model_path=${1:-../../ckpt/Qwen2-Audio-7B-Instruct_mixture_0_non_cot}



# model_path=${1:-../../ckpt/Qwen2-Audio-7B-Instruct_mixture_0.5}


# model_path=${1:-../../ckpt/Qwen2-Audio-7B-Instruct_mixture_0}
# model_path=${1:-../../ckpt/Qwen2-Audio-7B-Instruct_mixture_1}
# model_path=${1:-mispeech/r1-aqa}
# model_path=${1:-Qwen/Qwen2-Audio-7B-Instruct}
# model_path=${1:-../../ckpt/Qwen2-Audio-7B-Instruct_mixture_0_avqa_non_cot}
# model_path=${1:-../../ckpt/Qwen2-Audio-7B-Instruct_mixture_0_avqa}
# model_path=${1:-zhifeixie/Audio-Reasoner} 

# Evaluate safety mixture
# model_path=${1:-../../ckpt/Qwen2-Audio-7B-Instruct_mixture_0.05}
# model_path=${1:-../../ckpt/Qwen2-Audio-7B-Instruct_mixture_0.1}
# model_path=${1:-../../ckpt/Qwen2-Audio-7B-Instruct_mixture_0.2}
# model_path=${1:-../../ckpt/Qwen2-Audio-7B-Instruct_mixture_0.5}

model_path=nvidia/audio-flamingo-3-hf
# model_path=${1:-ckpt/audio-flamingo-3-hf_sft_mixture_0.5}
# model_path=ckpt/audio-flamingo-3-hf_noise2_10_0.5 

path_after_slash=$(basename "$model_path") 
echo "The short model path is: $path_after_slash"
# echo "Safety Mixture: $safety_mixture"

cd  ../../                            # Change to working directory


CUDA_VISIBLE_DEVICES=0 python pred_vllm.py \
	--model_dir ${model_path} \
	--data_dir anonymous4486/advbench \
	--output_dir data/poison/${path_after_slash} 

cd eval/safety

CUDA_VISIBLE_DEVICES=0 python eval.py \
	--input_path ../../data/poison/${path_after_slash}
