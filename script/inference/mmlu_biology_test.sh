#!/bin/bash
#SBATCH -J audio                 # Job name
#SBATCH -N1 --gres=gpu:H100:1
#SBATCH -t 480                                    # Duration of the job (Ex: 15 mins)
#SBATCH --mem-per-cpu=80G
#SBATCH -o mmlu_biology_bench-%j.out                         # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences


module load anaconda3/2023.03
module load gcc/12.3.0
source activate audio

# model_path=${1:-../ckpt/Qwen2-Audio-7B-Instruct_mixture_1_mixture_0}
 model_path=${1:-../ckpt/Qwen2-Audio-7B-Instruct_mixture_0_non_cot}
# model_path=${1:-../ckpt/Qwen2-Audio-7B-Instruct_mixture_0.5}
# model_path=${1:-../ckpt/Qwen2-Audio-7B-Instruct_mixture_1}
# model_path=${1:-../ckpt/Qwen2-Audio-7B-Instruct_mixture_0}
# model_path=${1:-mispeech/r1-aqa}
# model_path=${1:-Qwen/Qwen2-Audio-7B-Instruct}
# model_path=${1:-zhifeixie/Audio-Reasoner} 
path_after_slash=$(basename "$model_path") 
echo "The short model path is: $path_after_slash"

cd  ../../                            # Change to working directory

cd mmlu  

CUDA_VISIBLE_DEVICES=0 python pred.py \
	--model_dir ${model_path} \
	--data_dir anonymous4486/audio_mmlu_high_school_biology \
	--output_dir ../data/mmlu/result/biology/${path_after_slash} \
	--reward_format cot

CUDA_VISIBLE_DEVICES=0 python eval.py \
	--input_path ../data/mmlu/result/biology/${path_after_slash}
