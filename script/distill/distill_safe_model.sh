#!/bin/bash
#SBATCH -J audio                 # Job name
#SBATCH -N1 --gres=gpu:H200:1
#SBATCH -t 480                                    # Duration of the job (Ex: 15 mins)
#SBATCH --mem-per-cpu=80G
#SBATCH -o gsm8k_bench-%j.out                         # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences


module load anaconda3/2023.03
module load gcc/12.3.0
source activate audio

model_path=${1:-ckpt/Qwen2-Audio-7B-Instruct_mixture_0}

path_after_slash=$(basename "$model_path") 
echo "The short model path is: $path_after_slash"

cd  ../../                            # Change to working directory

CUDA_VISIBLE_DEVICES=0 python distill.py \
	--model_name_or_path ${model_path} \
	--data_file anonymous4486/audio_beavertail_30k_train
	

