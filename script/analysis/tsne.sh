#!/bin/bash
#SBATCH -J audio                 # Job name
#SBATCH -N1 --gres=gpu:H200:1
#SBATCH -t 480                                    # Duration of the job (Ex: 15 mins)
#SBATCH --mem-per-cpu=80G
#SBATCH -o tsne_analysis-%j.out                         # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences


module load anaconda3/2023.03
module load gcc/12.3.0
source activate audio

model_path=${1:-ckpt/Qwen2-Audio-7B-Instruct_sft_mixture_0}
path_after_slash=$(basename "$model_path") 
echo "The short model path is: $path_after_slash"

cd  ../../                            # Change to working directory

CUDA_VISIBLE_DEVICES=0 python tsne_visual.py \
	--model_name_or_path ${model_path} \
	--data_path1 anonymous4486/advbench\
	--data_path2 anonymous4486/Qwen2-Audio-7B-Instruct_sft_mixture_0_advwave_50000 \
	--data_path3 anonymous4486/Qwen2-Audio-7B-Instruct_sft_mixture_0.1_advwave_50000 \
	--data_path4 anonymous4486/Qwen2-Audio-7B-Instruct_sft_mixture_0.5_advwave_50000 \
	--data_path5 anonymous4486/Qwen2-Audio-7B-Instruct_noise2_10_0.5_advwave_200000
	
	# --data_path3 anonymous4486/Qwen2-Audio-7B-Instruct_noise2_10_0.5_advwave \

# data_file=anonymous4486/gsm8k_cot_audio_train
# safe_data_file=anonymous4486/star1_audio