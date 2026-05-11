#!/bin/bash
#SBATCH -J audio                 # Job name
#SBATCH -N1 --gres=gpu:H200:1
#SBATCH -t 480                                    # Duration of the job (Ex: 15 mins)
#SBATCH --mem-per-cpu=30G
#SBATCH -o reasoning_bench-%j.out                         # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences


module load anaconda3/2023.03
module load gcc/12.3.0
source activate audio

# model_path=${1:-../ckpt/r1_aqa}
# model_path=${1:-../ckpt/Qwen2-Audio-7B-Instruct_mixture_0_mixture_1/checkpoint-700}

model_path=${1:-../ckpt/Qwen2-Audio-7B-Instruct_mixture_0_avqa_non_cot}
model_path=${1:-../ckpt/Qwen2-Audio-7B-Instruct_mixture_0_avqa}
# model_path=${1:-../ckpt/Qwen2-Audio-7B-Instruct_mixture_1_mixture_0}
# model_path=${1:-../ckpt/Qwen2-Audio-7B-Instruct_mixture_0}
# model_path=${1:-../ckpt/Qwen2-Audio-7B-Instruct_mixture_0.5}
# model_path=${1:-../ckpt/Qwen2-Audio-7B-Instruct_mixture_1}
# model_path=${1:-mispeech/r1-aqa}
# model_path=${1:-Qwen/Qwen2-Audio-7B-Instruct} \
# model_path=${1:-zhifeixie/Audio-Reasoner} 
path_after_slash=$(basename "$model_path") 
echo "The short model path is: $path_after_slash"
cd  ../../                            # Change to working directory

cd mmau  

CUDA_VISIBLE_DEVICES=0 python pred.py \
	--model_dir ${model_path} \
	--data_dir ../data/mmau/mmau-test-mini.json \
	--audio_dir ../data/mmau \
	--output_dir ../data/mmau/result/${path_after_slash}_inf_non_cot.json \
	--reward_format non_cot

	
python evaluation.py --input ../data/mmau/result/${path_after_slash}_inf_non_cot.json 

# python show_acc.py -i ../data/mmau/result/${path_after_slash}.json