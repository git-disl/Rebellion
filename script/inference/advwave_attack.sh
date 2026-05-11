#!/bin/bash
#SBATCH -J audio                 # Job name
#SBATCH -N1 --gres=gpu:H100:1
#SBATCH -t 600                                    # Duration of the job (Ex: 15 mins)
#SBATCH --mem-per-cpu=80G
#SBATCH -o  advwave_attack-%j.out                         # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences
#SBATCH --exclude=atl1-1-03-017-23-0

module load anaconda3/2023.03
module load gcc/12.3.0
source activate audio
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# model_path=${1:-ckpt/Qwen2-Audio-7B-Instruct_mixture_0}
model_path=${1:-Qwen/Qwen2-Audio-7B-Instruct}
#  model_path=${1:-../ckpt/Qwen2-Audio-7B-Instruct_mixture_0_non_cot}
# model_path=${1:-../ckpt/Qwen2-Audio-7B-Instruct_mixture_0.5}
# model_path=${1:-../ckpt/Qwen2-Audio-7B-Instruct_mixture_1}
# model_path=${1:-../ckpt/Qwen2-Audio-7B-Instruct_mixture_0}
# model_path=${1:-mispeech/r1-aqa}
# model_path=${1:-Qwen/Qwen2-Audio-7B-Instruct}
# model_path=${1:-zhifeixie/Audio-Reasoner}

suffix_len=${2:-200000}
path_after_slash=$(basename "$model_path")
advwave_target_model=${path_after_slash}
advwave_data=anonymous4486/${advwave_target_model}_advwave_${suffix_len}_500
echo "The short model path is: $path_after_slash"

cd  ../../                            # Change to working directory

cd tools/advwave

CUDA_VISIBLE_DEVICES=0 python attack.py \
	--model_name_or_path ../../${model_path} \
	--num_token_suffix ${suffix_len} 


cd ../../
CUDA_VISIBLE_DEVICES=0 python pred_vllm.py \
	--model_dir ${model_path} \
	--data_dir ${advwave_data} \
	--output_dir data/poison/${path_after_slash}_${advwave_target_model}_${suffix_len}_500

cd eval/safety

CUDA_VISIBLE_DEVICES=0 python eval.py \
	--input_path ../../data/poison/${path_after_slash}_${advwave_target_model}_${suffix_len}_500




