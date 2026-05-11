#!/bin/bash
#SBATCH -J audio                 # Job name
#SBATCH -N1 --gres=gpu:H100:1
#SBATCH -t 600                                    # Duration of the job (Ex: 15 mins)
#SBATCH --mem-per-cpu=200G
#SBATCH -o safety_test_jaml-%j.out                         # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences
#SBATCH --exclude=atl1-1-03-017-23-0
module load anaconda3/2023.03
module load gcc/12.3.0
source activate audio
export CUBLAS_WORKSPACE_CONFIG=:4096:8
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
model_path=${1:-Qwen/Qwen2-Audio-7B-Instruct}
# model_path=${1:-../../ckpt/Qwen2-Audio-7B-Instruct_mixture_0.1}
# model_path=${1:-../../ckpt/Qwen2-Audio-7B-Instruct_mixture_0.2}
# model_path=${1:-../../ckpt/Qwen2-Audio-7B-Instruct_mixture_0.5}
advwave_data=AnonymousUser000/JALMBench
# advwave_data=anonymous4486/advwave_blackbox
path_after_slash=$(basename "$model_path") 
echo "The short model path is: $path_after_slash"
# echo "Safety Mixture: $safety_mixture"

cd  ../../                            # Change to working directory

CUDA_VISIBLE_DEVICES=0 python pred_vllm.py \
	--model_dir ${model_path} \
	--data_dir ${advwave_data} \
	--output_dir data/poison/${path_after_slash}_JALMBench \

cd eval/safety 
CUDA_VISIBLE_DEVICES=0 python eval.py \
	--input_path ../../data/poison/${path_after_slash}_JALMBench
