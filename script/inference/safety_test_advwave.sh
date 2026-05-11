#!/bin/bash
#SBATCH -J audio                 # Job name
#SBATCH -N1 --gres=gpu:H200:1
#SBATCH -t 480                                    # Duration of the job (Ex: 15 mins)
#SBATCH --mem-per-cpu=80G
#SBATCH -o safety_test_advwave-%j.out                         # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences


module load anaconda3/2023.03
module load gcc/12.3.0
source activate audio

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
model_path=${1:-../../ckpt/Qwen2-Audio-7B-Instruct_mixture_0.05}
# model_path=${1:-../../ckpt/Qwen2-Audio-7B-Instruct_mixture_0.1}
# model_path=${1:-../../ckpt/Qwen2-Audio-7B-Instruct_mixture_0.2}
# model_path=${1:-../../ckpt/Qwen2-Audio-7B-Instruct_mixture_0.5}
advwave_target_model=Qwen2-Audio-7B-Instruct
advwave_data=anonymous4486/${advwave_target_model}_advwave 

path_after_slash=$(basename "$model_path") 
echo "The short model path is: $path_after_slash"
# echo "Safety Mixture: $safety_mixture"

cd  ../../                            # Change to working directory


CUDA_VISIBLE_DEVICES=0 python advwave_pred.py \
	--model_dir ${model_path} \
	--advwave_data ${advwave_data} \
	--output_dir ../../data/poison/${path_after_slash}_${advwave_target_model}_advwave \
	--reward_format cot


cd poison
  
CUDA_VISIBLE_DEVICES=0 python eval_sentiment.py \
	--input_path ../../data/poison/${path_after_slash}_${advwave_target_model}_advwave
