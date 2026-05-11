#!/bin/bash
#SBATCH -J audio                 # Job name
#SBATCH -N1 --gres=gpu:H200:8
#SBATCH -t 180                                    # Duration of the job (Ex: 15 mins)
#SBATCH  -q coc-grade
#SBATCH --mem-per-cpu=80G
#SBATCH --cpus-per-task=16
#SBATCH -o grpo_safety_mix-%j.out                         # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences
# unset CC
# unset CXX
# unset CFLAGS
# unset CXXFLAGS
# unset LDFLAGS
# unset LIBRARY_PATH
# unset CPATH
# unset CPLUS_INCLUDE_PATH

module load anaconda3/2023.03
# module load cuda/12.1.1
module load gcc/12.3.0

source activate audio

# *** THIS IS CRITICAL: UNSET LD_LIBRARY_PATH ***
unset LD_LIBRARY_PATH
export PATH=~/.conda/envs/audio/bin:$PATH

MODEL_NP=ckpt/Qwen2-Audio-7B-Instruct_mixture_0
# MODEL_NP=Qwen/Qwen2-Audio-7B-Instruct
path_after_slash=$(basename "$MODEL_NP")
GPU_NUM=8
NODE_NUM=1
NODE_RANK=0
MASTER_ADDR="127.0.0.1"
MASTER_PORT=32777
data_file=anonymous4486/gsm8k_audio_train
mix_ratio=${1:-0.1} 
echo "The short model path is: $path_after_slash"
echo "Safety Mixture: $mix_ratio"

cd ../../
torchrun --nproc_per_node=${GPU_NUM} \
    --nnodes=${NODE_NUM} \
    --node-rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    train.py \
    --config_path config/ds_zero3.json \
    --model_name_or_path ${MODEL_NP} \
    --reward_format cot \
    --safety_mixture ${mix_ratio} \
    --dataset_num 1000 \
    --data_file ${data_file} \
    --out_dir ckpt/${path_after_slash}_mixture_${mix_ratio} \
    --max_steps 500 \
    --use_wandb false || exit 1