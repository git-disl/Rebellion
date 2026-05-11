#!/bin/bash
#SBATCH -J audio                 # Job name
#SBATCH -N1 --gres=gpu:H200:8
#SBATCH -t 120                                    # Duration of the job (Ex: 15 mins)
#SBATCH --mem-per-cpu=80G
#SBATCH --cpus-per-task=16
#SBATCH -o grpo_avqa_non_cot-%j.out                         # Combined output and error messages file
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

MODEL_NP=Qwen/Qwen2-Audio-7B-Instruct
# MODEL_NP=ckpt/Qwen2-Audio-7B-Instruct_mixture_1
path_after_slash=$(basename "$MODEL_NP")
safety_mixture=0
GPU_NUM=8
NODE_NUM=1
NODE_RANK=0
MASTER_ADDR="127.0.0.1"
MASTER_PORT=32777

echo "The short model path is: $path_after_slash"
echo "Safety Mixture: $safety_mixture"

cd ../../
torchrun --nproc_per_node=${GPU_NUM} \
    --nnodes=${NODE_NUM} \
    --node-rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    train.py \
    --config_path config/ds_zero3.json \
    --model_name_or_path ${MODEL_NP} \
    --reward_format non_cot \
    --safety_mixture ${safety_mixture} \
    --out_dir ckpt/${path_after_slash}_mixture_${safety_mixture}_avqa_non_cot\
    --dataset_num 1000 \
    --use_wandb false || exit 1