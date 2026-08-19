#!/bin/bash
#SBATCH -J audio                 # Job name
#SBATCH --gres=gpu:H200:2
#SBATCH -t 300                                    # Duration of the job (Ex: 15 mins)
#SBATCH  -q coc-grade
#SBATCH --mem-per-cpu=80G
#SBATCH --cpus-per-task=16
#SBATCH -o sft_safety_mix-%j.out                         # Combined output and error messages file
#SBATCH --exclude=atl1-1-03-017-16-0 # Exclude a specific problematic node
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
module load ffmpeg
source activate hts
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# *** THIS IS CRITICAL: UNSET LD_LIBRARY_PATH ***
# unset LD_LIBRARY_PATH
export PATH=~/.conda/envs/hts/bin:$PATH
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# MODEL_NP=ckpt/Qwen2-Audio-7B-Instruct_sft_mixture_0
MODEL_NP=nvidia/audio-flamingo-3-hf
path_after_slash=$(basename "$MODEL_NP")
GPU_NUM=2
NODE_NUM=1
NODE_RANK=0
MASTER_ADDR="127.0.0.1"
MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
data_file=anonymous4486/gsm8k_cot_audio_train
safe_data_file=anonymous4486/star1_audio
mix_ratio=${1:-0.5} 
echo "The short model path is: $path_after_slash"
echo "Safety Mixture: $mix_ratio"

cd ../../
torchrun --nproc_per_node=${GPU_NUM} \
    --nnodes=${NODE_NUM} \
    --node-rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    sft.py \
    --model_name_or_path ${MODEL_NP} \
    --reward_format cot \
    --safety_mixture ${mix_ratio} \
    --data_file ${data_file} \
    --safe_data_file  ${safe_data_file} \
    --out_dir ckpt/${path_after_slash}_sft_mixture_${mix_ratio} \
    --learning_rate 1e-5 \
    --max_steps 500 

# --config_path config/ds_zero3.json \