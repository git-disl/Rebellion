#!/bin/bash
#SBATCH -J audio                 # Job name
#SBATCH --gres=gpu:H200:2
#SBATCH -t 180                                    # Duration of the job (Ex: 15 mins)
#SBATCH  -q coc-grade
#SBATCH --mem=490G
#SBATCH --cpus-per-task=16
#SBATCH -o optimize_noise-%j.out                         # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences
#SBATCH --exclude=atl1-1-03-017-23-0 # Exclude a specific problematic node
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

# *** THIS IS CRITICAL: UNSET LD_LIBRARY_PATH ***
# unset LD_LIBRARY_PATH
export PATH=~/.conda/envs/hts/bin:$PATH
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# MODEL_NP=${1:-ckpt/Qwen2-Audio-7B-Instruct_mixture_0}
MODEL_NP=Qwen/Qwen2-Audio-7B-Instruct
# MODEL_NP=nvidia/audio-flamingo-3-hf
path_after_slash=$(basename "$MODEL_NP")
suffix_length=0
rho=${1:-10}
GPU_NUM=2
NODE_NUM=1
NODE_RANK=0
MASTER_ADDR="127.0.0.1"
MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo 'Chosen MASTER_PORT: '$MASTER_PORT
data_file=anonymous4486/gsm8k_cot_audio_train
safe_data_file=anonymous4486/star1_audio
mix_ratio=${2:-0.5}
noise_lr=1

echo "The short model path is: $path_after_slash"
echo "suffix_length: $suffix_length"
echo "rho: $rho"
echo "mix_ratio: $mix_ratio"
echo "noise_lr: $noise_lr"

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
    --out_dir ckpt/${path_after_slash}_noise2_${rho}_${mix_ratio} \
    --max_steps 500 \
    --method rebellion \
    --noise_lr ${noise_lr} \
    --learning_rate 1e-5 \
    --rho ${rho} 




