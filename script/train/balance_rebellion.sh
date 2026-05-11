#!/bin/bash
#SBATCH -J audio                 # Job name
#SBATCH --gres=gpu:H200:4
#SBATCH -t 180                                    # Duration of the job (Ex: 15 mins)
#SBATCH  -q coc-grade
#SBATCH --mem-per-cpu=40G
#SBATCH --cpus-per-task=16
#SBATCH --exclude=atl1-1-03-017-23-0
#SBATCH -o balance_rebellion-%j.out                         # Combined output and error messages file
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
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# MODEL_NP=${1:-ckpt/Qwen2-Audio-7B-Instruct_mixture_0}
MODEL_NP=Qwen/Qwen2-Audio-7B-Instruct
path_after_slash=$(basename "$MODEL_NP")
lamb=${1:-0.1}
rho=${2:-10}
GPU_NUM=3
NODE_NUM=1
NODE_RANK=0
MASTER_ADDR="127.0.0.1"
MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo 'Chosen MASTER_PORT: '$MASTER_PORT
data_file=anonymous4486/gsm8k_cot_audio_train
safe_data_file=anonymous4486/star1_audio
mix_ratio=${3:-1}

echo "The short model path is: $path_after_slash"
echo "lamb: $lamb"
echo "rho: $rho"
echo "mix_ratio: $mix_ratio"

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
    --out_dir ckpt/${path_after_slash}_balance_rebellion_${lamb}_${rho}_${mix_ratio} \
    --max_steps 500 \
    --method balance \
    --learning_rate 1e-5 \
    --rebellion_enable True \
    --lamb ${lamb} \
    --rho ${rho} \




