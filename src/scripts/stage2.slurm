#!/bin/bash
#SBATCH --gpus=4                 
#SBATCH --nodes=1                
#SBATCH --ntasks=4               
#SBATCH --cpus-per-gpu=1         
#SBATCH --mem-per-gpu=48GB       
#SBATCH --time=03:00:00          
#SBATCH --output=./outputs/outputfile_%j.out 

# Load required modules for your case. e.g.:
module load python3.9-anaconda/2021.11
source /sw/pkgs/arc/python3.9-anaconda/2021.11/etc/profile.d/conda.sh

# Navigate to project directory and activate environment
pushd /path/to/your/project/src/
conda activate /path/to/conda_env/myenv

export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=23000
export WORLD_SIZE=$(($SLURM_GPUS))
export NCCL_DEBUG=INFO

echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "SLURM_GPUS: $SLURM_GPUS"

# Variables
train_epochs=100
learning_rate=0.001
llama_layers=32
num_process=4
batch_size=16
d_model=32
d_ff=128

# Command to run
accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port main_stage2.py \
  --task_name long_term_forecast \
  --root_path ./data/ \
  --data_path ETTh1.csv \
  --data ETTh1 \
  --data1 ETTh1 \
  --features M \
  --seq-len 512 \
  --label-len 48 \
  --pred-len 96 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch-size $batch_size \
  --learning-rate $learning_rate \
  --llm_layers $llama_layers \
  --oti-steps $train_epochs \
  --llm_model LLAMA \
  --vocabulary vitro \
  --exp-name exp_stage2 \
  --exp-name-stage1 exp_stage1 \
  --percent 100 \
  --split train \
  --llama_model_type LlamaModel
