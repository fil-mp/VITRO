#!/bin/bash

train_epochs=100
learning_rate=0.001
llama_layers=32

master_port=23000
num_process=4 
batch_size=16
d_model=32
d_ff=128

# Set the CUDA_VISIBLE_DEVICES environment variable
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Create output directory if it doesn't exist
mkdir -p outputs

# Run the command and redirect output to a new log file
log_file="outputs/output_$(date +%s).log"

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
  --llama_model_type LlamaModel > "$log_file" 2>&1 & 
  #--lradj 'TST' depending on dataset