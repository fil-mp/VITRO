#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# Create output directory if it doesn't exist
mkdir -p outputs

# Run the command and redirect output to a new log file
log_file="outputs/output_$(date +%s).log"

# Command to run
# you can try longer than 200 oti-steps depending on your system
python main_stage1.py \
    --split train \
    --llama_model_type LlamaModel \
    --exp-name exp_stage1 \
    --batch-size 2 \
    --oti-steps 200 \
    --seq-len 512 \
    --learning-rate 0.0002 \
    --percent 100 \
    --pred-len 96 > "$log_file" 2>&1 & 
