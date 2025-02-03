#!/bin/bash
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=1
#SBATCH --mem-per-gpu=48GB
#SBATCH --output=./outputs/outputfile_%j.out

# Navigate to project directory and activate environment
pushd /path/to/your/project/src/
conda activate /path/to/conda_env/myenv

python main_stage1.py --split train --llama_model_type LlamaModel --exp-name exp_stage1 --batch-size 2  --oti-steps 200 --seq-len 512 --learning-rate 0.0002 --percent 100 --pred-len 96