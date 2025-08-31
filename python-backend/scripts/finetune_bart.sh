#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ag724@ic.ac.uk
#SBATCH --output=finetune-cah-%j.out
#SBATCH --time=8:00:00  # Request 8 hours of runtime for training

# Set up Python environment
export PATH=/vol/bitbucket/ag724/cahvenv/bin/:$PATH
source activate

# Set up CUDA
source /vol/cuda/12.0.0/setup.sh
/usr/bin/nvidia-smi
uptime

# Go to project directory
cd /vol/bitbucket/ag724/individual_project/cah/personalized-humour-generation/python-backend

# Install required packages if not already installed
pip install transformers datasets pandas torch wandb

# Create output directory
mkdir -p finetuned_models

# Force memory cleanup
python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None; import gc; gc.collect()"

echo "=== Starting fine-tuning on CAH dataset ==="

# Run fine-tuning with BART-base model
python src/finetune_bart_model.py \
    --data_dir ../python-backend/data/processed \
    --output_dir finetuned_models \
    --model_name facebook/bart-base \
    --batch_size 8 \
    --epochs 3 \
    --learning_rate 5e-5 \
    --safe

echo "=== Fine-tuning complete ==="

# Display model path
echo "=== Fine-tuned model saved to finetuned_models/bart-base-cah-finetuned ===" 