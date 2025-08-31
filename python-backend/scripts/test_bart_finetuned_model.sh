#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ag724@ic.ac.uk
#SBATCH --output=test-finetuned-cah-%j.out
#SBATCH --time=2:00:00  # Request 2 hours of runtime for testing

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
pip install transformers datasets pandas torch rouge_score

# Create output directory
mkdir -p test_results

# Force memory cleanup
python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None; import gc; gc.collect()"

echo "=== Testing fine-tuned model on CAH dataset ==="

# Test the fine-tuned model
# Replace with your actual model path after fine-tuning
MODEL_PATH="finetuned_models/bart-base-cah-finetuned"

python src/test_finetuned_model.py \
    --model_path $MODEL_PATH \
    --data_dir ../python-backend/data/processed \
    --output_dir finetuned_test_results \
    --num_samples 20 \
    --safe

echo "=== Testing complete ==="

# Display results location
echo "=== Test results saved to finetuned_test_results/ ===" 