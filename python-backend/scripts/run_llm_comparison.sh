#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ag724@ic.ac.uk
#SBATCH --output=llm-comparison-cah-%j.out
#SBATCH --time=8:00:00  # Request 8 hours of runtime for larger models

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
pip install rouge_score tqdm sentence-transformers bitsandbytes accelerate huggingface_hub
pip install --upgrade "transformers>=4.36.0"

# Create output directory
mkdir -p llm_comparison_results

# Force memory cleanup
python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None; import gc; gc.collect()"

# Login to Hugging Face to access LLaMA 3.1
echo "Logging in to Hugging Face to access LLaMA 3.1"
# Source the token file to set HF_TOKEN
source set_hf_token.sh
# Create HF credentials directory if it doesn't exist
mkdir -p ~/.huggingface
# Write token to the credentials file
echo "hf_token: $HF_TOKEN" > ~/.huggingface/token

# Verify login
echo "Verifying Hugging Face login..."
python -c "from huggingface_hub import HfApi; print('Login successful!' if HfApi().whoami() else 'Login failed!')"

echo "=== Running LLM comparison on CAH dataset ==="

# Run comparison with specific models
# Now using LLaMA 3.1 and the correct DeepSeek-V2-Chat model
python src/model_comparison_cah.py \
    --data_dir ../python-backend/data/processed \
    --output_dir llm_comparison_results \
    --num_samples 10 \
    --safe \
    --models facebook/bart-base meta-llama/Meta-Llama-3.1-8B deepseek-ai/DeepSeek-V2-Chat

echo "=== LLM comparison complete ==="

# Display summary of results
echo "=== Results Summary ==="
cat llm_comparison_results/comparison_summary.json 