#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ag724@ic.ac.uk
#SBATCH --output=model-comparison-cah-%j.out
#SBATCH --time=4:00:00  # Request 4 hours of runtime

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
pip install rouge_score tqdm sentence-transformers huggingface_hub
pip install --upgrade "transformers>=4.36.0"

# Create output directory
mkdir -p model_comparison_results

# Force memory cleanup
python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None; import gc; gc.collect()"

echo "=== Running model comparison on CAH dataset ==="

# Run comparison with 20 black cards by default
# Adjust the parameters as needed
python src/model_comparison_cah.py \
    --data_dir ../python-backend/data/processed \
    --output_dir model_comparison_results \
    --num_samples 20 \
    --safe \
    --models t5-small t5-base facebook/bart-base

echo "=== Model comparison complete ==="

# Display summary of results
echo "=== Results Summary ==="
cat model_comparison_results/comparison_summary.json 