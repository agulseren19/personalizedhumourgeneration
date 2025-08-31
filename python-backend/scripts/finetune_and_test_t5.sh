#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ag724@ic.ac.uk
#SBATCH --output=finetune-t5-cah-%j.out
#SBATCH --time=8:00:00  # Request 8 hours of runtime for training

# Set up Python environment - fix activation
export PATH=/vol/bitbucket/ag724/cahvenv/bin/:$PATH
# Properly activate the virtual environment
source /vol/bitbucket/ag724/cahvenv/bin/activate

# Set up CUDA
source /vol/cuda/12.0.0/setup.sh
/usr/bin/nvidia-smi
uptime

# Go to project directory
cd /vol/bitbucket/ag724/individual_project/cah/personalized-humour-generation/python-backend

# Verify Python environment and packages
which python
python --version
pip list | grep -E "transformers|pandas|torch|datasets"

# Install required packages if not already installed
pip install transformers datasets pandas torch rouge_score

# Create output directories
mkdir -p finetuned_models
mkdir -p logs
mkdir -p test_results

# Force memory cleanup
python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None; import gc; gc.collect()"

# Define model name and output paths
MODEL_NAME="t5-base"
OUTPUT_DIR="finetuned_models"
LOG_FILE="logs/finetune-t5-$(date +%Y%m%d-%H%M%S).out"

echo "=== Starting fine-tuning of $MODEL_NAME on CAH dataset ===" | tee -a $LOG_FILE
echo "Logs will be saved to $LOG_FILE" | tee -a $LOG_FILE

# Fine-tune the model
echo "=== Starting fine-tuning ===" | tee -a $LOG_FILE
python src/finetune_t5_cah.py \
    --model_name $MODEL_NAME \
    --output_dir $OUTPUT_DIR \
    --data_dir ../python-backend/data/processed \
    --batch_size 8 \
    --epochs 3 \
    --learning_rate 5e-5 \
    --safe | tee -a $LOG_FILE

# Force memory cleanup before testing
python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None; import gc; gc.collect()"

# Test the fine-tuned model
echo "=== Fine-tuning complete ===" | tee -a $LOG_FILE
echo "=== Starting model testing ===" | tee -a $LOG_FILE

FINETUNED_MODEL_PATH="$OUTPUT_DIR/${MODEL_NAME}-cah-finetuned"
TEST_OUTPUT_DIR="test_results"

python src/test_finetuned_model.py \
    --model_path $FINETUNED_MODEL_PATH \
    --data_dir ../python-backend/data/processed \
    --output_dir $TEST_OUTPUT_DIR \
    --num_samples 20 \
    --safe | tee -a $LOG_FILE

echo "=== Testing complete ===" | tee -a $LOG_FILE
echo "=== Results saved to $TEST_OUTPUT_DIR/${MODEL_NAME}-cah-finetuned_test_results.json ===" | tee -a $LOG_FILE 