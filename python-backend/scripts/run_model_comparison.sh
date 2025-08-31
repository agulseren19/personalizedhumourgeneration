#!/bin/bash

echo "=== Running model inference comparison ==="
echo "This script will run T5 and BART inference separately to avoid memory issues"

# Make sure the GPU is cleared
echo "Clearing GPU memory..."
python -c "import torch; torch.cuda.empty_cache(); import gc; gc.collect()"

# First run T5
echo "=== Running inference with T5 model ==="
python src/model_comparison_reduced.py --model t5

# Clear GPU memory again
echo "Clearing GPU memory..."
python -c "import torch; torch.cuda.empty_cache(); import gc; gc.collect()"

# Wait a bit to ensure all memory is cleared
echo "Waiting for 5 seconds to ensure memory is cleared..."
sleep 5

# Then run BART
echo "=== Running inference with BART model ==="
python src/model_comparison_reduced.py --model bart

echo "=== Model inference comparison complete ===" 