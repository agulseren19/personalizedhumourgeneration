#!/bin/bash
# Simple script to test model comparison with minimal resources

# Create output directory
mkdir -p model_comparison_results

# Force memory cleanup
python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None; import gc; gc.collect()"

echo "=== Running minimal model comparison on CAH dataset ==="

# Run comparison with just 5 samples and smaller models
python src/model_comparison_cah.py \
    --data_dir ../python-backend/data/processed \
    --output_dir model_comparison_results \
    --num_samples 5 \
    --safe \
    --models t5-small facebook/bart-base

echo "=== Model comparison complete ==="

# Display summary of results
echo "=== Results Summary ==="
cat model_comparison_results/comparison_summary.json 