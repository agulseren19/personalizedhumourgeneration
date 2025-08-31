import pandas as pd
import numpy as np
from pathlib import Path
import torch
from transformers import (
    AutoModelForSeq2SeqLM, 
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
from accelerate import init_empty_weights
from bitsandbytes.cextension import COMPILED_WITH_CUDA
import argparse 
try:
    from transformers import BitsAndBytesConfig
except ImportError:
    from transformers.utils.bitsandbytes import BitsAndBytesConfig
from datasets import Dataset
import evaluate
from sklearn.model_selection import train_test_split
import logging
import os
import gc
import psutil

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disable wandb
os.environ["WANDB_DISABLED"] = "true"

# Set PyTorch memory allocation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True,garbage_collection_threshold:0.6"

# Force CPU for tokenizer operations
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# For tracking memory in notebooks
memory_tracking_enabled = False
memory_tracking_callback = None

def enable_memory_tracking(callback_fn=None):
    """Enable memory tracking with optional callback function."""
    global memory_tracking_enabled, memory_tracking_callback
    memory_tracking_enabled = True
    memory_tracking_callback = callback_fn
    logger.info("Memory tracking enabled")

def track_memory(step_name=""):
    """Track memory if enabled."""
    if not memory_tracking_enabled:
        return
    
    # Get memory stats
    process = psutil.Process(os.getpid())
    ram_usage = process.memory_info().rss / 1e9  # GB
    
    gpu_allocated = 0
    if torch.cuda.is_available():
        gpu_allocated = torch.cuda.memory_allocated() / 1e9  # GB
    
    # Log memory
    logger.info(f"Memory at {step_name}: RAM={ram_usage:.2f}GB, GPU={gpu_allocated:.2f}GB")
    
    # Call callback if provided
    if memory_tracking_callback:
        memory_tracking_callback()

def log_memory_usage():
    """Log current memory usage."""
    process = psutil.Process(os.getpid())
    ram_usage = process.memory_info().rss / 1e9  # GB
    logger.info(f"RAM usage: {ram_usage:.2f} GB")
    
    if torch.cuda.is_available():
        gpu_allocated = torch.cuda.memory_allocated() / 1e9  # GB
        gpu_reserved = torch.cuda.memory_reserved() / 1e9  # GB
        logger.info(f"GPU allocated: {gpu_allocated:.2f} GB, reserved: {gpu_reserved:.2f} GB")

def clear_gpu_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        # Wait for CUDA operations to complete
        torch.cuda.synchronize()
        logger.info(f"GPU memory cleared. Current usage: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

def get_last_checkpoint(output_dir):
    """Get the last checkpoint from the output directory."""
    if not os.path.exists(output_dir):
        return None
    
    checkpoints = [
        f for f in os.listdir(output_dir) 
        if f.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, f))
    ]
    
    if not checkpoints:
        return None
    
    # Get the checkpoint with the highest step number
    last_checkpoint = max(
        checkpoints,
        key=lambda x: int(x.split("-")[1])
    )
    
    return os.path.join(output_dir, last_checkpoint)

def load_data():
    """Load and prepare the safe dataset."""
    track_memory("before_load_data")
    data_dir = Path('data/processed')
    df = pd.read_parquet(data_dir / 'cah_train_safe.parquet')
    
    # Create input-output pairs
    data = {
        'input_text': [],
        'target_text': []
    }
    
    # For black cards, pair with placeholder response
    black_cards = df[df['card_type'] == 'black']['text'].tolist()
    for card in black_cards[:50]:  # Use only 50 black cards (reduced from 100)
        data['input_text'].append(card)
        data['target_text'].append("Generate a humorous response")
    
    # For white cards, use as both input and target
    white_cards = df[df['card_type'] == 'white']['text'].tolist()
    for card in white_cards[:150]:  # Use only 150 white cards (reduced from 200)
        data['input_text'].append(card)
        data['target_text'].append(card)
    
    # Create HuggingFace dataset
    dataset = Dataset.from_dict(data)
    
    # For quick comparison, use only 1% of the data instead of 5%
    dataset = dataset.shuffle(seed=42).select(range(len(dataset) // 200))  # Reduced further by dividing by 200 instead of 100
    logger.info(f"Using {len(dataset)} examples for quick model comparison")
    track_memory("after_load_data")
    return dataset

def preprocess_function(examples, tokenizer, max_length=24):  # Reduced max length further from 32 to 24
    """Preprocess the data for the model."""
    model_inputs = tokenizer(
        examples['input_text'],
        max_length=max_length,
        padding='max_length',
        truncation=True
    )
    
    # Setup the tokenizer for targets
    labels = tokenizer(
        examples['target_text'],
        max_length=max_length,
        padding='max_length',
        truncation=True
    )

    model_inputs['labels'] = labels['input_ids']
    return model_inputs

def prepare_datasets(dataset, tokenizer):
    """Prepare training and validation datasets."""
    track_memory("before_prepare_datasets")
    # Split dataset
    train_test = dataset.train_test_split(test_size=0.1, seed=42)
    
    # Process in smaller batches to reduce memory usage
    batch_size = 32
    
    def process_dataset_in_batches(dataset):
        processed_datasets = []
        for i in range(0, len(dataset), batch_size):
            batch = dataset.select(range(i, min(i + batch_size, len(dataset))))
            processed = batch.map(
                lambda x: preprocess_function(x, tokenizer),
                batched=True,
                remove_columns=dataset.column_names
            )
            processed_datasets.append(processed)
            # Force garbage collection after each batch
            gc.collect()
        
        # Combine all processed batches
        from datasets import concatenate_datasets
        return concatenate_datasets(processed_datasets)
    
    # Process train and validation datasets
    train_dataset = process_dataset_in_batches(train_test['train'])
    val_dataset = process_dataset_in_batches(train_test['test'])
    
    track_memory("after_prepare_datasets")
    return train_dataset, val_dataset

def compute_metrics(eval_preds):
    """Compute evaluation metrics."""
    metric = evaluate.load("rouge")
    predictions, labels = eval_preds
    
    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 with pad token id
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Compute ROUGE scores
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    
    return {k: round(v * 100, 4) for k, v in result.items()}

def train_model(model_name, train_dataset, val_dataset, tokenizer):
    """Train a model and return the trainer."""
    track_memory("before_train_model")
    output_dir = f"models/{model_name.split('/')[-1]}"
    
    # Check for existing checkpoint
    last_checkpoint = get_last_checkpoint(output_dir)
    
    try:
        # Try loading with different memory optimization approaches
        model = None
        
        # First try: 8-bit quantization if available
        if torch.cuda.is_available() and COMPILED_WITH_CUDA:
            try:
                logger.info(f"Attempting to load {model_name} with 8-bit quantization")
                # Use 8-bit quantization for memory efficiency
        quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_use_double_quant=True,
        )
        
        model_kwargs = {
            "quantization_config": quantization_config,
            "device_map": "auto",
            "torch_dtype": torch.float16,
            "low_cpu_mem_usage": True,
        }
        
                # Clear cache before loading
                torch.cuda.empty_cache()
                gc.collect()
                
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **model_kwargs)
                logger.info("Successfully loaded model with 8-bit quantization")
            except Exception as e:
                logger.warning(f"Failed to load with 8-bit quantization: {str(e)}")
                # Clean up in case of partial loading
                if model is not None:
                    del model
                torch.cuda.empty_cache()
                gc.collect()
                model = None
        
        # Second try: FP16 precision
        if model is None and torch.cuda.is_available():
            try:
                logger.info(f"Attempting to load {model_name} with FP16 precision")
                model_kwargs = {
                    "device_map": "auto",
                    "torch_dtype": torch.float16,
                    "low_cpu_mem_usage": True,
                }
                
                # Clear cache before loading
                torch.cuda.empty_cache()
                gc.collect()
                
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **model_kwargs)
                logger.info("Successfully loaded model with FP16 precision")
            except Exception as e:
                logger.warning(f"Failed to load with FP16 precision: {str(e)}")
                # Clean up in case of partial loading
                if model is not None:
                    del model
                torch.cuda.empty_cache()
                gc.collect()
                model = None
        
        # Third try: CPU fallback (will be very slow but at least it works)
        if model is None:
            logger.warning(f"Falling back to CPU for {model_name} (will be very slow)")
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="cpu")
            logger.info("Successfully loaded model on CPU")
        
        # Enable gradient checkpointing if on GPU
        if torch.cuda.is_available() and next(model.parameters()).device.type == 'cuda':
        model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
        
        # Log memory usage after model loading
        log_memory_usage()
        track_memory("after_model_load")
        
        # Common training parameters - adjust based on where model is loaded
        if next(model.parameters()).device.type == 'cuda':
            batch_size = 1
            grad_accum = 32
        else:
            # For CPU, we need to use even smaller batches and more accumulation
        batch_size = 1
            grad_accum = 64
        
        max_length = 24
        
        # Training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            eval_strategy="steps",
            eval_steps=100,
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            weight_decay=0.01,
            save_total_limit=1,
            num_train_epochs=0.02,
            predict_with_generate=True,
            fp16=next(model.parameters()).device.type == 'cuda',  # Only use fp16 if on GPU
            logging_dir=f"logs/{model_name.split('/')[-1]}",
            logging_steps=50,
            save_strategy="steps",
            save_steps=100,
            load_best_model_at_end=True,
            metric_for_best_model="rouge1",
            report_to="none",
            generation_max_length=max_length,
            generation_num_beams=1,
            gradient_checkpointing=next(model.parameters()).device.type == 'cuda',  # Only if on GPU
            optim="adafactor",
            max_grad_norm=0.5,
            dataloader_num_workers=0,
            group_by_length=True,
            # Memory optimization
            deepspeed=None,
            ddp_find_unused_parameters=False,
            tf32=next(model.parameters()).device.type == 'cuda',  # Only if on GPU
        )
        
        # Data collator with model-specific padding
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=-100,
            pad_to_multiple_of=8
        )
        
        # Initialize trainer
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        
        track_memory("after_trainer_init")
        return trainer, last_checkpoint
        
    except Exception as e:
        logger.error(f"Error initializing model {model_name}: {str(e)}")
        raise

def evaluate_model(trainer, test_dataset):
    """Evaluate model performance."""
    track_memory("before_evaluate")
    results = trainer.evaluate(test_dataset)
    track_memory("after_evaluate")
    return results

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Evaluate a model for CAH without training.')
    parser.add_argument('--model', type=str, choices=['bart', 't5', 'all'], default='t5',
                      help='Model to evaluate: bart, t5, or all (default: t5)')
    args = parser.parse_args()
    
    global tokenizer  # Make tokenizer available to compute_metrics
    
    try:
        # Print GPU info
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            logger.info(f"Initial GPU memory usage: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        else:
            logger.warning("No GPU available!")
        
        # Log initial memory usage
        log_memory_usage()
        
        # Load data
        logger.info("Loading data...")
        dataset = load_data()
        
        # Define available models
        available_models = {
            "bart": ("facebook/bart-small", "BART"),
            "t5": ("t5-small", "T5")
        }
        
        # Select models to run based on the command-line argument
        models_to_run = {}
        if args.model == 'all':
            models_to_run = {k[0]: k[1] for k in available_models.values()}
        else:
            model_name, model_type = available_models[args.model]
            models_to_run = {model_name: model_type}
        
        logger.info(f"Running inference with models: {list(models_to_run.values())}")
        
        results = {}
        
        # Process one model at a time with memory cleanup between
        for model_name, model_type in models_to_run.items():
            try:
                logger.info(f"\nRunning inference for {model_type}...")
                
                # Ensure clean GPU memory
                clear_gpu_memory()
                log_memory_usage()
                
                # Process this model completely
                with torch.device('cpu'):
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                
                # Prepare datasets
                train_dataset, val_dataset = prepare_datasets(dataset, tokenizer)
                
                # Force memory cleanup before model loading
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.synchronize()
                
                # Load model for inference only
                try:
                    logger.info(f"Loading model {model_name} for inference...")
                    
                    # Try loading in different precision modes
                    if torch.cuda.is_available():
                        try:
                            # First try FP16 precision
                            model = AutoModelForSeq2SeqLM.from_pretrained(
                                model_name,
                                torch_dtype=torch.float16,
                                device_map="auto",
                                low_cpu_mem_usage=True
                            )
                            logger.info("Successfully loaded model with FP16 precision")
                        except Exception as e:
                            logger.warning(f"Failed to load with FP16 precision: {str(e)}")
                            # Clean up
                            torch.cuda.empty_cache()
                            gc.collect()
                            
                            # Try regular precision
                            model = AutoModelForSeq2SeqLM.from_pretrained(
                                model_name,
                                device_map="auto",
                                low_cpu_mem_usage=True
                            )
                            logger.info("Successfully loaded model with regular precision")
                    else:
                        # CPU fallback
                        model = AutoModelForSeq2SeqLM.from_pretrained(
                            model_name,
                            device_map="cpu"
                        )
                        logger.info("Successfully loaded model on CPU")
                    
                    # Log memory after model loading
                    log_memory_usage()
                    
                    # Setup trainer for evaluation only
                    batch_size = 1 if next(model.parameters()).device.type == 'cuda' else 1
                    training_args = Seq2SeqTrainingArguments(
                        output_dir=f"models/{model_name.split('/')[-1]}",
                        per_device_eval_batch_size=batch_size,
                        predict_with_generate=True,
                        generation_max_length=24,
                        generation_num_beams=1,
                        fp16=next(model.parameters()).device.type == 'cuda',
                        report_to="none",
                    )
                    
                    # Data collator
                    data_collator = DataCollatorForSeq2Seq(
                        tokenizer,
                        model=model,
                        label_pad_token_id=-100,
                        pad_to_multiple_of=8
                    )
                    
                    # Initialize trainer for evaluation only
                    trainer = Seq2SeqTrainer(
                        model=model,
                        args=training_args,
                        data_collator=data_collator,
                        compute_metrics=compute_metrics,
                    )
                    
                    # Create a small evaluation dataset
                    small_val_dataset = val_dataset.select(range(min(len(val_dataset), 30)))
                    
                    # Run evaluation
                    logger.info(f"Running evaluation for {model_type}...")
                    eval_results = trainer.evaluate(small_val_dataset)
                    results[model_type] = eval_results
                    
                    logger.info(f"{model_type} Inference Results:")
                    for metric, value in eval_results.items():
                        logger.info(f"{metric}: {value}")
                    
                finally:
                    # Clean up
                    if 'model' in locals():
                        del model
                    if 'trainer' in locals():
                    del trainer
                    if 'small_val_dataset' in locals():
                        del small_val_dataset
                    del train_dataset
                    del val_dataset
                    clear_gpu_memory()
                    gc.collect()
                
            except Exception as e:
                logger.error(f"Error processing {model_type}: {str(e)}")
                continue
        
        # Compare results
        if results:
            logger.info("\nModel Comparison Results:")
            best_model = None
            best_rouge1 = -1
            
            for model_type, metrics in results.items():
                logger.info(f"\n{model_type}:")
                for metric, value in metrics.items():
                    logger.info(f"{metric}: {value}")
                    if metric == 'eval_rouge1' and value > best_rouge1:
                        best_rouge1 = value
                        best_model = model_type
            
            logger.info(f"\nBest performing model: {best_model} with ROUGE-1 score: {best_rouge1}")
        else:
            logger.warning("No results available for comparison.")
            
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    # Make sure GPU memory is clear before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.synchronize()
    
    main() 