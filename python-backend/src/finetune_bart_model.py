import os
import argparse
import pandas as pd
import torch
import numpy as np
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset
import logging
import wandb
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_cah_data(data_path, safe=True):
    """Load CAH data from parquet file and prepare for fine-tuning."""
    # Determine which file to use based on safe parameter
    if safe:
        file_path = os.path.join(data_path, "cah_train_safe.parquet")
        val_path = os.path.join(data_path, "cah_val_safe.parquet")
    else:
        file_path = os.path.join(data_path, "cah_train.parquet")
        val_path = os.path.join(data_path, "cah_val.parquet")
    
    logger.info(f"Loading CAH data from {file_path}")
    
    try:
        train_df = pd.read_parquet(file_path)
        
        # Check if validation file exists, otherwise split train data
        if os.path.exists(val_path):
            val_df = pd.read_parquet(val_path)
        else:
            # Split train data into train and validation
            train_df = pd.read_parquet(file_path)
            # Shuffle the data
            train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
            # Split into train and validation (90/10)
            val_size = int(len(train_df) * 0.1)
            val_df = train_df[:val_size].copy()
            train_df = train_df[val_size:].copy()
        
        # Process the data for fine-tuning
        train_data = process_cah_data_for_finetuning(train_df)
        val_data = process_cah_data_for_finetuning(val_df)
        
        return train_data, val_data
        
    except Exception as e:
        logger.error(f"Error loading CAH data: {str(e)}")
        raise

def process_cah_data_for_finetuning(df):
    """Process CAH data for fine-tuning."""
    # Filter data
    black_cards = df[df['card_type'] == 'black']
    white_cards = df[df['card_type'] == 'white']
    
    logger.info(f"Found {len(black_cards)} black cards and {len(white_cards)} white cards")
    
    # Create training examples
    examples = []
    
    # For each black card, find suitable white card completions
    for _, black_row in black_cards.iterrows():
        black_text = black_row['text']
        
        # Sample white cards (in a real scenario, you'd use actual matches)
        # Here we're randomly sampling, but ideally you'd use actual played combinations
        sample_white_cards = white_cards.sample(min(5, len(white_cards)), random_state=int(hash(black_text)) % 10000)
        
        for _, white_row in sample_white_cards.iterrows():
            white_text = white_row['text']
            
            # Create input-output pair
            input_text = f"Complete this card: {black_text}"
            
            # For cards with blanks, replace the blank with the white card text
            if "_" in black_text:
                output_text = white_text
            else:
                # For cards without explicit blanks, just use the white card as the completion
                output_text = white_text
            
            examples.append({
                "input": input_text,
                "output": output_text,
                "black_card": black_text,
                "white_card": white_text
            })
    
    logger.info(f"Created {len(examples)} training examples")
    return examples

def create_dataset(examples):
    """Convert examples to HuggingFace Dataset."""
    return Dataset.from_pandas(pd.DataFrame(examples))

def tokenize_function(examples, tokenizer, max_input_length=128, max_output_length=64):
    """Tokenize the examples for training."""
    model_inputs = tokenizer(
        examples["input"],
        max_length=max_input_length,
        truncation=True,
        padding="max_length",
    )
    
    # Set up the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["output"],
            max_length=max_output_length,
            truncation=True,
            padding="max_length",
        )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def finetune_model(args):
    """Fine-tune a model on CAH data."""
    # Load data
    train_examples, val_examples = load_cah_data(args.data_dir, args.safe)
    
    # Create datasets
    train_dataset = create_dataset(train_examples)
    val_dataset = create_dataset(val_examples)
    
    logger.info(f"Created training dataset with {len(train_dataset)} examples")
    logger.info(f"Created validation dataset with {len(val_dataset)} examples")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    
    # If the tokenizer doesn't have a pad token, set it
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Tokenize the datasets
    tokenized_train_dataset = train_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    
    tokenized_val_dataset = val_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        remove_columns=val_dataset.column_names,
    )
    
    # Set up training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(args.output_dir, f"{args.model_name.split('/')[-1]}-cah"),
        eval_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=args.epochs,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),  # Use mixed precision if available
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=100,
        save_strategy="epoch",
        report_to="wandb" if args.use_wandb else "none",
    )
    
    # Set up data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )
    
    # Set up trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Start training
    logger.info("Starting training...")
    trainer.train()
    
    # Save the fine-tuned model
    model_save_path = os.path.join(args.output_dir, f"{args.model_name.split('/')[-1]}-cah-finetuned")
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    
    logger.info(f"Model saved to {model_save_path}")
    
    return model_save_path

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a model on CAH data")
    parser.add_argument("--data_dir", type=str, default="../python-backend/data/processed",
                       help="Directory containing processed CAH data")
    parser.add_argument("--output_dir", type=str, default="./finetuned_models",
                       help="Directory to save fine-tuned models")
    parser.add_argument("--model_name", type=str, default="facebook/bart-base",
                       help="Model to fine-tune")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of epochs to train for")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                       help="Learning rate for training")
    parser.add_argument("--safe", action="store_true", default=True,
                      help="Use safe dataset version")
    parser.add_argument("--use_wandb", action="store_true",
                      help="Use Weights & Biases for tracking")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize wandb if requested
    if args.use_wandb:
        wandb.init(
            project="cah-finetuning",
            name=f"{args.model_name.split('/')[-1]}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            config=vars(args),
        )
    
    # Fine-tune the model
    model_path = finetune_model(args)
    
    logger.info(f"Fine-tuning complete! Model saved to {model_path}")

if __name__ == "__main__":
    main() 