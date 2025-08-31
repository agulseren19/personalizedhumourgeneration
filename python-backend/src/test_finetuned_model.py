#!/usr/bin/env python
# Test script for fine-tuned CAH models

import os
import argparse
import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForSequenceClassification
import logging
from rouge_score import rouge_scorer
from tqdm import tqdm
import json
from datetime import datetime
import random

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_test_data(data_path, num_samples=20, safe=True):
    """Load CAH data for testing."""
    # Determine which file to use based on safe parameter
    if safe:
        file_path = os.path.join(data_path, "cah_test_safe.parquet")
        # If test file doesn't exist, use train file
        if not os.path.exists(file_path):
            file_path = os.path.join(data_path, "cah_train_safe.parquet")
    else:
        file_path = os.path.join(data_path, "cah_test.parquet")
        # If test file doesn't exist, use train file
        if not os.path.exists(file_path):
            file_path = os.path.join(data_path, "cah_train.parquet")
    
    logger.info(f"Loading test data from {file_path}")
    
    try:
        df = pd.read_parquet(file_path)
        
        # Separate black and white cards
        black_cards = df[df['card_type'] == 'black']
        white_cards = df[df['card_type'] == 'white']
        
        logger.info(f"Loaded {len(black_cards)} black cards and {len(white_cards)} white cards")
        
        # Sample a subset of black cards
        if len(black_cards) > num_samples:
            black_cards = black_cards.sample(num_samples, random_state=42)
        
        # Create test samples
        test_samples = []
        for _, row in black_cards.iterrows():
            black_card_text = row['text']
            # Find sample white card completions (in a real game these would be player choices)
            sample_white_cards = white_cards.sample(3, random_state=int(hash(black_card_text)) % 10000)
            
            # Create a test sample
            test_sample = {
                "black_card": black_card_text,
                "formatted_input": f"Complete this card: {black_card_text}",
                "reference_white_cards": sample_white_cards['text'].tolist()
            }
            test_samples.append(test_sample)
        
        logger.info(f"Created {len(test_samples)} test samples")
        return test_samples
        
    except Exception as e:
        logger.error(f"Error loading test data: {str(e)}")
        raise

def extract_completion(text, black_card):
    """Extract just the completion part from generated text."""
    # Find the blank position in the black card
    blank_pos = black_card.find("_")
    if blank_pos == -1:
        # If no blank, just return the text
        return text.strip()
    
    # Try different extraction strategies
    # 1. Look for text after "Answer:" or similar
    for marker in ["Answer:", "A:", "Response:", "Completion:"]:
        if marker in text:
            return text.split(marker, 1)[1].strip()
    
    # 2. If the black card text appears in the generated text, try to extract what comes after the blank
    if black_card in text:
        parts = text.split(black_card, 1)[1].strip()
        # Remove any trailing text that might be instructions or formatting
        for end_marker in [".", "!", "?", "\n"]:
            if end_marker in parts:
                return parts.split(end_marker, 1)[0].strip()
        return parts
    
    # 3. If we can't find a good extraction, just return the shortest line that's not empty
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if lines:
        return min(lines, key=len)
    
    return text.strip()

def evaluate_humor(black_card, white_cards, model_name="mohameddhiab/humor-no-humor"):
    """Evaluate humor of white card completions using a humor classification model."""
    try:
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Move model to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        
        humor_scores = []
        for white_card in white_cards:
            # Format the combined text
            combined_text = black_card.replace("_", white_card) if "_" in black_card else f"{black_card} {white_card}"
            
            # Tokenize and get model prediction
            inputs = tokenizer(combined_text, return_tensors="pt", truncation=True).to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)
                # Get probability for humor class (assuming binary classification where 1 is humor)
                humor_score = probs[0, 1].item()
                humor_scores.append(humor_score)
        
        return humor_scores
    except Exception as e:
        logger.error(f"Error evaluating humor: {str(e)}")
        return [0.0] * len(white_cards)  # Return zeros in case of error

def test_model(model_path, test_samples, output_dir="test_results"):
    """Test a fine-tuned model on CAH examples."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Load model and tokenizer
    logger.info(f"Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    
    # Move model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    # Set up results structure
    results = {
        "model_path": model_path,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "device": device,
        "samples": [],
        "metrics": {}
    }
    
    # Process test samples
    generation_times = []
    rouge_scores = {
        "rouge1": [],
        "rouge2": [],
        "rougeL": []
    }
    humor_scores = []  # Track humor scores across all samples
    
    logger.info(f"Testing model on {len(test_samples)} samples")
    for sample in tqdm(test_samples):
        # Format input
        input_text = sample["formatted_input"]
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        
        # Generate output
        start_time = datetime.now()
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_length=64,
                num_beams=4,
                num_return_sequences=3,
                no_repeat_ngram_size=2,
                early_stopping=True,
                temperature=0.8,
                do_sample=True
            )
        generation_time = (datetime.now() - start_time).total_seconds()
        generation_times.append(generation_time)
        
        # Decode outputs
        raw_generated_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
        
        # Post-process to extract just the completion
        generated_texts = []
        for text in raw_generated_texts:
            completion = extract_completion(text, sample["black_card"])
            generated_texts.append(completion)
        
        # Calculate ROUGE scores
        sample_rouge_scores = {
            "rouge1": [],
            "rouge2": [],
            "rougeL": []
        }
        
        for reference in sample["reference_white_cards"]:
            for generated in generated_texts:
                # ROUGE scores
                scores = scorer.score(reference, generated)
                sample_rouge_scores["rouge1"].append(scores["rouge1"].fmeasure)
                sample_rouge_scores["rouge2"].append(scores["rouge2"].fmeasure)
                sample_rouge_scores["rougeL"].append(scores["rougeL"].fmeasure)
        
        # Use best scores for this sample
        if sample_rouge_scores["rouge1"]:  # Check if we have any scores
            rouge_scores["rouge1"].append(max(sample_rouge_scores["rouge1"]))
            rouge_scores["rouge2"].append(max(sample_rouge_scores["rouge2"]))
            rouge_scores["rougeL"].append(max(sample_rouge_scores["rougeL"]))
        
        # Evaluate humor scores for generated texts
        sample_humor_scores = evaluate_humor(sample["black_card"], generated_texts)
        if sample_humor_scores:
            humor_scores.append(max(sample_humor_scores))  # Track best humor score for metrics
        
        # Store sample result
        sample_result = {
            "black_card": sample["black_card"],
            "formatted_input": sample["formatted_input"],
            "reference_white_cards": sample["reference_white_cards"],
            "generated_texts": generated_texts,
            "generation_time_seconds": generation_time,
            "rouge_scores": {
                "rouge1": max(sample_rouge_scores["rouge1"]) if sample_rouge_scores["rouge1"] else 0,
                "rouge2": max(sample_rouge_scores["rouge2"]) if sample_rouge_scores["rouge2"] else 0,
                "rougeL": max(sample_rouge_scores["rougeL"]) if sample_rouge_scores["rougeL"] else 0
            },
            "humor_scores": sample_humor_scores  # Add humor scores to sample results
        }
        
        results["samples"].append(sample_result)
    
    # Compute aggregate metrics
    results["metrics"] = {
        "avg_generation_time": sum(generation_times) / len(generation_times) if generation_times else 0,
        "avg_rouge1": sum(rouge_scores["rouge1"]) / len(rouge_scores["rouge1"]) if rouge_scores["rouge1"] else 0,
        "avg_rouge2": sum(rouge_scores["rouge2"]) / len(rouge_scores["rouge2"]) if rouge_scores["rouge2"] else 0,
        "avg_rougeL": sum(rouge_scores["rougeL"]) / len(rouge_scores["rougeL"]) if rouge_scores["rougeL"] else 0,
        "avg_humor_score": sum(humor_scores) / len(humor_scores) if humor_scores else 0  # Add average humor score
    }
    
    # Log metrics
    logger.info(f"Average generation time: {results['metrics']['avg_generation_time']:.3f}s")
    logger.info(f"Average ROUGE-1: {results['metrics']['avg_rouge1']:.4f}")
    logger.info(f"Average ROUGE-2: {results['metrics']['avg_rouge2']:.4f}")
    logger.info(f"Average ROUGE-L: {results['metrics']['avg_rougeL']:.4f}")
    logger.info(f"Average humor score: {results['metrics']['avg_humor_score']:.4f}")
    
    # Save results
    model_name = os.path.basename(model_path)
    output_file = os.path.join(output_dir, f"{model_name}_test_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_file}")
    
    # Print a few examples
    logger.info("\nExample generations:")
    for i in range(min(3, len(results["samples"]))):
        sample = results["samples"][i]
        logger.info(f"\nBlack card: {sample['black_card']}")
        logger.info(f"Generated: {sample['generated_texts'][0]}")
        if "humor_scores" in sample and sample["humor_scores"]:
            logger.info(f"Humor score: {sample['humor_scores'][0]:.4f}")
        logger.info(f"Reference: {sample['reference_white_cards'][0]}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Test fine-tuned CAH models")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the fine-tuned model")
    parser.add_argument("--data_dir", type=str, default="../python-backend/data/processed",
                       help="Directory containing processed CAH data")
    parser.add_argument("--output_dir", type=str, default="./test_results",
                       help="Directory to save test results")
    parser.add_argument("--num_samples", type=int, default=20,
                       help="Number of black cards to sample for testing")
    parser.add_argument("--safe", action="store_true", default=True,
                      help="Use safe dataset version")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load test data
    test_samples = load_test_data(args.data_dir, args.num_samples, args.safe)
    
    # Test the model
    test_model(args.model_path, test_samples, args.output_dir)

if __name__ == "__main__":
    main() 