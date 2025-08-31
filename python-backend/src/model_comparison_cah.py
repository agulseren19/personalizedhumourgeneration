import torch
import gc
import os
import json
import argparse
import pandas as pd
import time
import numpy as np
from datetime import datetime
from transformers import (
    AutoModelForSeq2SeqLM, 
    AutoTokenizer, 
    AutoModel,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoConfig
)
from rouge_score import rouge_scorer
import logging
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set memory options
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True,garbage_collection_threshold:0.8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def clear_gpu_memory():
    """Clear GPU memory cache thoroughly."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.synchronize()
        logger.info(f"GPU memory cleared. Current usage: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

def log_memory():
    """Log current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        logger.info(f"GPU memory: allocated={allocated:.2f} GB, reserved={reserved:.2f} GB")

def setup_llama_auth():
    """Set up authentication for Llama models"""
    import os
    
    # Check if HF_TOKEN is set
    if "HF_TOKEN" not in os.environ:
        logger.info("Logging in to Hugging Face to access LLaMA 3.1")
        try:
            # Use huggingface_hub for login
            from huggingface_hub import login
            
            # Try to read token from ~/.huggingface/token
            token_path = os.path.expanduser("~/.huggingface/token")
            if os.path.exists(token_path):
                with open(token_path, "r") as f:
                    token = f.read().strip()
                os.environ["HF_TOKEN"] = token
                logger.info("Hugging Face token set as environment variable HF_TOKEN")
            
            # Verify login
            login(token=os.environ.get("HF_TOKEN", None))
            logger.info("Login successful!")
            return True
        except Exception as e:
            logger.error(f"Failed to log in to Hugging Face: {str(e)}")
            return False
    else:
        logger.info("HF_TOKEN already set")
        return True

def load_deberta_model():
    """Load DeBERTa model for semantic similarity evaluation."""
    logger.info("Loading DeBERTa-v3-base for semantic evaluation...")
    try:
        # Use sentence-transformers wrapper for easier similarity computation
        model = SentenceTransformer('microsoft/deberta-v3-base')
        if torch.cuda.is_available():
            model = model.to('cuda')
        logger.info("DeBERTa model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading DeBERTa model: {str(e)}")
        return None

def compute_semantic_similarity(text1, text2, model):
    """Compute semantic similarity between two texts using DeBERTa."""
    try:
        # Encode texts to get embeddings
        embeddings1 = model.encode(text1, convert_to_tensor=True)
        embeddings2 = model.encode(text2, convert_to_tensor=True)
        
        # Compute cosine similarity
        similarity = torch.nn.functional.cosine_similarity(embeddings1.unsqueeze(0), embeddings2.unsqueeze(0))
        return similarity.item()
    except Exception as e:
        logger.error(f"Error computing semantic similarity: {str(e)}")
        return 0.0

def compute_humor_metrics(generated_text, reference_texts, deberta_model):
    """Compute advanced humor metrics using DeBERTa."""
    # Semantic similarity with references
    similarities = []
    for ref in reference_texts:
        similarity = compute_semantic_similarity(generated_text, ref, deberta_model)
        similarities.append(similarity)
    
    # Get best similarity score
    best_similarity = max(similarities) if similarities else 0
    
    # Compute additional metrics
    # 1. Length appropriateness (penalize if too short or too long)
    gen_length = len(generated_text.split())
    avg_ref_length = np.mean([len(ref.split()) for ref in reference_texts])
    length_score = 1.0 - min(abs(gen_length - avg_ref_length) / max(avg_ref_length, 10), 1.0)
    
    # 2. Humor keywords presence
    humor_keywords = {'funny', 'hilarious', 'laugh', 'joke', 'ridiculous', 'absurd', 'ironic', 'silly'}
    gen_words = set(generated_text.lower().split())
    humor_keyword_score = len(gen_words.intersection(humor_keywords)) / len(humor_keywords) if humor_keywords else 0
    
    # 3. Unexpectedness score (higher is better for humor)
    # This is a proxy - we assume higher semantic distance from average reference indicates surprise
    avg_similarity = np.mean(similarities) if similarities else 0
    unexpectedness_score = 1.0 - avg_similarity
    
    # Combine scores (weights can be adjusted)
    final_score = (
        0.4 * best_similarity +  # Semantic relevance
        0.2 * length_score +     # Appropriate length
        0.2 * humor_keyword_score +  # Humor indicators
        0.2 * unexpectedness_score    # Surprise factor
    )
    
    return {
        "semantic_similarity": best_similarity,
        "length_score": length_score,
        "humor_keyword_score": humor_keyword_score,
        "unexpectedness_score": unexpectedness_score,
        "final_humor_score": final_score
    }

def load_cah_data(data_path, num_samples=50, safe=True):
    """Load CAH data from parquet file and prepare samples for testing."""
    # Determine which file to use based on safe parameter
    if safe:
        file_path = os.path.join(data_path, "cah_train_safe.parquet")
    else:
        file_path = os.path.join(data_path, "cah_train.parquet")
    
    logger.info(f"Loading CAH data from {file_path}")
    
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
        logger.error(f"Error loading CAH data: {str(e)}")
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

def run_model_comparison(models, test_samples, output_dir="model_comparison_results"):
    """Run inference with multiple models on test samples and compare results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Load DeBERTa model for semantic evaluation
    deberta_model = load_deberta_model()
    
    # Track results for all models
    all_results = {}
    
    # Process each model
    for model_key, model_info in models.items():
        logger.info(f"Testing model: {model_key} ({model_info['model_name']})")
        
        # Create results structure
        model_results = {
            "model_name": model_info['model_name'],
            "model_size": model_info.get('model_size', 'unknown'),
            "model_type": model_info.get('model_type', 'encoder-decoder'),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "memory_stats": {},
            "samples": [],
            "metrics": {}
        }
        
        # Clear GPU memory before loading model
        clear_gpu_memory()
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            model_results["memory_stats"]["before_load"] = {
                "allocated_gb": allocated,
                "reserved_gb": reserved
            }
        
        try:
            # Load tokenizer
            logger.info(f"Loading tokenizer for {model_info['model_name']}...")
            start_time = time.time()
            
            # Special handling for DeepSeek models
            if "deepseek" in model_info['model_name'].lower():
                tokenizer = AutoTokenizer.from_pretrained(
                    model_info['model_name'],
                    trust_remote_code=True
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_info['model_name'])
                
            tokenizer_load_time = time.time() - start_time
            logger.info(f"Tokenizer loaded in {tokenizer_load_time:.2f} seconds")
            
            # Determine device
            if not torch.cuda.is_available() or torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory > 0.8:
                logger.info("Using CPU for inference")
                device = "cpu"
            else:
                logger.info("Using GPU for inference")
                device = "cuda"
            
            # Load model based on type
            logger.info(f"Loading model {model_info['model_name']}...")
            start_time = time.time()
            
            # Special handling for specific model families
            if "meta-llama" in model_info['model_name'].lower() or "llama-3" in model_info['model_name'].lower():
                # Ensure we're authenticated for gated models
                setup_llama_auth()
                
                try:
                    # Use direct model loading without sharded file expectation
                    model = AutoModelForCausalLM.from_pretrained(
                        model_info['model_name'],
                        torch_dtype=torch.float16,
                        device_map="auto",
                        low_cpu_mem_usage=True,
                        token=os.environ.get("HF_TOKEN", True)  # Use token for authentication
                    )
                    model_load_time = time.time() - start_time
                    logger.info(f"Llama model loaded in {model_load_time:.2f} seconds")
                except Exception as e:
                    logger.error(f"Error loading Llama model: {str(e)}")
                    raise
            elif "deepseek" in model_info['model_name'].lower():
                try:
                    # First load config with trust_remote_code
                    logger.info(f"Loading DeepSeek config with trust_remote_code=True")
                    config = AutoConfig.from_pretrained(
                        model_info['model_name'],
                        trust_remote_code=True
                    )
                    
                    # Set model_type if missing
                    if not hasattr(config, "model_type") or not config.model_type:
                        logger.info("Setting model_type for DeepSeek model")
                        config.model_type = "deepseek_chat"
                    
                    # Load model with trust_remote_code
                    logger.info(f"Loading DeepSeek model with trust_remote_code=True")
                    model = AutoModelForCausalLM.from_pretrained(
                        model_info['model_name'],
                        config=config,
                        torch_dtype=torch.float16,
                        device_map="auto",
                        low_cpu_mem_usage=True,
                        trust_remote_code=True
                    )
                    model_load_time = time.time() - start_time
                    logger.info(f"DeepSeek model loaded in {model_load_time:.2f} seconds")
                except Exception as e:
                    logger.error(f"Error loading DeepSeek model: {str(e)}")
                    raise
            else:
                # Regular model loading for other models
                model_type = model_info.get('model_type', 'encoder-decoder')
                model_size = model_info.get('model_size', 'unknown')
                
                if device == "cuda":
                    # Determine if we should use 8-bit quantization for large models
                    use_8bit = False
                    if model_size.lower() in ['7b', '8b', '13b', 'large'] or any(x in model_info['model_name'].lower() for x in ['7b', '8b', '13b', 'large']):
                        use_8bit = True
                        logger.info(f"Using 8-bit quantization for large model: {model_info['model_name']}")
                    
                    # Load with appropriate quantization
                    if model_type == 'decoder-only':
                        if use_8bit:
                            try:
                                quantization_config = BitsAndBytesConfig(
                                    load_in_8bit=True,
                                    bnb_8bit_use_double_quant=True,
                                    bnb_8bit_quant_type="nf4",
                                    bnb_8bit_compute_dtype=torch.float16
                                )
                                
                                model = AutoModelForCausalLM.from_pretrained(
                                    model_info['model_name'],
                                    quantization_config=quantization_config,
                                    device_map="auto",
                                    low_cpu_mem_usage=True
                                )
                            except Exception as e:
                                logger.warning(f"Failed to load with 8-bit quantization: {str(e)}. Falling back to FP16.")
                                model = AutoModelForCausalLM.from_pretrained(
                                    model_info['model_name'],
                                    torch_dtype=torch.float16,
                                    device_map="auto",
                                    low_cpu_mem_usage=True
                                )
                        else:
                            model = AutoModelForCausalLM.from_pretrained(
                                model_info['model_name'],
                                torch_dtype=torch.float16,
                                device_map="auto",
                                low_cpu_mem_usage=True
                            )
                    else:  # encoder-decoder
                        model = AutoModelForSeq2SeqLM.from_pretrained(
                            model_info['model_name'],
                            torch_dtype=torch.float16,
                            device_map="auto",
                            low_cpu_mem_usage=True
                        )
                else:
                    # Load on CPU
                    if model_type == 'decoder-only':
                        model = AutoModelForCausalLM.from_pretrained(model_info['model_name'])
                    else:  # encoder-decoder
                        model = AutoModelForSeq2SeqLM.from_pretrained(model_info['model_name'])
                
                model_load_time = time.time() - start_time
                logger.info(f"Model loaded in {model_load_time:.2f} seconds")
            
            # Log memory after loading
            log_memory()
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1e9
                reserved = torch.cuda.memory_reserved() / 1e9
                model_results["memory_stats"]["after_load"] = {
                    "allocated_gb": allocated,
                    "reserved_gb": reserved
                }
            
            # Process each test sample
            generation_times = []
            rouge_scores = {
                "rouge1": [],
                "rouge2": [],
                "rougeL": []
            }
            semantic_scores = []
            humor_scores = []
            
            logger.info(f"Processing {len(test_samples)} test samples...")
            for sample in tqdm(test_samples):
                # Format input based on model type
                if model_type == 'decoder-only':
                    # For decoder-only models, we need to format the prompt differently
                    # Special handling for different model families
                    if "llama-3" in model_info['model_name'].lower() or "meta-llama/meta-llama-3" in model_info['model_name'].lower():
                        # LLaMA 3.1 format
                        formatted_input = f"<|system|>\nYou are a helpful assistant that completes Cards Against Humanity prompts with humorous responses. Fill in the blank (marked by _) with a funny phrase.\n<|user|>\n{sample['formatted_input']}\n<|assistant|>\n"
                    elif "deepseek-v2" in model_info['model_name'].lower() or "deepseek-ai/DeepSeek-V2-Chat" in model_info['model_name'].lower():
                        # DeepSeek-V2-Chat format - using standard chat template
                        formatted_input = f"User: I'm playing Cards Against Humanity. Fill in the blank (marked by _) with a funny phrase: {sample['formatted_input']}\n\nA:"
                    elif "deepseek" in model_info['model_name'].lower():
                        formatted_input = f"<|user|>\nI'm playing Cards Against Humanity. Fill in the blank (marked by _) with a funny phrase: {sample['formatted_input']}\n<|assistant|>\n"
                    else:
                        formatted_input = f"Fill in the blank (marked by _) with a funny phrase: {sample['formatted_input']} Answer: "
                    
                    inputs = tokenizer(formatted_input, return_tensors="pt")
                    
                    # For GPT-2 specifically, ensure the tokenizer has the right padding token
                    if "gpt2" in model_info['model_name'].lower() or tokenizer.pad_token_id is None:
                        tokenizer.pad_token = tokenizer.eos_token
                else:
                    # For encoder-decoder models
                    # Extract the blank pattern from the black card
                    black_card = sample["black_card"]
                    blank_pattern = "_"
                    
                    # Format the input to make it clear we want to fill in the blank
                    formatted_input = f"Fill in the blank with something funny: {sample['formatted_input']}"
                    inputs = tokenizer(formatted_input, return_tensors="pt")
                
                if device == "cuda":
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                # Generate output
                start_time = time.time()
                with torch.no_grad():
                    if model_type == 'decoder-only':
                        # Generate with decoder-only model
                        try:
                            output_ids = model.generate(
                                **inputs,
                                max_new_tokens=64,  # Only generate new tokens
                                num_return_sequences=3,
                                do_sample=True,
                                temperature=0.8,
                                no_repeat_ngram_size=2,
                                early_stopping=True
                            )
                            
                            # Decode outputs - only take the newly generated part
                            generated_texts = []
                            input_length = inputs["input_ids"].shape[1]
                            
                            for ids in output_ids:
                                try:
                                    # Extract only the new tokens (excluding input)
                                    new_tokens = ids[input_length:]
                                    # Decode the new tokens
                                    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                                    generated_texts.append(generated_text)
                                except Exception as e:
                                    logger.error(f"Error decoding tokens: {str(e)}")
                                    generated_texts.append(f"Error: {str(e)}")
                                
                            # If we didn't get any valid outputs, use a fallback approach
                            if not any(generated_texts) or all(text.startswith("Error:") for text in generated_texts):
                                logger.warning("Using fallback decoding approach")
                                try:
                                    full_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
                                    generated_texts = []
                                    for full_text in full_texts:
                                        try:
                                            # Handle different model formats
                                            if "llama-3" in model_info['model_name'].lower() or "meta-llama/meta-llama-3" in model_info['model_name'].lower():
                                                if "<|assistant|>" in full_text:
                                                    answer = full_text.split("<|assistant|>", 1)[1].strip()
                                                    # Remove any trailing tags
                                                    if "<|" in answer:
                                                        answer = answer.split("<|", 1)[0].strip()
                                                    generated_texts.append(answer)
                                                else:
                                                    # Just take everything after the user prompt
                                                    answer = full_text.replace(sample["formatted_input"], "").strip()
                                                    generated_texts.append(answer)
                                            elif "deepseek-v2" in model_info['model_name'].lower() or "deepseek-ai/DeepSeek-V2-Chat" in model_info['model_name'].lower():
                                                # DeepSeek-V2-Chat format
                                                if "\n\nA:" in full_text:
                                                    answer = full_text.split("\n\nA:", 1)[1].strip()
                                                    # Remove any trailing tags
                                                    if " " in answer:
                                                        answer = answer.split(" ", 1)[0].strip()
                                                    generated_texts.append(answer)
                                                else:
                                                    # Just take everything after the user prompt
                                                    answer = full_text.replace(sample["formatted_input"], "").strip()
                                                    generated_texts.append(answer)
                                            elif "deepseek" in model_info['model_name'].lower():
                                                if "<|assistant|>" in full_text:
                                                    answer = full_text.split("<|assistant|>", 1)[1].strip()
                                                    generated_texts.append(answer)
                                                else:
                                                    # Just take everything after the user prompt
                                                    answer = full_text.replace(sample["formatted_input"], "").strip()
                                                    generated_texts.append(answer)
                                            elif "Answer:" in full_text:
                                                answer = full_text.split("Answer:", 1)[1].strip()
                                                generated_texts.append(answer)
                                            else:
                                                # Just take everything after the prompt
                                                answer = full_text.replace(sample["formatted_input"], "").strip()
                                                generated_texts.append(answer)
                                        except Exception as e:
                                            logger.error(f"Error in fallback text processing: {str(e)}")
                                            generated_texts.append(f"Fallback Error: {str(e)}")
                                except Exception as e:
                                    logger.error(f"Error in fallback decoding: {str(e)}")
                                    generated_texts = [f"Fallback Decoding Error: {str(e)}"] * 3
                            
                            # Post-process to extract just the completion
                            processed_texts = []
                            for text in generated_texts:
                                processed = extract_completion(text, sample["black_card"])
                                processed_texts.append(processed)
                            
                            # Use the processed texts if they're not empty
                            if any(processed_texts) and not all(t == "" for t in processed_texts):
                                generated_texts = processed_texts
                            
                            # Ensure we have at least some output
                            if not generated_texts:
                                generated_texts = ["No output generated"] * 3
                            elif len(generated_texts) < 3:
                                # Duplicate the last output to ensure we have 3
                                while len(generated_texts) < 3:
                                    generated_texts.append(generated_texts[-1])
                                    
                        except Exception as e:
                            logger.error(f"Error generating with decoder-only model: {str(e)}")
                            # Provide empty responses to continue evaluation
                            generated_texts = [f"Generation Error: {str(e)}"] * 3
                    else:
                        # Generate with encoder-decoder model
                        output_ids = model.generate(
                            **inputs,
                            max_length=64,
                            num_beams=4,
                            num_return_sequences=3,
                            no_repeat_ngram_size=2,
                            early_stopping=True,
                            temperature=0.8,
                            do_sample=True  # Enable sampling for more creative outputs
                        )
                        # Decode outputs
                        raw_generated_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
                        
                        # Post-process to extract just the completion
                        generated_texts = []
                        for text in raw_generated_texts:
                            # First try the extract_completion function
                            completion = extract_completion(text, sample["black_card"])
                            if completion and not completion.startswith("Fill in") and len(completion) < len(text):
                                generated_texts.append(completion)
                                continue
                                
                            # If that didn't work well, try other approaches
                            if ":" in text:
                                # Try to get content after the last colon
                                completion = text.split(":")[-1].strip()
                                generated_texts.append(completion)
                            elif "Fill in the blank" in text:
                                # Try to remove the instruction part
                                completion = text.replace("Fill in the blank with something funny:", "").strip()
                                completion = completion.replace(sample["formatted_input"], "").strip()
                                generated_texts.append(completion)
                            else:
                                # If no colon, just use the whole output
                                generated_texts.append(text.strip())
                        
                        # If we got empty completions, try a different approach
                        if not any(generated_texts) or all(t == "" for t in generated_texts):
                            # Just use the raw texts
                            generated_texts = [t.strip() for t in raw_generated_texts]
                        
                        # Ensure we have at least some output
                        if not generated_texts:
                            generated_texts = ["No output generated"] * 3
                        elif len(generated_texts) < 3:
                            # Duplicate the last output to ensure we have 3
                            while len(generated_texts) < 3:
                                generated_texts.append(generated_texts[-1])
                
                generation_time = time.time() - start_time
                generation_times.append(generation_time)
                
                # Calculate ROUGE scores
                sample_rouge_scores = {
                    "rouge1": [],
                    "rouge2": [],
                    "rougeL": []
                }
                
                # Calculate semantic and humor scores using DeBERTa
                sample_semantic_scores = []
                sample_humor_scores = []
                
                for reference in sample["reference_white_cards"]:
                    for generated in generated_texts:
                        # ROUGE scores
                        scores = scorer.score(reference, generated)
                        sample_rouge_scores["rouge1"].append(scores["rouge1"].fmeasure)
                        sample_rouge_scores["rouge2"].append(scores["rouge2"].fmeasure)
                        sample_rouge_scores["rougeL"].append(scores["rougeL"].fmeasure)
                        
                        # Compute DeBERTa-based metrics if model is loaded
                        if deberta_model:
                            humor_metrics = compute_humor_metrics(generated, sample["reference_white_cards"], deberta_model)
                            sample_semantic_scores.append(humor_metrics["semantic_similarity"])
                            sample_humor_scores.append(humor_metrics["final_humor_score"])
                
                # Use best scores for this sample
                if sample_rouge_scores["rouge1"]:  # Check if we have any scores
                    rouge_scores["rouge1"].append(max(sample_rouge_scores["rouge1"]))
                    rouge_scores["rouge2"].append(max(sample_rouge_scores["rouge2"]))
                    rouge_scores["rougeL"].append(max(sample_rouge_scores["rougeL"]))
                
                if sample_semantic_scores:
                    semantic_scores.append(max(sample_semantic_scores))
                
                if sample_humor_scores:
                    humor_scores.append(max(sample_humor_scores))
                
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
                    }
                }
                
                # Add semantic and humor scores if available
                if deberta_model:
                    best_humor_idx = np.argmax(sample_humor_scores) if sample_humor_scores else 0
                    sample_result["deberta_scores"] = {
                        "semantic_similarity": sample_semantic_scores[best_humor_idx] if sample_semantic_scores else 0,
                        "humor_score": sample_humor_scores[best_humor_idx] if sample_humor_scores else 0
                    }
                
                model_results["samples"].append(sample_result)
            
            # Compute aggregate metrics
            model_results["metrics"] = {
                "avg_generation_time": np.mean(generation_times),
                "avg_rouge1": np.mean(rouge_scores["rouge1"]) if rouge_scores["rouge1"] else 0,
                "avg_rouge2": np.mean(rouge_scores["rouge2"]) if rouge_scores["rouge2"] else 0,
                "avg_rougeL": np.mean(rouge_scores["rougeL"]) if rouge_scores["rougeL"] else 0
            }
            
            # Add DeBERTa metrics if available
            if deberta_model and semantic_scores and humor_scores:
                model_results["metrics"]["avg_semantic_similarity"] = np.mean(semantic_scores)
                model_results["metrics"]["avg_humor_score"] = np.mean(humor_scores)
            
            # Log metrics
            logger.info(f"Model: {model_key} - Avg Generation Time: {model_results['metrics']['avg_generation_time']:.3f}s")
            logger.info(f"Model: {model_key} - Avg ROUGE-1: {model_results['metrics']['avg_rouge1']:.4f}")
            logger.info(f"Model: {model_key} - Avg ROUGE-2: {model_results['metrics']['avg_rouge2']:.4f}")
            logger.info(f"Model: {model_key} - Avg ROUGE-L: {model_results['metrics']['avg_rougeL']:.4f}")
            
            if deberta_model and "avg_semantic_similarity" in model_results["metrics"]:
                logger.info(f"Model: {model_key} - Avg Semantic Similarity: {model_results['metrics']['avg_semantic_similarity']:.4f}")
                logger.info(f"Model: {model_key} - Avg Humor Score: {model_results['metrics']['avg_humor_score']:.4f}")
            
            # Save model results to file
            output_file = os.path.join(output_dir, f"{model_key}_results.json")
            with open(output_file, "w") as f:
                json.dump(model_results, f, indent=2)
            logger.info(f"Results saved to {output_file}")
            
            # Store for comparison
            all_results[model_key] = model_results
            
        except Exception as e:
            logger.error(f"Error testing model {model_key}: {str(e)}")
            
            # Save error to file
            error_result = {
                "model_name": model_info['model_name'],
                "error": str(e),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            output_file = os.path.join(output_dir, f"{model_key}_error.json")
            with open(output_file, "w") as f:
                json.dump(error_result, f, indent=2)
            logger.info(f"Error saved to {output_file}")
            
        finally:
            # Clean up
            if 'model' in locals():
                del model
            if 'inputs' in locals():
                del inputs
            if 'output_ids' in locals():
                del output_ids
            clear_gpu_memory()
    
    # Clean up DeBERTa model
    if deberta_model:
        del deberta_model
        clear_gpu_memory()
    
    # Create comparison summary
    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models_compared": list(models.keys()),
        "metrics": {}
    }
    
    # Add metrics for each model
    for model_key, results in all_results.items():
        if "metrics" in results:
            summary["metrics"][model_key] = results["metrics"]
    
    # Save summary
    summary_file = os.path.join(output_dir, "comparison_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Comparison summary saved to {summary_file}")
    
    return all_results

def main():
    parser = argparse.ArgumentParser(description="Compare models on CAH dataset")
    parser.add_argument("--data_dir", type=str, default="../python-backend/data/processed",
                       help="Directory containing processed CAH data")
    parser.add_argument("--output_dir", type=str, default="model_comparison_results",
                       help="Directory to save comparison results")
    parser.add_argument("--num_samples", type=int, default=20,
                       help="Number of black cards to sample for testing")
    parser.add_argument("--safe", action="store_true", default=True,
                      help="Use safe dataset version")
    parser.add_argument("--models", type=str, nargs="+", 
                      default=["t5-small", "t5-base", "facebook/bart-base", "meta-llama/Llama-3.1-8B", "deepseek-ai/DeepSeek-V2-Chat"],
                      help="List of model names to test")
    parser.add_argument("--use_deberta", action="store_true", default=True,
                      help="Use DeBERTa model for semantic evaluation")
    args = parser.parse_args()
    
    # Define model mapping
    models = {}
    for model_name in args.models:
        if model_name == "t5-small":
            models["t5-small"] = {
                "model_name": "t5-small",
                "model_size": "small",
                "model_type": "encoder-decoder"
            }
        elif model_name == "t5-base":
            models["t5-base"] = {
                "model_name": "t5-base",
                "model_size": "base",
                "model_type": "encoder-decoder"
            }
        elif model_name == "facebook/bart-base":
            models["bart-base"] = {
                "model_name": "facebook/bart-base",
                "model_size": "base",
                "model_type": "encoder-decoder"
            }
        elif model_name == "meta-llama/Llama-3.1-8B":
            models["llama-3.1-8b"] = {
                "model_name": "meta-llama/Llama-3.1-8B",
                "model_size": "8B",
                "model_type": "decoder-only"
            }
        elif model_name == "deepseek-ai/DeepSeek-V2-Chat":
            models["DeepSeek-V2-Chat"] = {
                "model_name": "deepseek-ai/DeepSeek-V2-Chat",
                "model_size": "236B",
                "model_type": "decoder-only"
            }
        elif model_name == "gpt2":
            models["gpt2"] = {
                "model_name": "gpt2",
                "model_size": "base",
                "model_type": "decoder-only"
            }
        else:
            # Determine model type based on name
            if any(x in model_name.lower() for x in ["llama", "gpt", "mistral", "falcon", "bloom", "deepseek"]):
                model_type = "decoder-only"
            else:
                model_type = "encoder-decoder"
                
            models[model_name.replace("/", "-")] = {
                "model_name": model_name,
                "model_size": "custom",
                "model_type": model_type
            }
    
    # Print GPU info
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        logger.info(f"Initial GPU memory usage: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    else:
        logger.warning("No GPU available, using CPU")
    
    # Load test samples
    test_samples = load_cah_data(args.data_dir, args.num_samples, args.safe)
    
    # Run comparison
    results = run_model_comparison(models, test_samples, args.output_dir)
    
    # Print summary
    logger.info("\nComparison Summary:")
    for model_key, model_results in results.items():
        if "metrics" in model_results:
            metrics = model_results["metrics"]
            logger.info(f"Model: {model_key}")
            logger.info(f"  Avg Generation Time: {metrics['avg_generation_time']:.3f}s")
            logger.info(f"  Avg ROUGE-1: {metrics['avg_rouge1']:.4f}")
            logger.info(f"  Avg ROUGE-2: {metrics['avg_rouge2']:.4f}")
            logger.info(f"  Avg ROUGE-L: {metrics['avg_rougeL']:.4f}")
            
            if "avg_semantic_similarity" in metrics:
                logger.info(f"  Avg Semantic Similarity: {metrics['avg_semantic_similarity']:.4f}")
                logger.info(f"  Avg Humor Score: {metrics['avg_humor_score']:.4f}")
    
    logger.info(f"\nDetailed results saved to {args.output_dir}/")

if __name__ == "__main__":
    # Make sure memory is clean at start
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    main() 