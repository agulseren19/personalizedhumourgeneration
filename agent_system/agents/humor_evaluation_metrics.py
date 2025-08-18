#!/usr/bin/env python3
"""
Humor Evaluation Metrics
Implements BLEU, ROUGE, and other baseline comparison metrics as discussed in literature
"""

import re
import math
from typing import List, Dict, Tuple, Optional
from collections import Counter, defaultdict
from dataclasses import dataclass

@dataclass
class OverlapMetrics:
    """Container for overlap-based evaluation metrics"""
    bleu_1: float
    bleu_2: float
    bleu_3: float
    bleu_4: float
    rouge_1_f: float
    rouge_2_f: float
    rouge_l_f: float
    distinct_1: float
    distinct_2: float

class HumorEvaluationMetrics:
    """
    Implements traditional NLP evaluation metrics for humor generation
    Note: Literature shows these are not ideal for humor but useful for baseline comparison
    """
    
    def __init__(self):
        pass
    
    def calculate_all_metrics(self, generated_text: str, reference_text: str) -> OverlapMetrics:
        """Calculate all overlap metrics for comprehensive evaluation"""
        
        # Tokenize texts
        gen_tokens = self._tokenize(generated_text)
        ref_tokens = self._tokenize(reference_text)
        
        # Calculate BLEU scores
        bleu_1 = self.calculate_bleu(gen_tokens, ref_tokens, n=1)
        bleu_2 = self.calculate_bleu(gen_tokens, ref_tokens, n=2)
        bleu_3 = self.calculate_bleu(gen_tokens, ref_tokens, n=3)
        bleu_4 = self.calculate_bleu(gen_tokens, ref_tokens, n=4)
        
        # Calculate ROUGE scores
        rouge_1_f = self.calculate_rouge_n(gen_tokens, ref_tokens, n=1)
        rouge_2_f = self.calculate_rouge_n(gen_tokens, ref_tokens, n=2)
        rouge_l_f = self.calculate_rouge_l(gen_tokens, ref_tokens)
        
        # Calculate diversity metrics
        distinct_1 = self.calculate_distinct_n(gen_tokens, n=1)
        distinct_2 = self.calculate_distinct_n(gen_tokens, n=2)
        
        return OverlapMetrics(
            bleu_1=bleu_1,
            bleu_2=bleu_2,
            bleu_3=bleu_3,
            bleu_4=bleu_4,
            rouge_1_f=rouge_1_f,
            rouge_2_f=rouge_2_f,
            rouge_l_f=rouge_l_f,
            distinct_1=distinct_1,
            distinct_2=distinct_2
        )
    
    def calculate_bleu(self, generated_tokens: List[str], reference_tokens: List[str], n: int = 4) -> float:
        """
        Calculate BLEU-n score
        Note: Literature shows BLEU has poor correlation with humor quality (r = 0.12)
        """
        if len(generated_tokens) == 0:
            return 0.0
        
        # Calculate n-gram precision
        gen_ngrams = self._get_ngrams(generated_tokens, n)
        ref_ngrams = self._get_ngrams(reference_tokens, n)
        
        if not gen_ngrams:
            return 0.0
        
        # Count matches
        matches = 0
        for ngram in gen_ngrams:
            if ngram in ref_ngrams:
                matches += min(gen_ngrams[ngram], ref_ngrams[ngram])
        
        precision = matches / sum(gen_ngrams.values())
        
        # Brevity penalty
        bp = min(1.0, math.exp(1 - len(reference_tokens) / len(generated_tokens)))
        
        return bp * precision
    
    def calculate_rouge_n(self, generated_tokens: List[str], reference_tokens: List[str], n: int = 1) -> float:
        """
        Calculate ROUGE-n F1 score
        ROUGE focuses on recall rather than precision
        """
        gen_ngrams = self._get_ngrams(generated_tokens, n)
        ref_ngrams = self._get_ngrams(reference_tokens, n)
        
        if not ref_ngrams or not gen_ngrams:
            return 0.0
        
        # Calculate overlapping n-grams
        overlap = 0
        for ngram in ref_ngrams:
            if ngram in gen_ngrams:
                overlap += min(ref_ngrams[ngram], gen_ngrams[ngram])
        
        # ROUGE recall
        recall = overlap / sum(ref_ngrams.values()) if sum(ref_ngrams.values()) > 0 else 0.0
        
        # ROUGE precision
        precision = overlap / sum(gen_ngrams.values()) if sum(gen_ngrams.values()) > 0 else 0.0
        
        # F1 score
        if precision + recall == 0:
            return 0.0
        
        return 2 * precision * recall / (precision + recall)
    
    def calculate_rouge_l(self, generated_tokens: List[str], reference_tokens: List[str]) -> float:
        """
        Calculate ROUGE-L score based on Longest Common Subsequence
        """
        if not generated_tokens or not reference_tokens:
            return 0.0
        
        lcs_length = self._longest_common_subsequence(generated_tokens, reference_tokens)
        
        # ROUGE-L recall and precision
        recall = lcs_length / len(reference_tokens)
        precision = lcs_length / len(generated_tokens)
        
        # F1 score
        if precision + recall == 0:
            return 0.0
        
        return 2 * precision * recall / (precision + recall)
    
    def calculate_distinct_n(self, tokens: List[str], n: int = 1) -> float:
        """
        Calculate Distinct-n for measuring diversity/originality
        Used in literature to measure creativity in humor generation
        """
        if len(tokens) < n:
            return 0.0
        
        ngrams = self._get_ngrams(tokens, n)
        unique_ngrams = len(ngrams)
        total_ngrams = sum(ngrams.values())
        
        return unique_ngrams / total_ngrams if total_ngrams > 0 else 0.0
    
    def calculate_semantic_similarity(self, generated_text: str, reference_text: str) -> float:
        """
        Simple semantic similarity based on word overlap
        In practice, would use sentence embeddings (BERT, etc.)
        """
        gen_words = set(self._tokenize(generated_text))
        ref_words = set(self._tokenize(reference_text))
        
        if not gen_words or not ref_words:
            return 0.0
        
        intersection = len(gen_words.intersection(ref_words))
        union = len(gen_words.union(ref_words))
        
        return intersection / union if union > 0 else 0.0
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization (in practice, would use proper tokenizer)"""
        # Convert to lowercase and split on whitespace/punctuation
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        return [token for token in text.split() if token.strip()]
    
    def _get_ngrams(self, tokens: List[str], n: int) -> Counter:
        """Extract n-grams from token list"""
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i + n])
            ngrams.append(ngram)
        return Counter(ngrams)
    
    def _longest_common_subsequence(self, seq1: List[str], seq2: List[str]) -> int:
        """Calculate length of longest common subsequence"""
        m, n = len(seq1), len(seq2)
        
        # Create DP table
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Fill the table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        
        return dp[m][n]
    
    def evaluate_against_multiple_references(
        self, 
        generated_text: str, 
        reference_texts: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate against multiple references (common in humor evaluation)
        Takes maximum BLEU/ROUGE scores across all references
        """
        if not reference_texts:
            return {"error": "No reference texts provided"}
        
        all_metrics = []
        for ref_text in reference_texts:
            metrics = self.calculate_all_metrics(generated_text, ref_text)
            all_metrics.append(metrics)
        
        # Take maximum scores across references
        result = {
            "bleu_1": max(m.bleu_1 for m in all_metrics),
            "bleu_2": max(m.bleu_2 for m in all_metrics),
            "bleu_3": max(m.bleu_3 for m in all_metrics),
            "bleu_4": max(m.bleu_4 for m in all_metrics),
            "rouge_1_f": max(m.rouge_1_f for m in all_metrics),
            "rouge_2_f": max(m.rouge_2_f for m in all_metrics),
            "rouge_l_f": max(m.rouge_l_f for m in all_metrics),
            "distinct_1": max(m.distinct_1 for m in all_metrics),
            "distinct_2": max(m.distinct_2 for m in all_metrics),
            "avg_semantic_sim": sum(self.calculate_semantic_similarity(generated_text, ref) 
                                   for ref in reference_texts) / len(reference_texts)
        }
        
        return result

# Global instance for easy access
humor_metrics = HumorEvaluationMetrics() 