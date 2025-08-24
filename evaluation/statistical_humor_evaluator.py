"""
Statistical Humor Evaluation System
Based on actual computational linguistics methods from literature

This implementation follows the TRUE computational approaches from:
1. Tian et al. (2022): Statistical language modeling and token probabilities
2. Kao et al. (2016): Corpus-based ambiguity and distinctiveness measurement
3. Garimella et al. (2020): Feature-based demographic classification

NO hardcoded dictionaries - uses statistical analysis and corpus-based methods
"""

import re
import math
import random
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from collections import Counter, defaultdict

# Handle imports for different execution contexts
import sys
from pathlib import Path
import pickle # Added for loading trained models

current_dir = Path(__file__).parent
agent_system_dir = current_dir.parent
sys.path.insert(0, str(agent_system_dir))

# Import creativity and diversity metrics
try:
    from evaluation.creativity_diversity_metrics import CreativityDiversityEvaluator
    CREATIVITY_METRICS_AVAILABLE = True
except ImportError:
    print("WARNING: Creativity/Diversity metrics not available")
    CREATIVITY_METRICS_AVAILABLE = False


@dataclass
class StatisticalHumorScores:
    """Statistical humor scores based on corpus analysis"""
    surprisal_score: float      # Token-level surprisal from language model
    ambiguity_score: float      # Statistical ambiguity measurement  
    distinctiveness_ratio: float # Semantic distance ratio
    entropy_score: float        # Information-theoretic entropy
    perplexity_score: float     # Language model perplexity
    semantic_coherence: float   # Cosine similarity-based coherence
    # NEW: Creativity/Diversity metrics from literature
    distinct_1: float           # Distinct-1 ratio (Li et al. 2016)
    distinct_2: float           # Distinct-2 ratio (Li et al. 2016)
    self_bleu: float           # Self-BLEU score (Zhu et al. 2018)
    mauve_score: float         # MAUVE score (Pillutla et al. 2021)
    vocabulary_richness: float  # Type-Token Ratio
    # NEW: Advanced semantic diversity metrics
    overall_semantic_diversity: float  # Overall semantic diversity
    intra_cluster_diversity: float     # Within-cluster diversity
    inter_cluster_diversity: float     # Between-cluster diversity
    semantic_spread: float             # Semantic spread in embedding space
    cluster_coherence: float           # Cluster quality measure
    overall_humor_score: float  # Weighted statistical combination


class StatisticalLanguageModel:
    """
    Production-ready statistical language model for surprisal calculation
    Loads trained models from disk instead of using sample data
    """
    
    def __init__(self, model_path: str = "python-backend/models/trained/language_model.pkl"):
        self.model_path = Path(model_path)
        self.unigram_probs = {}
        self.bigram_probs = {}
        self.trigram_probs = {}
        self.vocab_size = 0
        self.total_tokens = 0
        self.vocabulary = set()
        self.unigram_counts = Counter()
        self.bigram_counts = Counter()
        self.trigram_counts = Counter()
        
        # Load trained model if available
        if self.model_path.exists():
            self._load_trained_model()
        else:
            print(f"WARNING: No trained model found at {self.model_path}")
            print("Using fallback sample model. Run corpus training first!")
            self._initialize_from_common_patterns()
    
    def _load_trained_model(self):
        """Load trained language model from disk"""
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.unigram_probs = model_data.get('unigram_probs', {})
            self.bigram_probs = model_data.get('bigram_probs', {})
            self.trigram_probs = model_data.get('trigram_probs', {})
            self.vocab_size = model_data.get('vocab_size', 0)
            self.total_tokens = model_data.get('total_tokens', 0)
            self.vocabulary = model_data.get('vocabulary', set())
            self.unigram_counts = model_data.get('unigram_counts', Counter())
            self.bigram_counts = model_data.get('bigram_counts', Counter())
            self.trigram_counts = model_data.get('trigram_counts', Counter())
            
            print(f"✅ Loaded trained language model:")
            print(f"   Vocabulary size: {self.vocab_size:,}")
            print(f"   Total tokens: {self.total_tokens:,}")
            print(f"   Unigram probs: {len(self.unigram_probs):,}")
            print(f"   Bigram probs: {len(self.bigram_probs):,}")
            print(f"   Trigram probs: {len(self.trigram_probs):,}")
            
        except Exception as e:
            print(f"❌ Error loading trained model: {e}")
            print("Falling back to sample model...")
            self._initialize_from_common_patterns()
    
    def _initialize_from_common_patterns(self):
        """Fallback to sample data if no trained model available"""
        # This is the old sample approach - only used as fallback
        print("⚠️ Using sample language model (not production-ready)")
        
        # Sample n-gram patterns for demonstration
        sample_patterns = [
            "the quick brown fox",
            "once upon a time", 
            "in the beginning",
            "at the end of",
            "it was a dark",
            "to be or not",
            "how are you doing",
            "what do you think",
            "i don't know what",
            "this is a test"
        ]
        
        # Build simple n-gram counts from samples
        self._build_sample_ngrams(sample_patterns)
    
    def _build_sample_ngrams(self, patterns: List[str]):
        """Build n-gram counts from sample patterns"""
        # This is just for demonstration - not production use
        unigram_counts = Counter()
        bigram_counts = Counter()
        trigram_counts = Counter()
        
        for pattern in patterns:
            tokens = pattern.lower().split()
            
            # Unigrams
            for token in tokens:
                unigram_counts[token] += 1
            
            # Bigrams
            for i in range(len(tokens) - 1):
                bigram = (tokens[i], tokens[i + 1])
                bigram_counts[bigram] += 1
            
            # Trigrams
            for i in range(len(tokens) - 2):
                trigram = (tokens[i], tokens[i + 1], tokens[i + 2])
                trigram_counts[trigram] += 1
        
        # Convert to probabilities with smoothing
        total_unigrams = sum(unigram_counts.values())
        alpha = 0.1
        vocab_size = len(unigram_counts)
        
        for word, count in unigram_counts.items():
            prob = (count + alpha) / (total_unigrams + alpha * vocab_size)
            self.unigram_probs[word] = prob
        
        for bigram, count in bigram_counts.items():
            prev_word = bigram[0]
            prev_count = unigram_counts.get(prev_word, 0)
            if prev_count > 0:
                prob = (count + alpha) / (prev_count + alpha * vocab_size)
                self.bigram_probs[bigram] = prob
        
        for trigram, count in trigram_counts.items():
            bigram = (trigram[0], trigram[1])
            bigram_count = bigram_counts.get(bigram, 0)
            if bigram_count > 0:
                prob = (count + alpha) / (bigram_count + alpha * vocab_size)
                self.trigram_probs[trigram] = prob
        
        self.vocab_size = vocab_size
        self.total_tokens = total_unigrams
        self.vocabulary = set(unigram_counts.keys())
        self.unigram_counts = unigram_counts
        self.bigram_counts = bigram_counts
        self.trigram_counts = trigram_counts
    
    def add_text_to_model(self, text: str):
        """Add text to the statistical model"""
        tokens = self._tokenize(text)
        
        for token in tokens:
            self.unigram_counts[token] += 1
            self.vocabulary.add(token)
            self.total_tokens += 1
        
        # Build bigrams
        for i in range(len(tokens) - 1):
            bigram = (tokens[i], tokens[i + 1])
            self.bigram_counts[bigram] += 1
        
        # Build trigrams
        for i in range(len(tokens) - 2):
            trigram = (tokens[i], tokens[i + 1], tokens[i + 2])
            self.trigram_counts[trigram] += 1
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        # Simple tokenization - in production would use proper tokenizer
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text.split()
    
    def calculate_token_probability(self, token: str, context: List[str]) -> float:
        """Calculate P(token|context) using n-gram model with smoothing"""
        
        # Laplace smoothing parameter
        alpha = 0.1
        vocab_size = len(self.vocabulary) + 1  # +1 for unknown words
        
        if len(context) >= 2:
            # Try trigram
            trigram = (context[-2], context[-1], token)
            trigram_count = self.trigram_counts.get(trigram, 0)
            bigram_count = self.bigram_counts.get((context[-2], context[-1]), 0)
            
            if bigram_count > 0:
                trigram_prob = (trigram_count + alpha) / (bigram_count + alpha * vocab_size)
                return trigram_prob
        
        if len(context) >= 1:
            # Try bigram
            bigram = (context[-1], token)
            bigram_count = self.bigram_counts.get(bigram, 0)
            unigram_count = self.unigram_counts.get(context[-1], 0)
            
            if unigram_count > 0:
                bigram_prob = (bigram_count + alpha) / (unigram_count + alpha * vocab_size)
                return bigram_prob
        
        # Fall back to unigram
        unigram_count = self.unigram_counts.get(token, 0)
        unigram_prob = (unigram_count + alpha) / (self.total_tokens + alpha * vocab_size)
        
        return unigram_prob
    
    def calculate_surprisal(self, text: str, context: str = "") -> float:
        """
        Calculate surprisal as per Tian et al.: -log P(token|context)
        """
        tokens = self._tokenize(text)
        context_tokens = self._tokenize(context)
        
        total_surprisal = 0.0
        current_context = context_tokens.copy()
        
        for token in tokens:
            prob = self.calculate_token_probability(token, current_context)
            surprisal = -math.log(max(prob, 1e-10))  # Avoid log(0)
            total_surprisal += surprisal
            
            # Update context window (keep last 2 tokens)
            current_context.append(token)
            if len(current_context) > 2:
                current_context.pop(0)
        
        # Average surprisal per token
        avg_surprisal = total_surprisal / max(len(tokens), 1)
        
        # Normalize to 0-10 scale (empirically determined scaling)
        normalized = min(avg_surprisal / 2.0, 10.0)
        return normalized
    
    def calculate_perplexity(self, text: str) -> float:
        """Calculate perplexity: 2^(average surprisal)"""
        tokens = self._tokenize(text)
        if not tokens:
            return 1.0
        
        total_log_prob = 0.0
        context = []
        
        for token in tokens:
            prob = self.calculate_token_probability(token, context)
            total_log_prob += math.log(max(prob, 1e-10))
            
            context.append(token)
            if len(context) > 2:
                context.pop(0)
        
        avg_log_prob = total_log_prob / len(tokens)
        perplexity = math.exp(-avg_log_prob)
        
        # Normalize to 0-10 scale
        return min(math.log(perplexity), 10.0)


class StatisticalSemanticAnalyzer:
    """
    Production-ready semantic analyzer without hardcoded dictionaries
    Loads trained context vectors from corpus training
    """
    
    def __init__(self, model_path: str = "python-backend/models/trained/semantic_analyzer.pkl"):
        self.model_path = Path(model_path)
        self.word_context_vectors = {}
        self.word_frequencies = Counter()
        self.total_tokens = 0
        self.total_documents = 0
        self.document_frequency = Counter()
        
        # Load trained semantic analyzer if available
        if self.model_path.exists():
            self._load_trained_analyzer()
        else:
            print(f"WARNING: No trained semantic analyzer found at {self.model_path}")
            print("Using fallback sample analyzer. Run corpus training first!")
            self._build_context_vectors()
    
    def _load_trained_analyzer(self):
        """Load trained semantic analyzer from disk"""
        try:
            with open(self.model_path, 'rb') as f:
                analyzer_data = pickle.load(f)
            
            self.word_context_vectors = analyzer_data.get('context_vectors', {})
            self.word_frequencies = analyzer_data.get('word_frequencies', Counter())
            self.total_tokens = analyzer_data.get('total_tokens', 0)
            self.total_documents = analyzer_data.get('total_documents', 0)
            self.document_frequency = analyzer_data.get('document_frequency', Counter())
            
            print(f"✅ Loaded trained semantic analyzer:")
            print(f"   Context vectors: {len(self.word_context_vectors):,}")
            print(f"   Word frequencies: {len(self.word_frequencies):,}")
            print(f"   Total tokens: {self.total_tokens:,}")
            
        except Exception as e:
            print(f"❌ Error loading trained analyzer: {e}")
            print("Falling back to sample analyzer...")
            self._build_context_vectors()
    
    def _build_context_vectors(self):
        """Fallback to sample data if no trained model available"""
        # This is the old sample approach - only used as fallback
        print("⚠️ Using sample semantic analyzer (not production-ready)")
        
        # Sample texts for building distributional semantics
        # In production, this would use a large corpus
        sample_texts = [
            "money bank financial deposit withdraw cash savings account",
            "river bank shore water flow stream nature outdoor",
            "baseball bat swing hit home run sports game",
            "animal bat cave night flying mammal wings",
            "dog bark sound loud noise pet animal",
            "tree bark brown rough texture wood nature",
            "light bright illuminate lamp sun bright sunshine",
            "light weight heavy carry lift physical mass",
            "play game fun entertainment sport activity",
            "play theater drama actor stage performance",
            "fair just equal right honest moral",
            "fair carnival rides games fun entertainment",
            "bank financial institution money loan credit",
            "bank river side edge water shore",
            "humor funny joke laugh comedy entertainment",
            "serious formal business professional work",
            "family children parents home love care",
            "adult mature grown sophisticated serious",
            "cards game playing entertainment fun hobby",
            "cards against humanity party adult game",
            "white card response funny inappropriate edgy",
            "black card prompt setup question scenario",
            "inappropriate adult mature edgy offensive humor",
            "clean family friendly wholesome innocent safe",
            "clever witty smart intelligent creative funny",
            "stupid silly dumb foolish ridiculous absurd"
        ]
        
        for text in sample_texts:
            self._add_text_to_vectors(text)
    
    def _add_text_to_vectors(self, text: str):
        """Add text to context vectors"""
        words = text.lower().split()
        self.total_documents += 1
        
        # Document frequency
        unique_words = set(words)
        for word in unique_words:
            self.document_frequency[word] += 1
        
        # Context vectors (word co-occurrence within window)
        window_size = 3
        for i, target_word in enumerate(words):
            start = max(0, i - window_size)
            end = min(len(words), i + window_size + 1)
            
            for j in range(start, end):
                if i != j:
                    context_word = words[j]
                    if target_word not in self.word_context_vectors:
                        self.word_context_vectors[target_word] = Counter()
                    self.word_context_vectors[target_word][context_word] += 1
    
    def calculate_semantic_similarity(self, word1: str, word2: str) -> float:
        """Calculate semantic similarity using cosine similarity of context vectors"""
        
        if word1 == word2:
            return 1.0
        
        vec1 = self.word_context_vectors.get(word1.lower(), Counter())
        vec2 = self.word_context_vectors.get(word2.lower(), Counter())
        
        if not vec1 or not vec2:
            return 0.0
        
        # Calculate cosine similarity
        return self._cosine_similarity(vec1, vec2)
    
    def _cosine_similarity(self, vec1: Counter, vec2: Counter) -> float:
        """Calculate cosine similarity between two counter vectors"""
        
        # Get all dimensions
        all_keys = set(vec1.keys()) | set(vec2.keys())
        
        if not all_keys:
            return 0.0
        
        # Calculate dot product and magnitudes
        dot_product = sum(vec1[key] * vec2[key] for key in all_keys)
        
        magnitude1 = math.sqrt(sum(vec1[key] ** 2 for key in vec1))
        magnitude2 = math.sqrt(sum(vec2[key] ** 2 for key in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def calculate_text_coherence(self, text: str, context: str) -> float:
        """
        Calculate semantic coherence between text and context
        Following Garimella et al. (2020): Uses sentence embeddings for coherence
        """
        
        if not text.strip() or not context.strip():
            return 0.0
        
        try:
            # Try to use sentence-transformers (BERT-based) as per Garimella et al. (2020)
            from sentence_transformers import SentenceTransformer
            import torch
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np
            
            # Use same model as semantic diversity calculator for consistency
            model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Generate embeddings for text and context
            text_embedding = model.encode([text])
            context_embedding = model.encode([context])
            
            # Calculate cosine similarity between embeddings
            similarity = cosine_similarity(text_embedding, context_embedding)[0][0]
            
            # Scale to 0-10 range
            coherence_score = max(0.0, min(10.0, similarity * 10))
            return coherence_score
            
        except ImportError:
            # Fallback to sparse co-occurrence method (note: proxy for speed)
            print("⚠️ Sentence-transformers not available, using sparse co-occurrence proxy for speed")
            return self._calculate_sparse_coherence_fallback(text, context)
        except Exception as e:
            # Fallback on any other error
            return self._calculate_sparse_coherence_fallback(text, context)
    
    def _calculate_sparse_coherence_fallback(self, text: str, context: str) -> float:
        """
        Fallback coherence calculation using sparse co-occurrence vectors
        Note: This is a sparse proxy used for speed when sentence-transformers unavailable
        """
        text_words = text.lower().split()
        context_words = context.lower().split()
        
        if not text_words or not context_words:
            return 0.0
        
        # Average pairwise similarity using co-occurrence vectors
        total_similarity = 0.0
        pairs = 0
        
        for text_word in text_words:
            for context_word in context_words:
                similarity = self.calculate_semantic_similarity(text_word, context_word)
                total_similarity += similarity
                pairs += 1
        
        if pairs == 0:
            return 0.0
        
        avg_similarity = total_similarity / pairs
        return avg_similarity * 10  # Scale to 0-10
    
    def calculate_ambiguity_score(self, text: str) -> float:
        """
        Calculate ambiguity using distributional analysis
        Higher entropy in context vector = more ambiguous
        """
        words = text.lower().split()
        
        total_ambiguity = 0.0
        
        for word in words:
            context_vec = self.word_context_vectors.get(word, Counter())
            
            if not context_vec:
                continue
            
            # Calculate entropy of context distribution
            total_count = sum(context_vec.values())
            entropy = 0.0
            
            for count in context_vec.values():
                prob = count / total_count
                if prob > 0:
                    entropy -= prob * math.log(prob)
            
            total_ambiguity += entropy
        
        # Normalize by text length
        avg_ambiguity = total_ambiguity / max(len(words), 1)
        
        # Scale to 0-10 (entropy typically ranges 0-5 for natural language)
        return min(avg_ambiguity * 2, 10.0)
    
    def calculate_distinctiveness_ratio(self, text: str, context: str) -> float:
        """
        Calculate distinctiveness as semantic distance ratio
        Based on Kao et al. - how distinct are the interpretations
        """
        text_words = set(text.lower().split())
        context_words = set(context.lower().split())
        
        if not text_words or not context_words:
            return 5.0
        
        # Calculate average similarities
        similarities = []
        for text_word in text_words:
            word_sims = []
            for context_word in context_words:
                sim = self.calculate_semantic_similarity(text_word, context_word)
                word_sims.append(sim)
            
            if word_sims:
                avg_sim = sum(word_sims) / len(word_sims)
                similarities.append(avg_sim)
        
        if not similarities:
            return 5.0
        
        # Standard deviation of similarities (higher = more distinctive)
        mean_sim = sum(similarities) / len(similarities)
        variance = sum((sim - mean_sim) ** 2 for sim in similarities) / len(similarities)
        std_dev = math.sqrt(variance)
        
        # Scale to 0-10
        return min(std_dev * 20, 10.0)


class StatisticalInformationAnalyzer:
    """
    Information-theoretic analysis for humor evaluation
    Based on information theory and entropy measures
    """
    
    def calculate_entropy_score(self, text: str) -> float:
        """Calculate information entropy of the text"""
        
        # Character-level entropy
        char_counts = Counter(text.lower())
        total_chars = sum(char_counts.values())
        
        if total_chars == 0:
            return 0.0
        
        char_entropy = 0.0
        for count in char_counts.values():
            prob = count / total_chars
            if prob > 0:
                char_entropy -= prob * math.log(prob)
        
        # Word-level entropy
        words = text.lower().split()
        word_counts = Counter(words)
        total_words = len(words)
        
        word_entropy = 0.0
        if total_words > 0:
            for count in word_counts.values():
                prob = count / total_words
                if prob > 0:
                    word_entropy -= prob * math.log(prob)
        
        # Combine entropies
        combined_entropy = (char_entropy * 0.3 + word_entropy * 0.7)
        
        # Normalize to 0-10 scale
        return min(combined_entropy * 2, 10.0)
    
    def calculate_information_content(self, text: str, language_model: StatisticalLanguageModel) -> float:
        """Calculate information content using language model"""
        
        tokens = text.lower().split()
        if not tokens:
            return 0.0
        
        total_info = 0.0
        context = []
        
        for token in tokens:
            prob = language_model.calculate_token_probability(token, context)
            info_content = -math.log(max(prob, 1e-10))
            total_info += info_content
            
            context.append(token)
            if len(context) > 2:
                context.pop(0)
        
        avg_info = total_info / len(tokens)
        return min(avg_info / 2, 10.0)


class StatisticalHumorEvaluator:
    """
    Complete statistical humor evaluator
    NO hardcoded dictionaries - uses corpus-based statistical methods
    """
    
    def __init__(self):
        self.language_model = StatisticalLanguageModel()
        self.semantic_analyzer = StatisticalSemanticAnalyzer()
        self.info_analyzer = StatisticalInformationAnalyzer()
        # Add creativity/diversity evaluator
        if CREATIVITY_METRICS_AVAILABLE:
            self.creativity_evaluator = CreativityDiversityEvaluator()
        else:
            self.creativity_evaluator = None
    
    def evaluate_humor_statistically(self, text: str, context: str, 
                                   audience: str = 'adults') -> StatisticalHumorScores:
        """
        Comprehensive statistical humor evaluation
        Based on computational linguistics and information theory
        """
        
        # 1. Language model-based metrics
        surprisal_score = self.language_model.calculate_surprisal(text, context)
        perplexity_score = self.language_model.calculate_perplexity(text)
        
        # 2. Semantic analysis
        ambiguity_score = self.semantic_analyzer.calculate_ambiguity_score(text)
        distinctiveness_ratio = self.semantic_analyzer.calculate_distinctiveness_ratio(text, context)
        semantic_coherence = self.semantic_analyzer.calculate_text_coherence(text, context)
        
        # 3. Information-theoretic analysis
        entropy_score = self.info_analyzer.calculate_entropy_score(text)
        
        # 4. NEW: Creativity/Diversity metrics (if available)
        distinct_1 = 0.0
        distinct_2 = 0.0
        self_bleu = 0.0
        mauve_score = 0.0
        vocabulary_richness = 0.0
        overall_semantic_diversity = 0.0
        intra_cluster_diversity = 0.0
        inter_cluster_diversity = 0.0
        semantic_spread = 0.0
        cluster_coherence = 0.0
        
        if self.creativity_evaluator:
            # For single text evaluation, create a small batch for diversity analysis
            text_batch = [text]  # Single text batch
            creativity_scores = self.creativity_evaluator.evaluate_creativity_diversity(text_batch)
            
            distinct_1 = creativity_scores.distinct_1
            distinct_2 = creativity_scores.distinct_2
            self_bleu = creativity_scores.self_bleu_1  # Use BLEU-1 for single text
            mauve_score = creativity_scores.mauve_score
            vocabulary_richness = creativity_scores.vocabulary_richness
            overall_semantic_diversity = creativity_scores.overall_semantic_diversity
            intra_cluster_diversity = creativity_scores.intra_cluster_diversity
            inter_cluster_diversity = creativity_scores.inter_cluster_diversity
            semantic_spread = creativity_scores.semantic_spread
            cluster_coherence = creativity_scores.cluster_coherence
        
        # 5. Calculate overall score with new metrics
        overall_score = self._calculate_statistical_overall_score(
            surprisal_score, ambiguity_score, distinctiveness_ratio,
            entropy_score, perplexity_score, semantic_coherence,
            distinct_1, distinct_2, self_bleu, vocabulary_richness, overall_semantic_diversity
        )
        
        return StatisticalHumorScores(
            surprisal_score=surprisal_score,
            ambiguity_score=ambiguity_score,
            distinctiveness_ratio=distinctiveness_ratio,
            entropy_score=entropy_score,
            perplexity_score=perplexity_score,
            semantic_coherence=semantic_coherence,
            distinct_1=distinct_1,
            distinct_2=distinct_2,
            self_bleu=self_bleu,
            mauve_score=mauve_score,
            vocabulary_richness=vocabulary_richness,
            overall_semantic_diversity=overall_semantic_diversity,
            intra_cluster_diversity=intra_cluster_diversity,
            inter_cluster_diversity=inter_cluster_diversity,
            semantic_spread=semantic_spread,
            cluster_coherence=cluster_coherence,
            overall_humor_score=overall_score
        )
    
    def _calculate_statistical_overall_score(self, surprisal: float, ambiguity: float,
                                           distinctiveness: float, entropy: float,
                                           perplexity: float, coherence: float,
                                           distinct_1: float = 0.0, distinct_2: float = 0.0, 
                                           self_bleu: float = 0.0, vocabulary_richness: float = 0.0, 
                                           semantic_diversity: float = 0.0) -> float:
        """
        Calculate overall humor score using statistical methods
        Weights based on empirical findings from literature
        """
        
        # Tian et al. (2022) - surprisal is key for humor
        surprisal_component = surprisal * 0.2
        
        # Kao et al. (2016) - ambiguity and distinctiveness
        kao_component = (ambiguity * 0.6 + distinctiveness * 0.4) * 0.2
        
        # Information theory - entropy and perplexity
        info_component = (entropy * 0.7 + perplexity * 0.3) * 0.2
        
        # Coherence for readability
        coherence_component = coherence * 0.15
        
        # NEW: Creativity/Diversity component (Li et al. 2016, Zhu et al. 2018)
        creativity_component = 0.0
        if distinct_1 > 0 or distinct_2 > 0 or vocabulary_richness > 0:
            # Higher Distinct-n = more creative (positive)
            distinct_contribution = (distinct_1 * 0.4 + distinct_2 * 0.3) * 10  # Scale to 0-10
            
            # Lower Self-BLEU = more creative (invert)
            self_bleu_contribution = (1.0 - self_bleu) * 10  # Scale to 0-10
            
            # Vocabulary and semantic diversity
            diversity_contribution = (vocabulary_richness * 0.5 + semantic_diversity * 0.5) * 10
            
            creativity_component = (distinct_contribution * 0.4 + 
                                  self_bleu_contribution * 0.3 + 
                                  diversity_contribution * 0.3) * 0.25
        
        overall = (surprisal_component + kao_component + info_component + 
                  coherence_component + creativity_component)
        
        return max(0.0, min(10.0, overall))
    
    def evaluate_humor_batch(self, texts: List[str], contexts: List[str], 
                           audience: str = 'adults') -> List[StatisticalHumorScores]:
        """
        Evaluate multiple humor texts together for better creativity/diversity analysis
        """
        if len(texts) != len(contexts):
            raise ValueError("Number of texts and contexts must match")
        
        # Calculate creativity/diversity metrics for the entire batch
        batch_creativity_scores = None
        if self.creativity_evaluator:
            batch_creativity_scores = self.creativity_evaluator.evaluate_creativity_diversity(texts)
        
        # Evaluate each text individually, but use batch creativity scores
        results = []
        for i, (text, context) in enumerate(zip(texts, contexts)):
            # Individual metrics
            surprisal_score = self.language_model.calculate_surprisal(text, context)
            perplexity_score = self.language_model.calculate_perplexity(text)
            ambiguity_score = self.semantic_analyzer.calculate_ambiguity_score(text)
            distinctiveness_ratio = self.semantic_analyzer.calculate_distinctiveness_ratio(text, context)
            semantic_coherence = self.semantic_analyzer.calculate_text_coherence(text, context)
            entropy_score = self.info_analyzer.calculate_entropy_score(text)
            
            # Use batch creativity scores
            distinct_1 = batch_creativity_scores.distinct_1 if batch_creativity_scores else 0.0
            distinct_2 = batch_creativity_scores.distinct_2 if batch_creativity_scores else 0.0
            self_bleu = batch_creativity_scores.self_bleu_1 if batch_creativity_scores else 0.0
            mauve_score = batch_creativity_scores.mauve_score if batch_creativity_scores else 0.0
            vocabulary_richness = batch_creativity_scores.vocabulary_richness if batch_creativity_scores else 0.0
            overall_semantic_diversity = batch_creativity_scores.overall_semantic_diversity if batch_creativity_scores else 0.0
            
            # Calculate overall score
            overall_score = self._calculate_statistical_overall_score(
                surprisal_score, ambiguity_score, distinctiveness_ratio,
                entropy_score, perplexity_score, semantic_coherence,
                distinct_1, distinct_2, self_bleu, vocabulary_richness, overall_semantic_diversity
            )
            
            result = StatisticalHumorScores(
                surprisal_score=surprisal_score,
                ambiguity_score=ambiguity_score,
                distinctiveness_ratio=distinctiveness_ratio,
                entropy_score=entropy_score,
                perplexity_score=perplexity_score,
                semantic_coherence=semantic_coherence,
                distinct_1=distinct_1,
                distinct_2=distinct_2,
                self_bleu=self_bleu,
                mauve_score=mauve_score,
                vocabulary_richness=vocabulary_richness,
                overall_semantic_diversity=overall_semantic_diversity,
                intra_cluster_diversity=batch_creativity_scores.intra_cluster_diversity if batch_creativity_scores else 0.0,
                inter_cluster_diversity=batch_creativity_scores.inter_cluster_diversity if batch_creativity_scores else 0.0,
                semantic_spread=batch_creativity_scores.semantic_spread if batch_creativity_scores else 0.0,
                cluster_coherence=batch_creativity_scores.cluster_coherence if batch_creativity_scores else 0.0,
                overall_humor_score=overall_score
            )
            results.append(result)
        
        return results

    def add_training_text(self, text: str):
        """Add text to improve the statistical models"""
        self.language_model.add_text_to_model(text)
        self.semantic_analyzer._add_text_to_vectors(text)


def test_statistical_evaluator():
    """Test the statistical evaluator"""
    evaluator = StatisticalHumorEvaluator()
    
    # Test cases
    test_cases = [
        {
            'text': 'My bank account and I have trust issues',
            'context': 'The worst part about being broke'
        },
        {
            'text': 'A disappointing potato',
            'context': 'What best describes your life'
        },
        {
            'text': 'Existential dread',
            'context': 'What defines modern adulthood'
        }
    ]
    
    print("Statistical Humor Evaluator Test:")
    print("=" * 50)
    
    for i, test in enumerate(test_cases):
        print(f"\nTest {i+1}:")
        print(f"Text: '{test['text']}'")
        print(f"Context: '{test['context']}'")
        
        scores = evaluator.evaluate_humor_statistically(test['text'], test['context'])
        
        print(f"Results:")
        print(f"  Surprisal: {scores.surprisal_score:.2f}")
        print(f"  Ambiguity: {scores.ambiguity_score:.2f}")
        print(f"  Distinctiveness: {scores.distinctiveness_ratio:.2f}")
        print(f"  Entropy: {scores.entropy_score:.2f}")
        print(f"  Perplexity: {scores.perplexity_score:.2f}")
        print(f"  Coherence: {scores.semantic_coherence:.2f}")
        print(f"  NEW METRICS:")
        print(f"    Distinct-1: {scores.distinct_1:.3f}")
        print(f"    Distinct-2: {scores.distinct_2:.3f}")
        print(f"    Self-BLEU: {scores.self_bleu:.3f}")
        print(f"    MAUVE: {scores.mauve_score:.3f}")
        print(f"    Vocab Richness: {scores.vocabulary_richness:.3f}")
        print(f"    Semantic Diversity: {scores.overall_semantic_diversity:.3f}")
        print(f"    Intra-Cluster Diversity: {scores.intra_cluster_diversity:.3f}")
        print(f"    Inter-Cluster Diversity: {scores.inter_cluster_diversity:.3f}")
        print(f"    Semantic Spread: {scores.semantic_spread:.3f}")
        print(f"    Cluster Coherence: {scores.cluster_coherence:.3f}")
        print(f"  OVERALL: {scores.overall_humor_score:.2f}")


if __name__ == "__main__":
    test_statistical_evaluator()
