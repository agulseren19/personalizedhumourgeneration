"""
Creativity and Diversity Metrics for Humor Evaluation
Based on literature review - implements Distinct-n, Self-BLEU, and MAUVE metrics

Literature Sources:
1. Li et al. (2016): Distinct-n for diversity measurement
2. Zhu et al. (2018): Self-BLEU for repetition penalty
3. Pillutla et al. (2021): MAUVE for generation quality
"""

import re
import math
import numpy as np
from typing import List, Dict, Set, Tuple, Any
from collections import Counter
from dataclasses import dataclass

# NLTK for BLEU calculation
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.tokenize import word_tokenize
    import nltk
    # Download required data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
except ImportError:
    print("WARNING: NLTK not available. Install with: pip install nltk")
    # Fallback tokenization
    def word_tokenize(text):
        return re.findall(r'\b\w+\b', text.lower())
    
    def sentence_bleu(references, hypothesis, weights=(0.25, 0.25, 0.25, 0.25)):
        return 0.0  # Fallback implementation

# Optional: MAUVE (requires additional dependencies)
try:
    import mauve
    MAUVE_AVAILABLE = True
except ImportError:
    MAUVE_AVAILABLE = False
    print("WARNING: MAUVE not available. Install with: pip install mauve-text")


@dataclass
class CreativityDiversityScores:
    """Creativity and diversity scores with advanced semantic metrics"""
    distinct_1: float          # Distinct-1 ratio
    distinct_2: float          # Distinct-2 ratio  
    distinct_3: float          # Distinct-3 ratio
    self_bleu_1: float        # Self-BLEU-1 score
    self_bleu_2: float        # Self-BLEU-2 score
    self_bleu_3: float        # Self-BLEU-3 score
    self_bleu_4: float        # Self-BLEU-4 score
    mauve_score: float        # MAUVE score (if available)
    vocabulary_richness: float # Type-Token Ratio
    # NEW: Advanced semantic diversity metrics
    overall_semantic_diversity: float  # Overall semantic diversity
    intra_cluster_diversity: float     # Within-cluster diversity
    inter_cluster_diversity: float     # Between-cluster diversity
    semantic_spread: float             # Semantic spread in embedding space
    cluster_coherence: float           # Cluster quality measure
    overall_creativity: float  # Combined creativity score


class DistinctNCalculator:
    """
    Implements Distinct-n metrics as described in Li et al. (2016)
    
    Distinct-n = number of distinct n-grams / total number of n-grams
    Higher values indicate more diverse/creative output
    """
    
    def __init__(self):
        pass
    
    def calculate_distinct_n(self, texts: List[str], n: int) -> float:
        """
        Calculate Distinct-n for a list of generated texts
        
        Args:
            texts: List of generated text strings
            n: N-gram size (1, 2, 3, etc.)
            
        Returns:
            Distinct-n ratio (0-1, higher is more diverse)
        """
        if not texts:
            return 0.0
        
        all_ngrams = []
        
        for text in texts:
            tokens = self._tokenize(text)
            ngrams = self._get_ngrams(tokens, n)
            all_ngrams.extend(ngrams)
        
        if not all_ngrams:
            return 0.0
        
        unique_ngrams = len(set(all_ngrams))
        total_ngrams = len(all_ngrams)
        
        return unique_ngrams / total_ngrams
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        return word_tokenize(text.lower())
    
    def _get_ngrams(self, tokens: List[str], n: int) -> List[Tuple[str, ...]]:
        """Extract n-grams from token list"""
        if len(tokens) < n:
            return []
        
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            ngrams.append(ngram)
        
        return ngrams


class SelfBLEUCalculator:
    """
    Implements Self-BLEU as described in Zhu et al. (2018)
    
    Self-BLEU measures repetition by computing BLEU score between
    each generated text and all other generated texts.
    Lower Self-BLEU indicates more diversity.
    """
    
    def __init__(self):
        self.smoothing = SmoothingFunction().method1 if 'SmoothingFunction' in globals() else None
    
    def calculate_self_bleu(self, texts: List[str], n: int = 4) -> float:
        """
        Calculate Self-BLEU-n for a list of generated texts
        
        Args:
            texts: List of generated text strings
            n: Maximum n-gram order for BLEU (1-4)
            
        Returns:
            Average Self-BLEU score (0-1, lower is more diverse)
        """
        if len(texts) < 2:
            return 0.0
        
        # Set weights for BLEU-n
        if n == 1:
            weights = (1.0,)
        elif n == 2:
            weights = (0.5, 0.5)
        elif n == 3:
            weights = (1/3, 1/3, 1/3)
        else:  # n == 4
            weights = (0.25, 0.25, 0.25, 0.25)
        
        total_bleu = 0.0
        count = 0
        
        for i, text in enumerate(texts):
            hypothesis = word_tokenize(text.lower())
            
            # Use all other texts as references
            references = []
            for j, other_text in enumerate(texts):
                if i != j:
                    reference = word_tokenize(other_text.lower())
                    references.append(reference)
            
            if references and hypothesis:
                try:
                    if self.smoothing:
                        bleu_score = sentence_bleu(references, hypothesis, 
                                                 weights=weights, 
                                                 smoothing_function=self.smoothing)
                    else:
                        bleu_score = sentence_bleu(references, hypothesis, weights=weights)
                    
                    total_bleu += bleu_score
                    count += 1
                except:
                    # Handle edge cases
                    continue
        
        return total_bleu / max(count, 1)


class MAUVECalculator:
    """
    Implements MAUVE as described in Pillutla et al. (2021)
    
    MAUVE measures the gap between neural text and human text
    using divergence frontiers. Higher MAUVE indicates better quality.
    """
    
    def __init__(self):
        self.available = MAUVE_AVAILABLE
    
    def calculate_mauve_score(self, generated_texts: List[str], 
                            reference_texts: List[str]) -> float:
        """
        Calculate MAUVE score between generated and reference texts
        
        Args:
            generated_texts: List of generated text strings
            reference_texts: List of reference/human text strings
            
        Returns:
            MAUVE score (0-1, higher is better)
        """
        if not self.available:
            print("WARNING: MAUVE not available, returning fallback score")
            return self._calculate_fallback_mauve(generated_texts, reference_texts)
        
        try:
            # Calculate MAUVE score
            out = mauve.compute_mauve(
                p_text=generated_texts,
                q_text=reference_texts,
                device_id=0,  # Use CPU
                max_text_length=512,
                verbose=False
            )
            return out.mauve
        except Exception as e:
            print(f"MAUVE calculation failed: {e}")
            return self._calculate_fallback_mauve(generated_texts, reference_texts)
    
    def _calculate_fallback_mauve(self, generated_texts: List[str], 
                                reference_texts: List[str]) -> float:
        """
        Fallback MAUVE implementation using simple text statistics
        """
        if not generated_texts or not reference_texts:
            return 0.0
        
        # Simple statistical comparison
        gen_stats = self._text_statistics(generated_texts)
        ref_stats = self._text_statistics(reference_texts)
        
        # Calculate similarity between statistics
        similarity = 0.0
        total_metrics = 0
        
        for key in gen_stats:
            if key in ref_stats:
                # Normalize difference
                diff = abs(gen_stats[key] - ref_stats[key])
                max_val = max(gen_stats[key], ref_stats[key], 1.0)
                norm_similarity = 1.0 - (diff / max_val)
                similarity += norm_similarity
                total_metrics += 1
        
        return similarity / max(total_metrics, 1)
    
    def _text_statistics(self, texts: List[str]) -> Dict[str, float]:
        """Calculate basic text statistics"""
        if not texts:
            return {}
        
        all_text = " ".join(texts)
        tokens = word_tokenize(all_text)
        
        return {
            "avg_length": np.mean([len(word_tokenize(t)) for t in texts]),
            "vocab_size": len(set(tokens)),
            "type_token_ratio": len(set(tokens)) / max(len(tokens), 1),
            "avg_word_length": np.mean([len(w) for w in tokens]) if tokens else 0
        }


class VocabularyRichnessCalculator:
    """
    Calculate vocabulary richness using Type-Token Ratio and variants
    """
    
    def __init__(self):
        pass
    
    def calculate_vocabulary_richness(self, texts: List[str]) -> float:
        """
        Calculate vocabulary richness using multiple measures
        
        Returns combined richness score (0-1, higher is richer)
        """
        if not texts:
            return 0.0
        
        all_text = " ".join(texts)
        tokens = word_tokenize(all_text)
        
        if not tokens:
            return 0.0
        
        # Type-Token Ratio (TTR)
        ttr = len(set(tokens)) / len(tokens)
        
        # Root TTR (for length normalization)
        rttr = len(set(tokens)) / math.sqrt(len(tokens))
        
        # Corrected TTR
        cttr = len(set(tokens)) / math.sqrt(2 * len(tokens))
        
        # Combine measures (normalize RTTR and CTTR)
        combined = (ttr + min(rttr / 10, 1.0) + min(cttr / 10, 1.0)) / 3
        
        return min(combined, 1.0)


class SemanticDiversityCalculator:
    """
    Production-ready semantic diversity calculator using real embeddings
    
    Features:
    - Real sentence embeddings (SentenceTransformer)
    - Cosine similarity analysis
    - Semantic clustering
    - Diversity metrics (intra-cluster vs inter-cluster)
    - GPU acceleration support
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', use_gpu: bool = False):
        """
        Initialize production semantic diversity calculator
        
        Args:
            model_name: SentenceTransformer model to use
            use_gpu: Whether to use GPU acceleration
        """
        try:
            from sentence_transformers import SentenceTransformer
            import torch
            
            # Check GPU availability
            if use_gpu and torch.cuda.is_available():
                self.device = 'cuda'
                print(f"✅ Using GPU acceleration for semantic diversity calculation")
            else:
                self.device = 'cpu'
                if use_gpu:
                    print(f"⚠️ GPU requested but not available, using CPU")
                else:
                    print(f"✅ Using CPU for semantic diversity calculation")
            
            # Load sentence transformer model
            self.model = SentenceTransformer(model_name, device=self.device)
            self.model_name = model_name
            self.available = True
            
            print(f"✅ Loaded semantic model: {model_name}")
            
        except ImportError:
            print("❌ SentenceTransformers not available. Install with: pip install sentence-transformers")
            self.available = False
            self.model = None
        except Exception as e:
            print(f"❌ Error loading semantic model: {e}")
            self.available = False
            self.model = None
    
    def calculate_semantic_diversity(self, texts: List[str]) -> float:
        """
        Calculate semantic diversity using real embeddings
        
        Args:
            texts: List of text strings to analyze
            
        Returns:
            Semantic diversity score (0-1, higher is more diverse)
        """
        if not self.available or not self.model:
            print("⚠️ Semantic model not available, using fallback word overlap")
            return self._calculate_fallback_diversity(texts)
        
        if len(texts) < 2:
            return 0.0
        
        try:
            # Generate embeddings
            embeddings = self.model.encode(texts, convert_to_tensor=True, device=self.device)
            
            # Calculate pairwise cosine similarities
            similarities = self._calculate_cosine_similarities(embeddings)
            
            # Calculate diversity metrics
            diversity_score = self._calculate_diversity_from_similarities(similarities)
            
            return diversity_score
            
        except Exception as e:
            print(f"⚠️ Semantic diversity calculation failed: {e}")
            print("Falling back to word overlap method")
            return self._calculate_fallback_diversity(texts)
    
    def _calculate_cosine_similarities(self, embeddings) -> np.ndarray:
        """Calculate pairwise cosine similarities between embeddings"""
        try:
            import torch
            from torch.nn.functional import cosine_similarity
            
            # Normalize embeddings
            normalized_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            # Calculate cosine similarities
            similarities = cosine_similarity(normalized_embeddings.unsqueeze(1), 
                                          normalized_embeddings.unsqueeze(0), 
                                          dim=2)
            
            # Convert to numpy and remove diagonal (self-similarity)
            similarities_np = similarities.cpu().numpy()
            np.fill_diagonal(similarities_np, 0)  # Remove self-similarity
            
            return similarities_np
            
        except Exception as e:
            print(f"⚠️ Cosine similarity calculation failed: {e}")
            # Fallback to numpy implementation
            return self._calculate_cosine_similarities_numpy(embeddings.cpu().numpy())
    
    def _calculate_cosine_similarities_numpy(self, embeddings: np.ndarray) -> np.ndarray:
        """Fallback numpy implementation for cosine similarities"""
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            
            similarities = cosine_similarity(embeddings)
            np.fill_diagonal(similarities, 0)  # Remove self-similarity
            
            return similarities
            
        except ImportError:
            # Manual cosine similarity calculation
            similarities = np.zeros((len(embeddings), len(embeddings)))
            
            for i in range(len(embeddings)):
                for j in range(len(embeddings)):
                    if i != j:
                        # Cosine similarity: dot product / (norm1 * norm2)
                        dot_product = np.dot(embeddings[i], embeddings[j])
                        norm1 = np.linalg.norm(embeddings[i])
                        norm2 = np.linalg.norm(embeddings[j])
                        
                        if norm1 > 0 and norm2 > 0:
                            similarities[i, j] = dot_product / (norm1 * norm2)
                        else:
                            similarities[i, j] = 0
            
            return similarities
    
    def _calculate_diversity_from_similarities(self, similarities: np.ndarray) -> float:
        """
        Calculate diversity score from similarity matrix
        
        Higher diversity = lower average similarity
        """
        if similarities.size == 0:
            return 0.0
        
        # Calculate average similarity (excluding diagonal)
        avg_similarity = np.mean(similarities)
        
        # Convert to diversity: 1 - average_similarity
        # Higher similarity = lower diversity
        diversity = 1.0 - avg_similarity
        
        # Ensure score is in [0, 1] range
        return max(0.0, min(1.0, diversity))
    
    def calculate_advanced_semantic_diversity(self, texts: List[str]) -> Dict[str, float]:
        """
        Calculate advanced semantic diversity metrics
        
        Returns:
            Dictionary with multiple diversity measures
        """
        if not self.available or not self.model or len(texts) < 2:
            return {
                'overall_diversity': 0.0,
                'intra_cluster_diversity': 0.0,
                'inter_cluster_diversity': 0.0,
                'semantic_spread': 0.0,
                'cluster_coherence': 0.0
            }
        
        try:
            # Generate embeddings
            embeddings = self.model.encode(texts, convert_to_tensor=True, device=self.device)
            embeddings_np = embeddings.cpu().numpy()
            
            # Calculate similarities
            similarities = self._calculate_cosine_similarities(embeddings)
            
            # Advanced metrics
            metrics = {}
            
            # 1. Overall diversity
            metrics['overall_diversity'] = self._calculate_diversity_from_similarities(similarities)
            
            # 2. Semantic clustering analysis
            cluster_metrics = self._analyze_semantic_clusters(embeddings_np, similarities)
            metrics.update(cluster_metrics)
            
            # 3. Semantic spread (how far apart texts are in embedding space)
            metrics['semantic_spread'] = self._calculate_semantic_spread(embeddings_np)
            
            return metrics
            
        except Exception as e:
            print(f"⚠️ Advanced semantic diversity calculation failed: {e}")
            return {
                'overall_diversity': 0.0,
                'intra_cluster_diversity': 0.0,
                'inter_cluster_diversity': 0.0,
                'semantic_spread': 0.0,
                'cluster_coherence': 0.0
            }
    
    def _analyze_semantic_clusters(self, embeddings: np.ndarray, similarities: np.ndarray) -> Dict[str, float]:
        """Analyze semantic clustering patterns"""
        try:
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
            
            # Determine optimal number of clusters (1 to min(5, len(texts)//2))
            max_clusters = min(5, max(1, len(embeddings) // 2))
            
            if max_clusters < 2:
                return {
                    'intra_cluster_diversity': 0.0,
                    'inter_cluster_diversity': 0.0,
                    'cluster_coherence': 0.0
                }
            
            # Try different numbers of clusters
            best_silhouette = -1
            best_n_clusters = 2
            best_labels = None
            
            for n_clusters in range(2, max_clusters + 1):
                try:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(embeddings)
                    
                    if len(set(labels)) > 1:  # At least 2 clusters
                        silhouette = silhouette_score(embeddings, labels)
                        if silhouette > best_silhouette:
                            best_silhouette = silhouette
                            best_n_clusters = n_clusters
                            best_labels = labels
                except:
                    continue
            
            if best_labels is None:
                return {
                    'intra_cluster_diversity': 0.0,
                    'inter_cluster_diversity': 0.0,
                    'cluster_coherence': 0.0
                }
            
            # Calculate intra-cluster diversity (within clusters)
            intra_cluster_similarities = []
            for cluster_id in range(best_n_clusters):
                cluster_indices = np.where(best_labels == cluster_id)[0]
                if len(cluster_indices) > 1:
                    cluster_similarities = []
                    for i in cluster_indices:
                        for j in cluster_indices:
                            if i != j:
                                cluster_similarities.append(similarities[i, j])
                    if cluster_similarities:
                        intra_cluster_similarities.extend(cluster_similarities)
            
            intra_cluster_diversity = 0.0
            if intra_cluster_similarities:
                avg_intra_similarity = np.mean(intra_cluster_similarities)
                intra_cluster_diversity = 1.0 - avg_intra_similarity
            
            # Calculate inter-cluster diversity (between clusters)
            inter_cluster_similarities = []
            for i in range(len(embeddings)):
                for j in range(len(embeddings)):
                    if i != j and best_labels[i] != best_labels[j]:
                        inter_cluster_similarities.append(similarities[i, j])
            
            inter_cluster_diversity = 0.0
            if inter_cluster_similarities:
                avg_inter_similarity = np.mean(inter_cluster_similarities)
                inter_cluster_diversity = 1.0 - avg_inter_similarity
            
            # Cluster coherence (silhouette score)
            cluster_coherence = max(0.0, best_silhouette)  # Ensure non-negative
            
            return {
                'intra_cluster_diversity': intra_cluster_diversity,
                'inter_cluster_diversity': inter_cluster_diversity,
                'cluster_coherence': cluster_coherence
            }
            
        except Exception as e:
            print(f"⚠️ Cluster analysis failed: {e}")
            return {
                'intra_cluster_diversity': 0.0,
                'inter_cluster_diversity': 0.0,
                'cluster_coherence': 0.0
            }
    
    def _calculate_semantic_spread(self, embeddings: np.ndarray) -> float:
        """Calculate semantic spread (how far apart texts are in embedding space)"""
        try:
            # Calculate pairwise Euclidean distances
            distances = []
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    distance = np.linalg.norm(embeddings[i] - embeddings[j])
                    distances.append(distance)
            
            if not distances:
                return 0.0
            
            # Normalize by embedding dimension and convert to spread score
            avg_distance = np.mean(distances)
            embedding_dim = embeddings.shape[1]
            
            # Normalize distance by sqrt of embedding dimension
            normalized_distance = avg_distance / np.sqrt(embedding_dim)
            
            # Convert to 0-1 scale (higher = more spread out)
            spread_score = min(1.0, normalized_distance / 2.0)  # 2.0 is typical max for normalized embeddings
            
            return spread_score
            
        except Exception as e:
            print(f"⚠️ Semantic spread calculation failed: {e}")
            return 0.0
    
    def _calculate_fallback_diversity(self, texts: List[str]) -> float:
        """
        Fallback diversity calculation using word overlap analysis
        (Used when semantic model is not available)
        """
        if len(texts) < 2:
            return 0.0
        
        # Calculate pairwise word overlap
        total_overlap = 0.0
        pair_count = 0
        
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                tokens_i = set(word_tokenize(texts[i]))
                tokens_j = set(word_tokenize(texts[j]))
                
                if tokens_i and tokens_j:
                    overlap = len(tokens_i & tokens_j) / len(tokens_i | tokens_j)
                    total_overlap += overlap
                    pair_count += 1
        
        if pair_count == 0:
            return 0.0
        
        avg_overlap = total_overlap / pair_count
        
        # Convert overlap to diversity (inverse relationship)
        diversity = 1.0 - avg_overlap
        
        return max(0.0, min(1.0, diversity))
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the semantic model"""
        if not self.available:
            return {
                'available': False,
                'model_name': 'None',
                'device': 'None',
                'embedding_dimension': 0
            }
        
        try:
            embedding_dim = self.model.get_sentence_embedding_dimension()
            return {
                'available': True,
                'model_name': self.model_name,
                'device': self.device,
                'embedding_dimension': embedding_dim
            }
        except:
            return {
                'available': True,
                'model_name': self.model_name,
                'device': self.device,
                'embedding_dimension': 'Unknown'
            }


class CreativityDiversityEvaluator:
    """
    Main evaluator for creativity and diversity metrics
    """
    
    def __init__(self):
        self.distinct_calculator = DistinctNCalculator()
        self.self_bleu_calculator = SelfBLEUCalculator()
        self.mauve_calculator = MAUVECalculator()
        self.vocab_calculator = VocabularyRichnessCalculator()
        self.semantic_calculator = SemanticDiversityCalculator()
    
    def evaluate_creativity_diversity(self, generated_texts: List[str], 
                                    reference_texts: List[str] = None) -> CreativityDiversityScores:
        """
        Evaluate creativity and diversity of generated texts
        
        Args:
            generated_texts: List of generated humor texts
            reference_texts: Optional reference texts for MAUVE
            
        Returns:
            CreativityDiversityScores object with all metrics
        """
        if not generated_texts:
            return self._empty_scores()
        
        # Calculate Distinct-n scores
        distinct_1 = self.distinct_calculator.calculate_distinct_n(generated_texts, 1)
        distinct_2 = self.distinct_calculator.calculate_distinct_n(generated_texts, 2)
        distinct_3 = self.distinct_calculator.calculate_distinct_n(generated_texts, 3)
        
        # Calculate Self-BLEU scores
        self_bleu_1 = self.self_bleu_calculator.calculate_self_bleu(generated_texts, 1)
        self_bleu_2 = self.self_bleu_calculator.calculate_self_bleu(generated_texts, 2)
        self_bleu_3 = self.self_bleu_calculator.calculate_self_bleu(generated_texts, 3)
        self_bleu_4 = self.self_bleu_calculator.calculate_self_bleu(generated_texts, 4)
        
        # Calculate MAUVE score (if reference texts available)
        mauve_score = 0.0
        if reference_texts:
            mauve_score = self.mauve_calculator.calculate_mauve_score(
                generated_texts, reference_texts
            )
        
        # Calculate vocabulary richness
        vocabulary_richness = self.vocab_calculator.calculate_vocabulary_richness(generated_texts)
        
        # Calculate semantic diversity
        semantic_diversity = self.semantic_calculator.calculate_semantic_diversity(generated_texts)
        
        # Calculate advanced semantic diversity metrics
        advanced_semantic_metrics = self.semantic_calculator.calculate_advanced_semantic_diversity(generated_texts)
        
        # Calculate overall creativity score
        overall_creativity = self._calculate_overall_creativity(
            distinct_1, distinct_2, distinct_3,
            self_bleu_1, self_bleu_2, self_bleu_3, self_bleu_4,
            mauve_score, vocabulary_richness,
            advanced_semantic_metrics['overall_diversity'],
            advanced_semantic_metrics['intra_cluster_diversity'],
            advanced_semantic_metrics['inter_cluster_diversity'],
            advanced_semantic_metrics['semantic_spread']
        )
        
        return CreativityDiversityScores(
            distinct_1=distinct_1,
            distinct_2=distinct_2,
            distinct_3=distinct_3,
            self_bleu_1=self_bleu_1,
            self_bleu_2=self_bleu_2,
            self_bleu_3=self_bleu_3,
            self_bleu_4=self_bleu_4,
            mauve_score=mauve_score,
            vocabulary_richness=vocabulary_richness,
            overall_semantic_diversity=advanced_semantic_metrics['overall_diversity'],
            intra_cluster_diversity=advanced_semantic_metrics['intra_cluster_diversity'],
            inter_cluster_diversity=advanced_semantic_metrics['inter_cluster_diversity'],
            semantic_spread=advanced_semantic_metrics['semantic_spread'],
            cluster_coherence=advanced_semantic_metrics['cluster_coherence'],
            overall_creativity=overall_creativity
        )
    
    def _calculate_overall_creativity(self, distinct_1: float, distinct_2: float, distinct_3: float,
                                    self_bleu_1: float, self_bleu_2: float, self_bleu_3: float, self_bleu_4: float,
                                    mauve_score: float, vocab_richness: float,
                                    overall_semantic_diversity: float, intra_cluster_diversity: float,
                                    inter_cluster_diversity: float, semantic_spread: float) -> float:
        """
        Calculate overall creativity score based on literature weights
        
        Higher Distinct-n = more creative (positive weight)
        Lower Self-BLEU = more creative (negative weight) 
        Higher MAUVE = better quality (positive weight)
        Higher vocabulary/semantic diversity = more creative (positive weight)
        """
        
        # Distinct-n component (higher is better)
        distinct_component = (distinct_1 * 0.4 + distinct_2 * 0.4 + distinct_3 * 0.2) * 0.3
        
        # Self-BLEU component (lower is better, so invert)
        self_bleu_avg = (self_bleu_1 + self_bleu_2 + self_bleu_3 + self_bleu_4) / 4
        self_bleu_component = (1.0 - self_bleu_avg) * 0.3
        
        # Quality component (MAUVE)
        quality_component = mauve_score * 0.2
        
        # Advanced semantic diversity components
        semantic_diversity_component = (
            overall_semantic_diversity * 0.3 + 
            intra_cluster_diversity * 0.2 + 
            inter_cluster_diversity * 0.2 + 
            semantic_spread * 0.3
        ) * 0.2
        
        # Combine all components
        overall = (distinct_component + self_bleu_component + quality_component + semantic_diversity_component)
        
        # Scale to 0-10 range
        return min(max(overall * 10, 0.0), 10.0)
    
    def _empty_scores(self) -> CreativityDiversityScores:
        """Return empty scores for edge cases"""
        return CreativityDiversityScores(
            distinct_1=0.0, distinct_2=0.0, distinct_3=0.0,
            self_bleu_1=0.0, self_bleu_2=0.0, self_bleu_3=0.0, self_bleu_4=0.0,
            mauve_score=0.0, vocabulary_richness=0.0,
            overall_semantic_diversity=0.0, intra_cluster_diversity=0.0,
            inter_cluster_diversity=0.0, semantic_spread=0.0,
            overall_creativity=0.0
        )


def test_creativity_diversity_evaluator():
    """Test the creativity and diversity evaluator with production-ready semantic analysis"""
    evaluator = CreativityDiversityEvaluator()
    
    # Show semantic model information
    print("🔬 Semantic Diversity Calculator Status:")
    print("=" * 50)
    model_info = evaluator.semantic_calculator.get_model_info()
    for key, value in model_info.items():
        print(f"   {key}: {value}")
    
    # Test cases - diverse vs repetitive outputs
    diverse_texts = [
        "My bank account and I have trust issues",
        "A disappointing potato wearing a tuxedo", 
        "The existential dread of microwaving leftover pizza",
        "Accidentally joining a cult at IKEA",
        "The awkward silence after telling a dad joke"
    ]
    
    repetitive_texts = [
        "My bank account is empty",
        "My wallet is empty", 
        "My purse is empty",
        "My savings are empty",
        "My money is gone"
    ]
    
    reference_texts = [
        "Financial struggles in modern life",
        "The challenge of managing money",
        "Economic difficulties for young adults"
    ]
    
    print("\n🎭 Creativity and Diversity Evaluator Test:")
    print("=" * 60)
    
    print("\n1. Diverse Texts:")
    diverse_scores = evaluator.evaluate_creativity_diversity(diverse_texts, reference_texts)
    print(f"   Distinct-1: {diverse_scores.distinct_1:.3f}")
    print(f"   Distinct-2: {diverse_scores.distinct_2:.3f}")
    print(f"   Distinct-3: {diverse_scores.distinct_3:.3f}")
    print(f"   Self-BLEU-1: {diverse_scores.self_bleu_1:.3f}")
    print(f"   Self-BLEU-2: {diverse_scores.self_bleu_2:.3f}")
    print(f"   Self-BLEU-3: {diverse_scores.self_bleu_3:.3f}")
    print(f"   Self-BLEU-4: {diverse_scores.self_bleu_4:.3f}")
    print(f"   MAUVE: {diverse_scores.mauve_score:.3f}")
    print(f"   Vocab Richness: {diverse_scores.vocabulary_richness:.3f}")
    print(f"   🆕 ADVANCED SEMANTIC METRICS:")
    print(f"     Overall Semantic Diversity: {diverse_scores.overall_semantic_diversity:.3f}")
    print(f"     Intra-cluster Diversity: {diverse_scores.intra_cluster_diversity:.3f}")
    print(f"     Inter-cluster Diversity: {diverse_scores.inter_cluster_diversity:.3f}")
    print(f"     Semantic Spread: {diverse_scores.semantic_spread:.3f}")
    print(f"     Cluster Coherence: {diverse_scores.cluster_coherence:.3f}")
    print(f"   OVERALL CREATIVITY: {diverse_scores.overall_creativity:.2f}/10")
    
    print("\n2. Repetitive Texts:")
    repetitive_scores = evaluator.evaluate_creativity_diversity(repetitive_texts, reference_texts)
    print(f"   Distinct-1: {repetitive_scores.distinct_1:.3f}")
    print(f"   Distinct-2: {repetitive_scores.distinct_2:.3f}")
    print(f"   Distinct-3: {repetitive_scores.distinct_3:.3f}")
    print(f"   Self-BLEU-1: {repetitive_scores.self_bleu_1:.3f}")
    print(f"   Self-BLEU-2: {repetitive_scores.self_bleu_2:.3f}")
    print(f"   Self-BLEU-3: {repetitive_scores.self_bleu_3:.3f}")
    print(f"   Self-BLEU-4: {repetitive_scores.self_bleu_4:.3f}")
    print(f"   MAUVE: {repetitive_scores.mauve_score:.3f}")
    print(f"   Vocab Richness: {repetitive_scores.vocabulary_richness:.3f}")
    print(f"   🆕 ADVANCED SEMANTIC METRICS:")
    print(f"     Overall Semantic Diversity: {repetitive_scores.overall_semantic_diversity:.3f}")
    print(f"     Intra-cluster Diversity: {repetitive_scores.intra_cluster_diversity:.3f}")
    print(f"     Inter-cluster Diversity: {repetitive_scores.inter_cluster_diversity:.3f}")
    print(f"     Semantic Spread: {repetitive_scores.semantic_spread:.3f}")
    print(f"     Cluster Coherence: {repetitive_scores.cluster_coherence:.3f}")
    print(f"   OVERALL CREATIVITY: {repetitive_scores.overall_creativity:.2f}/10")
    
    print(f"\n📊 Comparison:")
    print(f"   Diverse texts creativity: {diverse_scores.overall_creativity:.2f}/10")
    print(f"   Repetitive texts creativity: {repetitive_scores.overall_creativity:.2f}/10")
    print(f"   Difference: {diverse_scores.overall_creativity - repetitive_scores.overall_creativity:.2f}")
    
    print(f"\n🔬 Semantic Analysis Insights:")
    print(f"   Diverse texts semantic spread: {diverse_scores.semantic_spread:.3f}")
    print(f"   Repetitive texts semantic spread: {repetitive_scores.semantic_spread:.3f}")
    print(f"   Diverse texts cluster coherence: {diverse_scores.cluster_coherence:.3f}")
    print(f"   Repetitive texts cluster coherence: {repetitive_scores.cluster_coherence:.3f}")


if __name__ == "__main__":
    test_creativity_diversity_evaluator()
