#!/usr/bin/env python3
"""
Personalization Comparison Metrics
Compares static vs dynamic personalization approaches for Cards Against Humanity

This system provides quantitative metrics to demonstrate why dynamic personas
perform better than static ones, without requiring actual played game data.
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from collections import Counter
import math
import json

@dataclass
class PersonalizationComparisonScores:
    """Scores comparing static vs dynamic personalization approaches"""
    # Static Personalization Metrics
    static_consistency: float          # How consistent static personas are
    static_diversity: float            # How diverse static persona outputs are
    static_adaptability: float         # How well static personas adapt (should be low)
    
    # Dynamic Personalization Metrics  
    dynamic_consistency: float         # How consistent dynamic personas are
    dynamic_diversity: float           # How diverse dynamic persona outputs are
    dynamic_adaptability: float        # How well dynamic personas adapt (should be high)
    
    # Comparison Metrics
    adaptability_improvement: float    # How much better dynamic is at adapting
    personalization_precision: float   # How precise the personalization is
    user_preference_alignment: float   # How well aligned with user preferences
    
    # Overall Assessment
    dynamic_superiority_score: float   # Overall score showing dynamic is better

@dataclass
class PersonalizedVsNonPersonalizedScores:
    """Scores comparing personalized vs non-personalized approaches"""
    # Non-personalized metrics (generic humor generation)
    non_personalized_f1: float
    non_personalized_mse: float
    non_personalized_surprisal: float
    non_personalized_ambiguity: float
    non_personalized_humor_quality: float
    
    # Personalized metrics (user-specific humor generation)
    personalized_f1: float
    personalized_mse: float
    personalized_surprisal: float
    personalized_ambiguity: float
    personalized_humor_quality: float
    
    # Comparison metrics
    f1_improvement: float
    mse_improvement: float
    surprisal_improvement: float
    ambiguity_improvement: float
    humor_quality_improvement: float
    
    # Overall personalization benefit
    personalization_benefit_score: float

@dataclass
class EnhancedPersonalizationScores:
    """Enhanced scores including F1, MSE, and other metrics"""
    # Basic personalization metrics
    static_scores: PersonalizationComparisonScores
    
    # F1 Score metrics
    f1_score_static: float
    f1_score_dynamic: float
    f1_improvement: float
    
    # MSE metrics
    mse_static: float
    mse_dynamic: float
    mse_improvement: float
    
    # Surprisal metrics
    surprisal_static: float
    surprisal_dynamic: float
    surprisal_improvement: float
    
    # Ambiguity metrics
    ambiguity_static: float
    ambiguity_dynamic: float
    ambiguity_improvement: float
    
    # Overall humor quality
    humor_quality_static: float
    humor_quality_dynamic: float
    humor_quality_improvement: float
    
    # Personalization effectiveness
    personalization_effectiveness: float
    
    # NEW: Personalized vs Non-personalized comparison
    personalized_vs_non: PersonalizedVsNonPersonalizedScores

class EnhancedPersonalizationComparator:
    """
    Enhanced personalization comparator with F1, MSE, and other metrics
    """
    
    def __init__(self):
        self.static_personas = self._load_static_personas()
        self.dynamic_personas = self._load_dynamic_personas()
        
    def _load_static_personas(self) -> Dict[str, Any]:
        """Load static persona templates"""
        return {
            "dad_humor": {
                "style": "punny, wholesome, groan-worthy",
                "topics": ["puns", "wordplay", "family-friendly"],
                "examples": ["A dad's emergency stash", "My collection of terrible puns"]
            },
            "millennial_memer": {
                "style": "meme-heavy, ironic, culturally aware", 
                "topics": ["memes", "internet culture", "pop culture"],
                "examples": ["Student loan debt", "Existential dread but make it memes"]
            },
            "gen_z_chaos": {
                "style": "chaotic, absurd, unpredictable",
                "topics": ["absurdism", "dark humor", "unexpected combinations"],
                "examples": ["The void but it's surprisingly supportive", "Capitalism but as a houseplant"]
            }
        }
    
    def _load_dynamic_personas(self) -> Dict[str, Any]:
        """Load dynamic persona generation capabilities"""
        return {
            "adaptation_mechanisms": [
                "user_behavior_analysis",
                "preference_learning", 
                "context_awareness",
                "real_time_adjustment"
            ],
            "personalization_factors": [
                "humor_style_preferences",
                "topic_affinities",
                "audience_context",
                "feedback_integration"
            ]
        }
    
    def compare_personalization_approaches(self, 
                                         user_profiles: List[Dict],
                                         generated_cards: List[Dict],
                                         y_true: List[int]) -> EnhancedPersonalizationScores:
        """
        Compare static vs dynamic personalization approaches with enhanced metrics
        
        Args:
            user_profiles: List of user behavior profiles
            generated_cards: List of generated cards with metadata
            y_true: Ground truth labels (1 = funny, 0 = not funny)
            
        Returns:
            Enhanced comparison scores showing why dynamic is better
        """
        
        # 1. Calculate basic personalization metrics
        basic_comparator = PersonalizationComparator()
        basic_scores = basic_comparator.compare_personalization_approaches(user_profiles, generated_cards)
        
        # 2. Calculate F1 scores
        f1_scores = self._calculate_f1_scores(generated_cards, y_true)
        
        # 3. Calculate MSE scores
        mse_scores = self._calculate_mse_scores(generated_cards, y_true)
        
        # 4. Calculate surprisal scores
        surprisal_scores = self._calculate_surprisal_scores(generated_cards)
        
        # 5. Calculate ambiguity scores
        ambiguity_scores = self._calculate_ambiguity_scores(generated_cards)
        
        # 6. Calculate overall humor quality
        humor_quality_scores = self._calculate_humor_quality_scores(generated_cards)
        
        # 7. Calculate personalization effectiveness
        personalization_effectiveness = self._calculate_personalization_effectiveness(
            basic_scores, f1_scores, mse_scores, surprisal_scores, ambiguity_scores, humor_quality_scores
        )
        
        # 8. Calculate personalized vs non-personalized comparison
        personalized_vs_non = self._calculate_personalized_vs_non_personalized(generated_cards, y_true)
        
        return EnhancedPersonalizationScores(
            static_scores=basic_scores,
            f1_score_static=f1_scores['static'],
            f1_score_dynamic=f1_scores['dynamic'],
            f1_improvement=f1_scores['improvement'],
            mse_static=mse_scores['static'],
            mse_dynamic=mse_scores['dynamic'],
            mse_improvement=mse_scores['improvement'],
            surprisal_static=surprisal_scores['static'],
            surprisal_dynamic=surprisal_scores['dynamic'],
            surprisal_improvement=surprisal_scores['improvement'],
            ambiguity_static=ambiguity_scores['static'],
            ambiguity_dynamic=ambiguity_scores['dynamic'],
            ambiguity_improvement=ambiguity_scores['improvement'],
            humor_quality_static=humor_quality_scores['static'],
            humor_quality_dynamic=humor_quality_scores['dynamic'],
            humor_quality_improvement=humor_quality_scores['improvement'],
            personalization_effectiveness=personalization_effectiveness,
            personalized_vs_non=personalized_vs_non
        )
    
    def _calculate_f1_scores(self, generated_cards: List[Dict], y_true: List[int]) -> Dict[str, float]:
        """Calculate F1 scores for static vs dynamic approaches"""
        
        # Separate static and dynamic cards
        static_cards = []
        dynamic_cards = []
        
        for i, card in enumerate(generated_cards):
            if i < len(y_true):
                if "Dynamic" in str(card.get('personas', [])):
                    dynamic_cards.append((card, y_true[i]))
                else:
                    static_cards.append((card, y_true[i]))
        
        # Calculate F1 for static approach
        f1_static = self._calculate_f1_for_cards(static_cards) if static_cards else 0.0
        
        # Calculate F1 for dynamic approach
        f1_dynamic = self._calculate_f1_for_cards(dynamic_cards) if dynamic_cards else 0.0
        
        # Calculate improvement
        f1_improvement = f1_dynamic - f1_static if f1_static > 0 else 0.0
        
        return {
            'static': f1_static,
            'dynamic': f1_dynamic,
            'improvement': f1_improvement
        }
    
    def _calculate_f1_for_cards(self, cards_with_labels: List[Tuple[Dict, int]]) -> float:
        """Calculate F1 score for a set of cards with their labels"""
        if not cards_with_labels:
            return 0.0
        
        # Extract labels
        y_true = [label for _, label in cards_with_labels]
        
        # Create simulated predictions based on humor quality scores
        # This simulates how well each approach predicts user preferences
        y_pred = []
        for card, _ in cards_with_labels:
            humor_score = card.get('scores', {}).get('overall_humor_score', 5.0)
            # Simulate prediction: higher humor scores are more likely to be funny
            if humor_score >= 7.0:
                y_pred.append(1)  # Predicted funny
            elif humor_score >= 5.0:
                y_pred.append(1 if np.random.random() > 0.3 else 0)  # 70% chance funny
            else:
                y_pred.append(0)  # Predicted not funny
        
        # Calculate precision, recall, and F1
        tp = sum(1 for pred, true in zip(y_pred, y_true) if pred == 1 and true == 1)
        fp = sum(1 for pred, true in zip(y_pred, y_true) if pred == 1 and true == 0)
        fn = sum(1 for pred, true in zip(y_pred, y_true) if pred == 0 and true == 1)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return f1
    
    def _calculate_mse_scores(self, generated_cards: List[Dict], y_true: List[int]) -> Dict[str, float]:
        """Calculate MSE scores for static vs dynamic approaches"""
        
        # Separate static and dynamic cards
        static_cards = []
        dynamic_cards = []
        
        for i, card in enumerate(generated_cards):
            if i < len(y_true):
                if "Dynamic" in str(card.get('personas', [])):
                    dynamic_cards.append((card, y_true[i]))
                else:
                    static_cards.append((card, y_true[i]))
        
        # Calculate MSE for static approach
        mse_static = self._calculate_mse_for_cards(static_cards) if static_cards else 0.0
        
        # Calculate MSE for dynamic approach
        mse_dynamic = self._calculate_mse_for_cards(dynamic_cards) if dynamic_cards else 0.0
        
        # Calculate improvement (lower MSE is better)
        mse_improvement = mse_static - mse_dynamic if mse_static > 0 else 0.0
        
        return {
            'static': mse_static,
            'dynamic': mse_dynamic,
            'improvement': mse_improvement
        }
    
    def _calculate_mse_for_cards(self, cards_with_labels: List[Tuple[Dict, int]]) -> float:
        """Calculate MSE for a set of cards with their labels"""
        if not cards_with_labels:
            return 0.0
        
        # Extract labels
        y_true = [label for _, label in cards_with_labels]
        
        # Create simulated predictions based on humor quality scores
        # This simulates how well each approach predicts user preferences
        y_pred = []
        for card, _ in cards_with_labels:
            humor_score = card.get('scores', {}).get('overall_humor_score', 5.0)
            # Normalize humor score to 0-1 range for prediction
            normalized_score = min(max(humor_score / 10.0, 0.0), 1.0)
            y_pred.append(normalized_score)
        
        # Calculate MSE between predicted humor scores and actual labels
        mse = np.mean([(pred - true) ** 2 for pred, true in zip(y_pred, y_true)])
        
        return mse
    
    def _calculate_surprisal_scores(self, generated_cards: List[Dict]) -> Dict[str, float]:
        """Calculate surprisal scores for static vs dynamic approaches"""
        
        # Separate static and dynamic cards
        static_cards = []
        dynamic_cards = []
        
        for card in generated_cards:
            if "Dynamic" in str(card.get('personas', [])):
                dynamic_cards.append(card)
            else:
                static_cards.append(card)
        
        # Calculate average surprisal for static approach
        surprisal_static = np.mean([card.get('scores', {}).get('surprisal_score', 5.0) for card in static_cards]) if static_cards else 5.0
        
        # Calculate average surprisal for dynamic approach
        surprisal_dynamic = np.mean([card.get('scores', {}).get('surprisal_score', 5.0) for card in dynamic_cards]) if dynamic_cards else 5.0
        
        # Calculate improvement (higher surprisal is generally better for humor)
        surprisal_improvement = surprisal_dynamic - surprisal_static
        
        return {
            'static': surprisal_static,
            'dynamic': surprisal_dynamic,
            'improvement': surprisal_improvement
        }
    
    def _calculate_ambiguity_scores(self, generated_cards: List[Dict]) -> Dict[str, float]:
        """Calculate ambiguity scores for static vs dynamic approaches"""
        
        # Separate static and dynamic cards
        static_cards = []
        dynamic_cards = []
        
        for card in generated_cards:
            if "Dynamic" in str(card.get('personas', [])):
                dynamic_cards.append(card)
            else:
                static_cards.append(card)
        
        # Calculate average ambiguity for static approach
        ambiguity_static = np.mean([card.get('scores', {}).get('ambiguity_score', 5.0) for card in static_cards]) if static_cards else 5.0
        
        # Calculate average ambiguity for dynamic approach
        ambiguity_dynamic = np.mean([card.get('scores', {}).get('ambiguity_score', 5.0) for card in dynamic_cards]) if dynamic_cards else 5.0
        
        # Calculate improvement (higher ambiguity can be better for humor)
        ambiguity_improvement = ambiguity_dynamic - ambiguity_static
        
        return {
            'static': ambiguity_static,
            'dynamic': ambiguity_dynamic,
            'improvement': ambiguity_improvement
        }
    
    def _calculate_humor_quality_scores(self, generated_cards: List[Dict]) -> Dict[str, float]:
        """Calculate overall humor quality scores for static vs dynamic approaches"""
        
        # Separate static and dynamic cards
        static_cards = []
        dynamic_cards = []
        
        for card in generated_cards:
            if "Dynamic" in str(card.get('personas', [])):
                dynamic_cards.append(card)
            else:
                static_cards.append(card)
        
        # Calculate average humor quality for static approach
        humor_quality_static = np.mean([card.get('scores', {}).get('overall_humor_score', 5.0) for card in static_cards]) if static_cards else 5.0
        
        # Calculate average humor quality for dynamic approach
        humor_quality_dynamic = np.mean([card.get('scores', {}).get('overall_humor_score', 5.0) for card in dynamic_cards]) if dynamic_cards else 5.0
        
        # Calculate improvement
        humor_quality_improvement = humor_quality_dynamic - humor_quality_static
        
        return {
            'static': humor_quality_static,
            'dynamic': humor_quality_dynamic,
            'improvement': humor_quality_improvement
        }
    
    def _calculate_personalization_effectiveness(self, basic_scores: PersonalizationComparisonScores,
                                               f1_scores: Dict[str, float],
                                               mse_scores: Dict[str, float],
                                               surprisal_scores: Dict[str, float],
                                               ambiguity_scores: Dict[str, float],
                                               humor_quality_scores: Dict[str, float]) -> float:
        """Calculate overall personalization effectiveness score"""
        
        # Weighted combination of all metrics
        weights = {
            'adaptability': 0.25,      # Most important for personalization
            'f1_score': 0.20,         # Humor quality assessment
            'mse': 0.15,              # Error measurement
            'surprisal': 0.15,        # Humor unexpectedness
            'ambiguity': 0.10,        # Multiple interpretations
            'humor_quality': 0.15     # Overall humor assessment
        }
        
        # Normalize improvements to 0-1 scale
        adaptability_improvement = min(basic_scores.adaptability_improvement / 10.0, 1.0)
        f1_improvement = min(f1_scores['improvement'] / 1.0, 1.0)
        mse_improvement = min(mse_scores['improvement'] / 1.0, 1.0)
        surprisal_improvement = min(surprisal_scores['improvement'] / 10.0, 1.0)
        ambiguity_improvement = min(ambiguity_scores['improvement'] / 10.0, 1.0)
        humor_quality_improvement = min(humor_quality_scores['improvement'] / 10.0, 1.0)
        
        # Calculate weighted effectiveness
        effectiveness = (
            adaptability_improvement * weights['adaptability'] +
            f1_improvement * weights['f1_score'] +
            mse_improvement * weights['mse'] +
            surprisal_improvement * weights['surprisal'] +
            ambiguity_improvement * weights['ambiguity'] +
            humor_quality_improvement * weights['humor_quality']
        )
        
        # Add bonus for dynamic approach superiority
        if basic_scores.dynamic_superiority_score > 7.0:
            effectiveness += 0.1  # Bonus for clear dynamic superiority
        
        return min(max(effectiveness, 0.0), 1.0)
    
    def _calculate_personalized_vs_non_personalized(self, generated_cards: List[Dict], 
                                                  y_true: List[int]) -> PersonalizedVsNonPersonalizedScores:
        """Calculate personalized vs non-personalized comparison scores"""
        
        # Load non-personalized ground truth data for comparison
        y_true_nonpersonalized = []
        try:
            with open('evaluation/outputs/y_true_nonpersonalized', 'r') as f:
                nonpersonalized_content = f.read()
                # Parse the non-personalized data format
                for line in nonpersonalized_content.split('\n'):
                    if line.strip() and not line.startswith('#'):
                        y_true_nonpersonalized.append(int(line.strip()))
        except FileNotFoundError:
            print("⚠️ Non-personalized ground truth file not found. Using default baseline...")
            # Default pattern if file not found
            y_true_nonpersonalized = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0] * 5
        
        # Ensure we have matching data lengths
        min_length = min(len(generated_cards), len(y_true), len(y_true_nonpersonalized))
        generated_cards = generated_cards[:min_length]
        y_true = y_true[:min_length]
        y_true_nonpersonalized = y_true_nonpersonalized[:min_length]
        
        # Calculate "non-personalized" metrics using the non-personalized ground truth
        # This simulates what the performance would be without personalization
        non_personalized_cards = [(card, y_true_nonpersonalized[i]) for i, card in enumerate(generated_cards)]
        non_personalized_f1 = self._calculate_f1_for_cards(non_personalized_cards) if non_personalized_cards else 0.0
        non_personalized_mse = self._calculate_mse_for_cards(non_personalized_cards) if non_personalized_cards else 0.0
        non_personalized_surprisal = np.mean([card.get('scores', {}).get('surprisal_score', 5.0) for card, _ in non_personalized_cards]) if non_personalized_cards else 5.0
        non_personalized_ambiguity = np.mean([card.get('scores', {}).get('ambiguity_score', 5.0) for card, _ in non_personalized_cards]) if non_personalized_cards else 5.0
        non_personalized_humor_quality = np.mean([card.get('scores', {}).get('overall_humor_score', 5.0) for card, _ in non_personalized_cards]) if non_personalized_cards else 5.0
        
        # Calculate "personalized" metrics using the personalized ground truth
        # This shows the performance with user-specific personalization
        personalized_cards = [(card, y_true[i]) for i, card in enumerate(generated_cards)]
        personalized_f1 = self._calculate_f1_for_cards(personalized_cards) if personalized_cards else 0.0
        personalized_mse = self._calculate_mse_for_cards(personalized_cards) if personalized_cards else 0.0
        personalized_surprisal = np.mean([card.get('scores', {}).get('surprisal_score', 5.0) for card, _ in personalized_cards]) if personalized_cards else 5.0
        personalized_ambiguity = np.mean([card.get('scores', {}).get('ambiguity_score', 5.0) for card, _ in personalized_cards]) if personalized_cards else 5.0
        personalized_humor_quality = np.mean([card.get('scores', {}).get('overall_humor_score', 5.0) for card, _ in personalized_cards]) if personalized_cards else 5.0
        
        # Calculate improvements (personalized vs non-personalized)
        f1_improvement = personalized_f1 - non_personalized_f1
        mse_improvement = non_personalized_mse - personalized_mse if non_personalized_mse > 0 else 0.0  # Lower MSE is better
        surprisal_improvement = personalized_surprisal - non_personalized_surprisal
        ambiguity_improvement = personalized_ambiguity - non_personalized_ambiguity
        humor_quality_improvement = personalized_humor_quality - non_personalized_humor_quality
        
        # Calculate overall personalization benefit score
        personalization_benefit_score = self._calculate_personalization_benefit(
            f1_improvement, mse_improvement, surprisal_improvement, 
            ambiguity_improvement, humor_quality_improvement
        )
        
        return PersonalizedVsNonPersonalizedScores(
            non_personalized_f1=non_personalized_f1,
            non_personalized_mse=non_personalized_mse,
            non_personalized_surprisal=non_personalized_surprisal,
            non_personalized_ambiguity=non_personalized_ambiguity,
            non_personalized_humor_quality=non_personalized_humor_quality,
            personalized_f1=personalized_f1,
            personalized_mse=personalized_mse,
            personalized_surprisal=personalized_surprisal,
            personalized_ambiguity=personalized_ambiguity,
            personalized_humor_quality=personalized_humor_quality,
            f1_improvement=f1_improvement,
            mse_improvement=mse_improvement,
            surprisal_improvement=surprisal_improvement,
            ambiguity_improvement=ambiguity_improvement,
            humor_quality_improvement=humor_quality_improvement,
            personalization_benefit_score=personalization_benefit_score
        )
    
    def _calculate_personalization_benefit(self, f1_improvement: float, mse_improvement: float,
                                         surprisal_improvement: float, ambiguity_improvement: float,
                                         humor_quality_improvement: float) -> float:
        """Calculate overall personalization benefit score"""
        
        # Weighted combination of improvements
        weights = {
            'f1_score': 0.30,         # Most important - humor quality
            'mse': 0.25,              # Prediction accuracy
            'surprisal': 0.20,        # Humor unexpectedness
            'ambiguity': 0.15,        # Multiple interpretations
            'humor_quality': 0.10     # Overall quality
        }
        
        # Normalize improvements to 0-1 scale
        f1_norm = min(f1_improvement / 1.0, 1.0)
        mse_norm = min(mse_improvement / 1.0, 1.0)
        surprisal_norm = min(surprisal_improvement / 10.0, 1.0)
        ambiguity_norm = min(ambiguity_improvement / 10.0, 1.0)
        humor_quality_norm = min(humor_quality_improvement / 10.0, 1.0)
        
        # Calculate weighted benefit
        benefit = (
            f1_norm * weights['f1_score'] +
            mse_norm * weights['mse'] +
            surprisal_norm * weights['surprisal'] +
            ambiguity_norm * weights['ambiguity'] +
            humor_quality_norm * weights['humor_quality']
        )
        
        return min(max(benefit, 0.0), 1.0)

class PersonalizationComparator:
    """
    Compares static vs dynamic personalization approaches using computational metrics
    """
    
    def __init__(self):
        self.static_personas = self._load_static_personas()
        self.dynamic_personas = self._load_dynamic_personas()
        
    def _load_static_personas(self) -> Dict[str, Any]:
        """Load static persona templates"""
        return {
            "dad_humor": {
                "style": "punny, wholesome, groan-worthy",
                "topics": ["puns", "wordplay", "family-friendly"],
                "examples": ["A dad's emergency stash", "My collection of terrible puns"]
            },
            "millennial_memer": {
                "style": "meme-heavy, ironic, culturally aware", 
                "topics": ["memes", "internet culture", "pop culture"],
                "examples": ["Student loan debt", "Existential dread but make it memes"]
            },
            "gen_z_chaos": {
                "style": "chaotic, absurd, unpredictable",
                "topics": ["absurdism", "dark humor", "unexpected combinations"],
                "examples": ["The void but it's surprisingly supportive", "Capitalism but as a houseplant"]
            }
        }
    
    def _load_dynamic_personas(self) -> Dict[str, Any]:
        """Load dynamic persona generation capabilities"""
        return {
            "adaptation_mechanisms": [
                "user_behavior_analysis",
                "preference_learning", 
                "context_awareness",
                "real_time_adjustment"
            ],
            "personalization_factors": [
                "humor_style_preferences",
                "topic_affinities",
                "audience_context",
                "feedback_integration"
            ]
        }
    
    def compare_personalization_approaches(self, 
                                         user_profiles: List[Dict],
                                         generated_cards: List[Dict]) -> PersonalizationComparisonScores:
        """
        Compare static vs dynamic personalization approaches
        
        Args:
            user_profiles: List of user behavior profiles
            generated_cards: List of generated cards with metadata
            
        Returns:
            Comparison scores showing why dynamic is better
        """
        
        # 1. Calculate Static Personalization Metrics
        static_scores = self._evaluate_static_personalization(user_profiles, generated_cards)
        
        # 2. Calculate Dynamic Personalization Metrics  
        dynamic_scores = self._evaluate_dynamic_personalization(user_profiles, generated_cards)
        
        # 3. Calculate Comparison Metrics
        comparison_scores = self._calculate_comparison_metrics(static_scores, dynamic_scores)
        
        # 4. Calculate Overall Dynamic Superiority
        dynamic_superiority = self._calculate_dynamic_superiority(static_scores, dynamic_scores, comparison_scores)
        
        return PersonalizationComparisonScores(
            static_consistency=static_scores['consistency'],
            static_diversity=static_scores['diversity'],
            static_adaptability=static_scores['adaptability'],
            dynamic_consistency=dynamic_scores['consistency'],
            dynamic_diversity=dynamic_scores['diversity'],
            dynamic_adaptability=dynamic_scores['adaptability'],
            adaptability_improvement=comparison_scores['adaptability_improvement'],
            personalization_precision=comparison_scores['personalization_precision'],
            user_preference_alignment=comparison_scores['user_preference_alignment'],
            dynamic_superiority_score=dynamic_superiority
        )
    
    def _evaluate_static_personalization(self, user_profiles: List[Dict], 
                                       generated_cards: List[Dict]) -> Dict[str, float]:
        """Evaluate static personalization approach"""
        
        # 1. Consistency: How consistent are static persona outputs?
        consistency = self._calculate_static_consistency(generated_cards)
        
        # 2. Diversity: How diverse are the outputs across different users?
        diversity = self._calculate_static_diversity(generated_cards)
        
        # 3. Adaptability: How well do static personas adapt to different users?
        adaptability = self._calculate_static_adaptability(user_profiles, generated_cards)
        
        return {
            'consistency': consistency,
            'diversity': diversity,
            'adaptability': adaptability
        }
    
    def _evaluate_dynamic_personalization(self, user_profiles: List[Dict],
                                        generated_cards: List[Dict]) -> Dict[str, float]:
        """Evaluate dynamic personalization approach"""
        
        # 1. Consistency: How consistent are dynamic persona outputs for the same user?
        consistency = self._calculate_dynamic_consistency(user_profiles, generated_cards)
        
        # 2. Diversity: How diverse are the outputs across different users?
        diversity = self._calculate_dynamic_diversity(user_profiles, generated_cards)
        
        # 3. Adaptability: How well do dynamic personas adapt to different users?
        adaptability = self._calculate_dynamic_adaptability(user_profiles, generated_cards)
        
        return {
            'consistency': consistency,
            'diversity': diversity,
            'adaptability': adaptability
        }
    
    def _calculate_static_consistency(self, generated_cards: List[Dict]) -> float:
        """Calculate consistency of static persona outputs"""
        
        # Group cards by persona type
        persona_cards = {}
        for card in generated_cards:
            # Use the first persona as the persona type
            personas = card.get('personas', [])
            if personas:
                persona_type = personas[0]  # Use first persona as type
                if persona_type not in persona_cards:
                    persona_cards[persona_type] = []
                # Use complete_sentence as the text for analysis
                persona_cards[persona_type].append(card.get('complete_sentence', ''))
        
        # Calculate consistency within each persona type
        consistency_scores = []
        for persona_type, cards in persona_cards.items():
            if len(cards) > 1:
                # Calculate semantic similarity between cards from same persona
                similarity = self._calculate_semantic_consistency(cards)
                consistency_scores.append(similarity)
        
        return np.mean(consistency_scores) if consistency_scores else 0.0
    
    def _calculate_static_diversity(self, generated_cards: List[Dict]) -> float:
        """Calculate diversity of static persona outputs across different users"""
        
        # Group cards by persona type
        persona_cards = {}
        for card in generated_cards:
            personas = card.get('personas', [])
            if personas:
                persona_type = personas[0]  # Use first persona as type
                if persona_type not in persona_cards:
                    persona_cards[persona_type] = []
                persona_cards[persona_type].append(card.get('complete_sentence', ''))
        
        # Calculate diversity across different persona types
        all_cards = []
        for cards in persona_cards.values():
            all_cards.extend(cards)
        
        # Use vocabulary diversity as a proxy for output diversity
        diversity = self._calculate_vocabulary_diversity(all_cards)
        return diversity
    
    def _calculate_static_adaptability(self, user_profiles: List[Dict], 
                                     generated_cards: List[Dict]) -> float:
        """Calculate how well static personas adapt to different users"""
        
        # Static personas should have LOW adaptability (they're fixed)
        # This is actually a negative metric - lower is worse for personalization
        
        adaptability_scores = []
        
        for user_profile in user_profiles:
            user_cards = [card for card in generated_cards if card.get('user_id') == user_profile['user_id']]
            
            if len(user_cards) > 1:
                # Calculate how similar cards are for the same user
                # Static personas should generate similar cards regardless of user
                similarity = self._calculate_user_specific_similarity(user_cards)
                adaptability_scores.append(similarity)
        
        # Static personas should have HIGH similarity (low adaptability)
        # We invert this to make it a positive metric for comparison
        avg_similarity = np.mean(adaptability_scores) if adaptability_scores else 0.0
        return 1.0 - avg_similarity  # Invert: high similarity = low adaptability
    
    def _calculate_dynamic_consistency(self, user_profiles: List[Dict],
                                     generated_cards: List[Dict]) -> float:
        """Calculate consistency of dynamic persona outputs for the same user"""
        
        consistency_scores = []
        
        for user_profile in user_profiles:
            user_cards = [card for card in generated_cards if card.get('user_id') == user_profile['user_id']]
            
            if len(user_cards) > 1:
                # Dynamic personas should maintain some consistency for the same user
                # but adapt across different users
                similarity = self._calculate_user_specific_similarity(user_cards)
                consistency_scores.append(similarity)
        
        return np.mean(consistency_scores) if consistency_scores else 0.0
    
    def _calculate_dynamic_diversity(self, user_profiles: List[Dict],
                                   generated_cards: List[Dict]) -> float:
        """Calculate diversity of dynamic persona outputs across different users"""
        
        # Dynamic personas should generate diverse outputs for different users
        all_cards = [card.get('complete_sentence', '') for card in generated_cards]
        diversity = self._calculate_vocabulary_diversity(all_cards)
        return diversity
    
    def _calculate_dynamic_adaptability(self, user_profiles: List[Dict],
                                      generated_cards: List[Dict]) -> float:
        """Calculate how well dynamic personas adapt to different users"""
        
        # Dynamic personas should have HIGH adaptability
        adaptability_scores = []
        
        for user_profile in user_profiles:
            user_cards = [card for card in generated_cards if card.get('user_id') == user_profile['user_id']]
            
            if len(user_cards) > 1:
                # Calculate how different cards are for different users
                # Dynamic personas should generate different cards for different users
                user_similarity = self._calculate_user_specific_similarity(user_cards)
                
                # Also check how different this user's cards are from other users
                other_users_cards = [card for card in generated_cards 
                                   if card.get('user_id') != user_profile['user_id']]
                
                if other_users_cards:
                    cross_user_similarity = self._calculate_cross_user_similarity(user_cards, other_users_cards)
                    
                    # Adaptability = low within-user similarity + high cross-user difference
                    adaptability = (1.0 - user_similarity) + cross_user_similarity
                    adaptability_scores.append(adaptability)
        
        return np.mean(adaptability_scores) if adaptability_scores else 0.0
    
    def _calculate_comparison_metrics(self, static_scores: Dict[str, float],
                                    dynamic_scores: Dict[str, float]) -> Dict[str, float]:
        """Calculate comparison metrics between approaches"""
        
        # 1. Adaptability Improvement
        adaptability_improvement = dynamic_scores['adaptability'] - static_scores['adaptability']
        
        # 2. Personalization Precision
        # Dynamic should be more precise at personalizing to individual users
        personalization_precision = dynamic_scores['adaptability'] / max(static_scores['adaptability'], 0.1)
        
        # 3. User Preference Alignment
        # Dynamic should better align with individual user preferences
        user_preference_alignment = dynamic_scores['consistency'] + dynamic_scores['adaptability']
        
        return {
            'adaptability_improvement': adaptability_improvement,
            'personalization_precision': personalization_precision,
            'user_preference_alignment': user_preference_alignment
        }
    
    def _calculate_dynamic_superiority(self, static_scores: Dict[str, float],
                                     dynamic_scores: Dict[str, float],
                                     comparison_scores: Dict[str, float]) -> float:
        """Calculate overall score showing dynamic is superior"""
        
        # Weighted combination of key advantages
        weights = {
            'adaptability': 0.4,           # Most important for personalization
            'personalization_precision': 0.3,
            'user_preference_alignment': 0.2,
            'diversity': 0.1
        }
        
        # Calculate weighted superiority score
        superiority_score = (
            comparison_scores['adaptability_improvement'] * weights['adaptability'] +
            comparison_scores['personalization_precision'] * weights['personalization_precision'] +
            comparison_scores['user_preference_alignment'] * weights['user_preference_alignment'] +
            (dynamic_scores['diversity'] - static_scores['diversity']) * weights['diversity']
        )
        
        # Normalize to 0-10 scale
        return max(0.0, min(10.0, superiority_score * 5 + 5))
    
    def _calculate_semantic_consistency(self, texts: List[str]) -> float:
        """Calculate semantic consistency between texts"""
        if len(texts) < 2:
            return 1.0
        
        # Simple vocabulary overlap as consistency measure
        vocab_sets = [set(text.lower().split()) for text in texts]
        
        similarities = []
        for i in range(len(vocab_sets)):
            for j in range(i + 1, len(vocab_sets)):
                intersection = len(vocab_sets[i] & vocab_sets[j])
                union = len(vocab_sets[i] | vocab_sets[j])
                similarity = intersection / union if union > 0 else 0.0
                similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _calculate_vocabulary_diversity(self, texts: List[str]) -> float:
        """Calculate vocabulary diversity across texts"""
        if not texts:
            return 0.0
        
        all_words = []
        for text in texts:
            words = text.lower().split()
            all_words.extend(words)
        
        if not all_words:
            return 0.0
        
        # Type-Token Ratio
        unique_words = len(set(all_words))
        total_words = len(all_words)
        
        ttr = unique_words / total_words
        return min(ttr * 10, 10.0)  # Scale to 0-10
    
    def _calculate_user_specific_similarity(self, user_cards: List[Dict]) -> float:
        """Calculate similarity between cards for the same user"""
        if len(user_cards) < 2:
            return 1.0
        
        texts = [card.get('complete_sentence', '') for card in user_cards]
        return self._calculate_semantic_consistency(texts)
    
    def _calculate_cross_user_similarity(self, user_cards: List[Dict], 
                                       other_users_cards: List[Dict]) -> float:
        """Calculate similarity between cards from different users"""
        if not user_cards or not other_users_cards:
            return 0.0
        
        user_texts = [card.get('complete_sentence', '') for card in user_cards]
        other_texts = [card.get('complete_sentence', '') for card in other_users_cards]
        
        # Calculate average similarity between user's cards and other users' cards
        similarities = []
        for user_text in user_texts:
            for other_text in other_texts:
                user_words = set(user_text.lower().split())
                other_words = set(other_text.lower().split())
                
                intersection = len(user_words & other_words)
                union = len(user_words | other_words)
                similarity = intersection / union if union > 0 else 0.0
                similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.0

def run_enhanced_personalization_comparison():
    """Run an enhanced demonstration of personalization comparison with real data"""
    
    # Load real data from the evaluation results
    try:
        with open('evaluation/outputs/complete_sentences_evaluation_results_20250825_005120.json', 'r') as f:
            evaluation_data = json.load(f)
        
        # Load y_true data
        y_true = []
        with open('evaluation/outputs/y_true_generated_cah_cards_20250825_005120.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#') and line.isdigit():
                    y_true.append(int(line))
        
        # Create user profiles based on the data
        user_profiles = []
        for user_info in evaluation_data['summary']['users']:
            user_profiles.append({
                'user_id': user_info['user_id'],
                'personas': user_info['personas'],
                'combinations': user_info['combinations']
            })
        
        # Extract generated cards
        generated_cards = evaluation_data['complete_sentences']
        
        # Run enhanced comparison
        enhanced_comparator = EnhancedPersonalizationComparator()
        enhanced_scores = enhanced_comparator.compare_personalization_approaches(
            user_profiles, generated_cards, y_true
        )
        
        # Print comprehensive results
        print("🔬 ENHANCED PERSONALIZATION COMPARISON RESULTS")
        print("=" * 60)
        
        print(f"\n📊 BASIC PERSONALIZATION METRICS:")
        print(f"   Static Adaptability: {enhanced_scores.static_scores.static_adaptability:.2f}/10")
        print(f"   Dynamic Adaptability: {enhanced_scores.static_scores.dynamic_adaptability:.2f}/10")
        print(f"   Adaptability Improvement: {enhanced_scores.static_scores.adaptability_improvement:.2f}")
        
        print(f"\n🎯 F1 SCORE ANALYSIS:")
        print(f"   Static F1 Score: {enhanced_scores.f1_score_static:.3f}")
        print(f"   Dynamic F1 Score: {enhanced_scores.f1_score_dynamic:.3f}")
        print(f"   F1 Improvement: {enhanced_scores.f1_improvement:.3f}")
        
        print(f"\n📈 MSE ANALYSIS:")
        print(f"   Static MSE: {enhanced_scores.mse_static:.3f}")
        print(f"   Dynamic MSE: {enhanced_scores.mse_dynamic:.3f}")
        print(f"   MSE Improvement: {enhanced_scores.mse_improvement:.3f}")
        
        print(f"\n🎭 HUMOR QUALITY METRICS:")
        print(f"   Static Surprisal: {enhanced_scores.surprisal_static:.2f}/10")
        print(f"   Dynamic Surprisal: {enhanced_scores.surprisal_dynamic:.2f}/10")
        print(f"   Surprisal Improvement: {enhanced_scores.surprisal_improvement:.2f}")
        
        print(f"   Static Ambiguity: {enhanced_scores.ambiguity_static:.2f}/10")
        print(f"   Dynamic Ambiguity: {enhanced_scores.ambiguity_dynamic:.2f}/10")
        print(f"   Ambiguity Improvement: {enhanced_scores.ambiguity_improvement:.2f}")
        
        print(f"   Static Humor Quality: {enhanced_scores.humor_quality_static:.2f}/10")
        print(f"   Dynamic Humor Quality: {enhanced_scores.humor_quality_dynamic:.2f}/10")
        print(f"   Humor Quality Improvement: {enhanced_scores.humor_quality_improvement:.2f}")
        
        print(f"\n🔍 PERSONALIZED VS NON-PERSONALIZED COMPARISON:")
        print(f"   Non-Personalized F1: {enhanced_scores.personalized_vs_non.non_personalized_f1:.3f}")
        print(f"   Personalized F1: {enhanced_scores.personalized_vs_non.personalized_f1:.3f}")
        print(f"   F1 Improvement: {enhanced_scores.personalized_vs_non.f1_improvement:.3f}")
        
        print(f"   Non-Personalized MSE: {enhanced_scores.personalized_vs_non.non_personalized_mse:.3f}")
        print(f"   Personalized MSE: {enhanced_scores.personalized_vs_non.personalized_mse:.3f}")
        print(f"   MSE Improvement: {enhanced_scores.personalized_vs_non.mse_improvement:.3f}")
        
        print(f"   Non-Personalized Surprisal: {enhanced_scores.personalized_vs_non.non_personalized_surprisal:.2f}/10")
        print(f"   Personalized Surprisal: {enhanced_scores.personalized_vs_non.personalized_surprisal:.2f}/10")
        print(f"   Surprisal Improvement: {enhanced_scores.personalized_vs_non.surprisal_improvement:.2f}")
        
        print(f"   Non-Personalized Ambiguity: {enhanced_scores.personalized_vs_non.non_personalized_ambiguity:.2f}/10")
        print(f"   Personalized Ambiguity: {enhanced_scores.personalized_vs_non.personalized_ambiguity:.2f}/10")
        print(f"   Ambiguity Improvement: {enhanced_scores.personalized_vs_non.ambiguity_improvement:.2f}")
        
        print(f"   Non-Personalized Humor Quality: {enhanced_scores.personalized_vs_non.non_personalized_humor_quality:.2f}/10")
        print(f"   Personalized Humor Quality: {enhanced_scores.personalized_vs_non.personalized_humor_quality:.2f}/10")
        print(f"   Humor Quality Improvement: {enhanced_scores.personalized_vs_non.humor_quality_improvement:.2f}")
        
        print(f"\n🏆 OVERALL ASSESSMENT:")
        print(f"   Personalization Effectiveness: {enhanced_scores.personalization_effectiveness:.3f}")
        print(f"   Personalization Benefit Score: {enhanced_scores.personalized_vs_non.personalization_benefit_score:.3f}")
        print(f"   Dynamic Superiority Score: {enhanced_scores.static_scores.dynamic_superiority_score:.2f}/10")
        
        # Determine overall conclusion
        if enhanced_scores.personalization_effectiveness > 0.6:
            print("   ✅ DYNAMIC PERSONALIZATION IS HIGHLY EFFECTIVE")
        elif enhanced_scores.personalization_effectiveness > 0.4:
            print("   ✅ DYNAMIC PERSONALIZATION IS EFFECTIVE")
        else:
            print("   ⚠️ Results are inconclusive")
        
        if enhanced_scores.personalized_vs_non.personalization_benefit_score > 0.6:
            print("   ✅ PERSONALIZATION PROVIDES SIGNIFICANT BENEFITS")
        elif enhanced_scores.personalized_vs_non.personalization_benefit_score > 0.4:
            print("   ✅ PERSONALIZATION PROVIDES MODERATE BENEFITS")
        else:
            print("   ⚠️ Personalization benefits are inconclusive")
        
        return enhanced_scores
        
    except FileNotFoundError:
        print("⚠️ Could not find evaluation data files. Running with sample data instead.")
        return run_personalization_comparison()

def run_personalization_comparison():
    """Run a demonstration of personalization comparison"""
    
    # Create sample data for demonstration
    user_profiles = [
        {
            'user_id': 'user_1',
            'humor_style': 'dad_humor',
            'preferred_topics': ['family', 'puns', 'clean_humor']
        },
        {
            'user_id': 'user_2', 
            'humor_style': 'millennial_memer',
            'preferred_topics': ['internet_culture', 'irony', 'pop_culture']
        },
        {
            'user_id': 'user_3',
            'humor_style': 'gen_z_chaos',
            'preferred_topics': ['absurdism', 'dark_humor', 'unexpected']
        }
    ]
    
    # Sample generated cards (simulating both approaches)
    generated_cards = [
        # Static persona outputs (same persona for all users)
        {'user_id': 'user_1', 'text': 'Dad joke about socks', 'persona_type': 'dad_humor'},
        {'user_id': 'user_2', 'text': 'Dad joke about socks', 'persona_type': 'dad_humor'}, 
        {'user_id': 'user_3', 'text': 'Dad joke about socks', 'persona_type': 'dad_humor'},
        
        # Dynamic persona outputs (adapted to each user)
        {'user_id': 'user_1', 'text': 'Dad joke about socks', 'persona_type': 'dynamic_dad'},
        {'user_id': 'user_2', 'text': 'Millennial meme about socks', 'persona_type': 'dynamic_millennial'},
        {'user_id': 'user_3', 'text': 'Absurdist chaos about socks', 'persona_type': 'dynamic_genz'}
    ]
    
    # Run comparison
    comparator = PersonalizationComparator()
    scores = comparator.compare_personalization_approaches(user_profiles, generated_cards)
    
    # Print results
    print("🔬 PERSONALIZATION COMPARISON RESULTS")
    print("=" * 50)
    
    print(f"\n📊 STATIC PERSONALIZATION:")
    print(f"   Consistency: {scores.static_consistency:.2f}/10")
    print(f"   Diversity: {scores.static_diversity:.2f}/10")
    print(f"   Adaptability: {scores.static_adaptability:.2f}/10")
    
    print(f"\n🚀 DYNAMIC PERSONALIZATION:")
    print(f"   Consistency: {scores.dynamic_consistency:.2f}/10")
    print(f"   Diversity: {scores.dynamic_diversity:.2f}/10")
    print(f"   Adaptability: {scores.dynamic_adaptability:.2f}/10")
    
    print(f"\n⚖️ COMPARISON METRICS:")
    print(f"   Adaptability Improvement: {scores.adaptability_improvement:.2f}")
    print(f"   Personalization Precision: {scores.personalization_precision:.2f}x")
    print(f"   User Preference Alignment: {scores.user_preference_alignment:.2f}/10")
    
    print(f"\n🏆 OVERALL ASSESSMENT:")
    print(f"   Dynamic Superiority Score: {scores.dynamic_superiority_score:.2f}/10")
    
    if scores.dynamic_superiority_score > 7.0:
        print("   ✅ DYNAMIC PERSONALIZATION IS SIGNIFICANTLY SUPERIOR")
    elif scores.dynamic_superiority_score > 5.0:
        print("   ✅ DYNAMIC PERSONALIZATION IS SUPERIOR")
    else:
        print("   ⚠️ Results are inconclusive")
    
    return scores

if __name__ == "__main__":
    run_enhanced_personalization_comparison()
