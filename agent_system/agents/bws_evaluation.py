#!/usr/bin/env python3
"""
Best-Worst Scaling (BWS) Evaluation System
Implements BWS as discussed in Horvitz et al. for more robust humor evaluation
"""

import random
import math
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime

@dataclass
class BWS_Item:
    """Individual item in BWS evaluation"""
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BWS_Comparison:
    """Single BWS comparison (4-tuple)"""
    comparison_id: str
    items: List[BWS_Item]  # Should be 4 items
    best_item_id: Optional[str] = None
    worst_item_id: Optional[str] = None
    user_id: Optional[str] = None
    timestamp: Optional[datetime] = None

@dataclass
class BWS_Results:
    """Results from BWS analysis"""
    item_scores: Dict[str, float]  # Item ID -> BWS score
    item_rankings: List[Tuple[str, float]]  # Sorted by score
    total_comparisons: int
    confidence_intervals: Dict[str, Tuple[float, float]]

class BestWorstScalingEvaluator:
    """
    Best-Worst Scaling evaluation for humor quality
    
    Literature: "BWS compares each response in several mini-contests" - Horvitz
    More robust than Likert scales with fewer judgments needed
    """
    
    def __init__(self):
        self.comparisons: List[BWS_Comparison] = []
        self.items: Dict[str, BWS_Item] = {}
        
    def add_items(self, items: List[BWS_Item]):
        """Add items to be evaluated"""
        for item in items:
            self.items[item.id] = item
    
    def generate_comparisons(self, n_comparisons: int = None) -> List[BWS_Comparison]:
        """
        Generate balanced BWS comparison sets
        Each comparison contains 4 items for best/worst selection
        """
        if len(self.items) < 4:
            raise ValueError("Need at least 4 items for BWS evaluation")
        
        item_list = list(self.items.values())
        
        # If not specified, generate enough comparisons for statistical power
        if n_comparisons is None:
            # Rule of thumb: 3-5 comparisons per item
            n_comparisons = max(len(item_list), 12)
        
        comparisons = []
        
        # Generate random 4-tuples with balanced appearance
        item_appearances = defaultdict(int)
        
        for i in range(n_comparisons):
            # Select 4 items, trying to balance appearances
            available_items = [item for item in item_list 
                             if item_appearances[item.id] < (n_comparisons // 2)]
            
            if len(available_items) < 4:
                available_items = item_list  # Reset if needed
            
            selected_items = random.sample(available_items, 4)
            
            # Update appearance counts
            for item in selected_items:
                item_appearances[item.id] += 1
            
            comparison = BWS_Comparison(
                comparison_id=f"bws_{i+1}",
                items=selected_items
            )
            comparisons.append(comparison)
        
        self.comparisons.extend(comparisons)
        return comparisons
    
    def record_judgment(
        self, 
        comparison_id: str, 
        best_item_id: str, 
        worst_item_id: str,
        user_id: str = None
    ):
        """Record a human judgment for a BWS comparison"""
        for comparison in self.comparisons:
            if comparison.comparison_id == comparison_id:
                comparison.best_item_id = best_item_id
                comparison.worst_item_id = worst_item_id
                comparison.user_id = user_id
                comparison.timestamp = datetime.now()
                break
        else:
            raise ValueError(f"Comparison {comparison_id} not found")
    
    def calculate_bws_scores(self) -> BWS_Results:
        """
        Calculate BWS scores from collected judgments
        
        BWS Score = (# times chosen as best - # times chosen as worst) / # appearances
        Range: [-1, +1] where +1 = always best, -1 = always worst
        """
        # Count best/worst selections and appearances
        best_counts = defaultdict(int)
        worst_counts = defaultdict(int)
        appearance_counts = defaultdict(int)
        
        completed_comparisons = 0
        
        for comparison in self.comparisons:
            if comparison.best_item_id and comparison.worst_item_id:
                completed_comparisons += 1
                
                # Count appearances
                for item in comparison.items:
                    appearance_counts[item.id] += 1
                
                # Count best/worst selections
                best_counts[comparison.best_item_id] += 1
                worst_counts[comparison.worst_item_id] += 1
        
        # Calculate BWS scores
        item_scores = {}
        for item_id in self.items:
            if appearance_counts[item_id] > 0:
                bws_score = (best_counts[item_id] - worst_counts[item_id]) / appearance_counts[item_id]
                item_scores[item_id] = bws_score
            else:
                item_scores[item_id] = 0.0
        
        # Sort by score (descending)
        item_rankings = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Calculate confidence intervals (simplified)
        confidence_intervals = {}
        for item_id, score in item_scores.items():
            n_appearances = appearance_counts[item_id]
            if n_appearances > 0:
                # Simple confidence interval based on binomial distribution
                std_error = math.sqrt((1 - score**2) / n_appearances)
                margin = 1.96 * std_error  # 95% confidence
                confidence_intervals[item_id] = (
                    max(-1.0, score - margin),
                    min(1.0, score + margin)
                )
            else:
                confidence_intervals[item_id] = (-1.0, 1.0)
        
        return BWS_Results(
            item_scores=item_scores,
            item_rankings=item_rankings,
            total_comparisons=completed_comparisons,
            confidence_intervals=confidence_intervals
        )
    
    def get_comparison_for_user(self, user_id: str = None) -> Optional[BWS_Comparison]:
        """Get next unevaluated comparison for a user"""
        for comparison in self.comparisons:
            if not comparison.best_item_id:  # Not yet evaluated
                return comparison
        return None
    
    def convert_to_likert_equivalent(self, bws_scores: Dict[str, float]) -> Dict[str, float]:
        """
        Convert BWS scores to Likert-like scale (1-10) for comparison
        BWS [-1, +1] -> Likert [1, 10]
        """
        likert_scores = {}
        for item_id, bws_score in bws_scores.items():
            # Linear transformation: [-1, 1] -> [1, 10]
            likert_score = 4.5 * (bws_score + 1) + 1
            likert_scores[item_id] = round(likert_score, 1)
        
        return likert_scores
    
    def compare_with_likert(self, likert_scores: Dict[str, float]) -> Dict[str, Any]:
        """
        Compare BWS results with Likert scale ratings
        Literature shows BWS often more reliable with fewer judgments
        """
        bws_results = self.calculate_bws_scores()
        bws_likert_equivalent = self.convert_to_likert_equivalent(bws_results.item_scores)
        
        # Calculate correlation if both have scores
        common_items = set(bws_likert_equivalent.keys()) & set(likert_scores.keys())
        
        if len(common_items) < 2:
            return {"error": "Not enough common items for comparison"}
        
        # Calculate Pearson correlation
        bws_values = [bws_likert_equivalent[item_id] for item_id in common_items]
        likert_values = [likert_scores[item_id] for item_id in common_items]
        
        correlation = self._calculate_correlation(bws_values, likert_values)
        
        # Calculate ranking correlation (Spearman)
        bws_rankings = {item_id: rank for rank, (item_id, _) in enumerate(bws_results.item_rankings)}
        likert_rankings = {item_id: rank for rank, (item_id, _) in 
                          enumerate(sorted(likert_scores.items(), key=lambda x: x[1], reverse=True))}
        
        rank_correlation = self._calculate_rank_correlation(bws_rankings, likert_rankings, common_items)
        
        return {
            "pearson_correlation": correlation,
            "spearman_correlation": rank_correlation,
            "bws_scores": bws_likert_equivalent,
            "likert_scores": likert_scores,
            "common_items": len(common_items),
            "bws_comparisons": bws_results.total_comparisons
        }
    
    def _calculate_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate Pearson correlation coefficient"""
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi * xi for xi in x)
        sum_y2 = sum(yi * yi for yi in y)
        
        numerator = n * sum_xy - sum_x * sum_y
        denominator = math.sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y))
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def _calculate_rank_correlation(
        self, 
        rankings1: Dict[str, int], 
        rankings2: Dict[str, int], 
        common_items: set
    ) -> float:
        """Calculate Spearman rank correlation"""
        if len(common_items) < 2:
            return 0.0
        
        rank_diffs = []
        for item_id in common_items:
            diff = rankings1[item_id] - rankings2[item_id]
            rank_diffs.append(diff * diff)
        
        n = len(common_items)
        if n < 2:
            return 0.0
        
        # Spearman correlation formula
        sum_d2 = sum(rank_diffs)
        rho = 1 - (6 * sum_d2) / (n * (n * n - 1))
        
        return rho
    
    def generate_evaluation_summary(self) -> Dict[str, Any]:
        """Generate comprehensive evaluation summary"""
        bws_results = self.calculate_bws_scores()
        
        # Calculate statistics
        scores = list(bws_results.item_scores.values())
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        top_items = bws_results.item_rankings[:3]
        bottom_items = bws_results.item_rankings[-3:]
        
        return {
            "total_items": len(self.items),
            "total_comparisons": len(self.comparisons),
            "completed_comparisons": bws_results.total_comparisons,
            "completion_rate": bws_results.total_comparisons / len(self.comparisons) if self.comparisons else 0.0,
            "average_bws_score": avg_score,
            "score_range": (min(scores), max(scores)) if scores else (0.0, 0.0),
            "top_items": [(self.items[item_id].text[:50], score) for item_id, score in top_items],
            "bottom_items": [(self.items[item_id].text[:50], score) for item_id, score in bottom_items],
            "statistical_power": "Good" if bws_results.total_comparisons >= len(self.items) * 3 else "Limited"
        }

# Global instance for easy access
bws_evaluator = BestWorstScalingEvaluator() 