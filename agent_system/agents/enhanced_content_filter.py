#!/usr/bin/env python3
"""
Enhanced Content Filtering
Implements advanced content filtering including Perspective API as discussed in literature
Literature: CleanComedy (vikhorev2024cleancomedy) - "using Detoxify and curated corpus"
"""

import asyncio
import json
import re
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import requests
import time

class ToxicityCategory(Enum):
    """Categories of toxicity to detect"""
    TOXICITY = "TOXICITY"
    SEVERE_TOXICITY = "SEVERE_TOXICITY"
    IDENTITY_ATTACK = "IDENTITY_ATTACK"
    INSULT = "INSULT"
    PROFANITY = "PROFANITY"
    THREAT = "THREAT"
    SEXUALLY_EXPLICIT = "SEXUALLY_EXPLICIT"
    FLIRTATION = "FLIRTATION"

@dataclass
class ContentFilterResult:
    """Result from content filtering"""
    is_safe: bool
    overall_toxicity: float
    category_scores: Dict[str, float]
    flagged_categories: List[str]
    confidence: float
    suggestions: List[str]
    sanitized_content: Optional[str] = None

@dataclass
class FilterThresholds:
    """Configurable thresholds for different toxicity categories"""
    toxicity: float = 0.7
    severe_toxicity: float = 0.5
    identity_attack: float = 0.6
    insult: float = 0.7
    profanity: float = 0.8
    threat: float = 0.3
    sexually_explicit: float = 0.6
    flirtation: float = 0.9

class EnhancedContentFilter:
    """
    Enhanced content filtering system with multiple detection methods
    Literature: "CleanComedy used Detoxify and curated corpus of non-toxic jokes"
    """
    
    def __init__(self, perspective_api_key: Optional[str] = None):
        self.perspective_api_key = perspective_api_key
        self.perspective_url = "https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze"
        
        # Fallback rules-based filtering
        self.toxic_patterns = {
            "hate_speech": [
                r"\b(hate|despise|loathe)\s+(all|every)\s+\w+",
                r"\b(kill|murder|die)\s+(all|every)\s+\w+",
                r"\b\w+\s+(are|is)\s+(evil|scum|trash|garbage)"
            ],
            "profanity": [
                r"\bf[*]?u[*]?c[*]?k\w*",
                r"\bs[*]?h[*]?i[*]?t\w*", 
                r"\bd[*]?a[*]?m[*]?n\w*",
                r"\bb[*]?i[*]?t[*]?c[*]?h\w*"
            ],
            "sexual_explicit": [
                r"\b(sex|sexual|porn|nude|naked)\s+(scene|act|content)",
                r"\b(penis|vagina|breast|genitals)\b",
                r"\bmasturbat\w*"
            ],
            "violence": [
                r"\b(kill|murder|stab|shoot|bomb)\s+\w+",
                r"\b(blood|gore|violence|torture)\b",
                r"\b(weapon|gun|knife|explosive)\b"
            ]
        }
        
        # Content sanitization rules
        self.sanitization_rules = {
            # Profanity replacements
            r"\bf[*]?u[*]?c[*]?k": "fudge",
            r"\bs[*]?h[*]?i[*]?t": "shoot",
            r"\bd[*]?a[*]?m[*]?n": "darn", 
            r"\bb[*]?i[*]?t[*]?c[*]?h": "person",
            r"\bcrap": "nonsense",
            r"\bsucks": "is unfortunate",
            r"\bstupid": "silly",
            r"\bidiot": "person",
            
            # Aggressive language
            r"\bhate": "dislike",
            r"\bkill": "defeat",
            r"\bmurder": "eliminate",
            r"\bdestroy": "fix",
            
            # Inappropriate references
            r"\bsexy": "attractive",
            r"\bhot": "appealing",
            r"\bcrazy": "unusual"
        }
        
        # Safe humor categories
        self.safe_humor_keywords = {
            "family_friendly": ["silly", "wholesome", "innocent", "sweet", "charming"],
            "observational": ["everyday", "relatable", "common", "typical", "ordinary"],
            "wordplay": ["pun", "clever", "witty", "smart", "linguistic"],
            "absurd": ["unexpected", "surreal", "bizarre", "unusual", "quirky"],
            "self_deprecating": ["myself", "my own", "personal", "admit", "confess"]
        }
    
    async def analyze_content_perspective_api(self, text: str) -> Dict[str, float]:
        """
        Analyze content using Perspective API
        Returns toxicity scores for different categories
        """
        if not self.perspective_api_key:
            # Fallback to rules-based analysis
            return await self._rules_based_analysis(text)
        
        try:
            # Prepare request data
            data = {
                'comment': {'text': text},
                'requestedAttributes': {
                    category.value: {} for category in ToxicityCategory
                }
            }
            
            # Make API request
            response = requests.post(
                f"{self.perspective_url}?key={self.perspective_api_key}",
                json=data,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                scores = {}
                
                for category in ToxicityCategory:
                    if category.value in result.get('attributeScores', {}):
                        score_data = result['attributeScores'][category.value]
                        scores[category.value] = score_data['summaryScore']['value']
                    else:
                        scores[category.value] = 0.0
                
                return scores
            else:
                print(f"Perspective API error: {response.status_code}")
                return await self._rules_based_analysis(text)
                
        except Exception as e:
            print(f"Perspective API request failed: {e}")
            return await self._rules_based_analysis(text)
    
    async def _rules_based_analysis(self, text: str) -> Dict[str, float]:
        """Fallback rules-based toxicity analysis"""
        text_lower = text.lower()
        scores = {}
        
        for category in ToxicityCategory:
            category_name = category.value.lower()
            score = 0.0
            
            # Check against pattern rules
            if category_name in ["toxicity", "severe_toxicity"]:
                # General toxicity check
                for pattern_category, patterns in self.toxic_patterns.items():
                    for pattern in patterns:
                        if re.search(pattern, text_lower, re.IGNORECASE):
                            score = max(score, 0.8 if pattern_category == "hate_speech" else 0.6)
            
            elif "profanity" in category_name:
                for pattern in self.toxic_patterns.get("profanity", []):
                    if re.search(pattern, text_lower, re.IGNORECASE):
                        score = max(score, 0.7)
            
            elif "sexually_explicit" in category_name:
                for pattern in self.toxic_patterns.get("sexual_explicit", []):
                    if re.search(pattern, text_lower, re.IGNORECASE):
                        score = max(score, 0.8)
            
            elif "threat" in category_name:
                for pattern in self.toxic_patterns.get("violence", []):
                    if re.search(pattern, text_lower, re.IGNORECASE):
                        score = max(score, 0.9)
            
            scores[category.value] = score
        
        return scores
    
    async def enhanced_filter(
        self, 
        text: str, 
        thresholds: FilterThresholds = None,
        require_sanitization: bool = True
    ) -> ContentFilterResult:
        """
        Comprehensive content filtering with multiple methods
        
        Process:
        1. Analyze with Perspective API (or rules)
        2. Apply category-specific thresholds
        3. Generate safety assessment
        4. Provide sanitization suggestions
        """
        
        if thresholds is None:
            thresholds = FilterThresholds()
        
        # Step 1: Get toxicity scores
        category_scores = await self.analyze_content_perspective_api(text)
        
        # Step 2: Check against thresholds
        flagged_categories = []
        threshold_map = {
            "TOXICITY": thresholds.toxicity,
            "SEVERE_TOXICITY": thresholds.severe_toxicity,
            "IDENTITY_ATTACK": thresholds.identity_attack,
            "INSULT": thresholds.insult,
            "PROFANITY": thresholds.profanity,
            "THREAT": thresholds.threat,
            "SEXUALLY_EXPLICIT": thresholds.sexually_explicit,
            "FLIRTATION": thresholds.flirtation
        }
        
        for category, score in category_scores.items():
            threshold = threshold_map.get(category, 0.7)
            if score > threshold:
                flagged_categories.append(category)
        
        # Step 3: Calculate overall safety
        overall_toxicity = max(category_scores.values()) if category_scores else 0.0
        is_safe = len(flagged_categories) == 0
        
        # Calculate confidence based on score distribution
        confidence = self._calculate_confidence(category_scores)
        
        # Step 4: Generate suggestions and sanitization
        suggestions = self._generate_suggestions(flagged_categories, category_scores)
        sanitized_content = None
        
        if require_sanitization and not is_safe:
            sanitized_content = await self._sanitize_content(text, flagged_categories)
        
        return ContentFilterResult(
            is_safe=is_safe,
            overall_toxicity=overall_toxicity,
            category_scores=category_scores,
            flagged_categories=flagged_categories,
            confidence=confidence,
            suggestions=suggestions,
            sanitized_content=sanitized_content
        )
    
    def _calculate_confidence(self, scores: Dict[str, float]) -> float:
        """Calculate confidence in the filtering decision"""
        if not scores:
            return 0.5
        
        score_values = list(scores.values())
        
        # High confidence if scores are either very low or very high
        max_score = max(score_values)
        min_score = min(score_values)
        
        if max_score > 0.8 or max_score < 0.2:
            confidence = 0.9
        elif max_score > 0.6 or max_score < 0.4:
            confidence = 0.7
        else:
            confidence = 0.5
        
        # Adjust based on score consistency
        score_range = max_score - min_score
        if score_range < 0.2:
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def _generate_suggestions(self, flagged_categories: List[str], scores: Dict[str, float]) -> List[str]:
        """Generate content improvement suggestions"""
        suggestions = []
        
        if "PROFANITY" in flagged_categories:
            suggestions.append("Consider replacing profanity with milder alternatives")
        
        if "TOXICITY" in flagged_categories or "SEVERE_TOXICITY" in flagged_categories:
            suggestions.append("Content contains toxic language - consider rephrasing")
        
        if "INSULT" in flagged_categories:
            suggestions.append("Avoid insulting language, even in jest")
        
        if "THREAT" in flagged_categories:
            suggestions.append("Remove threatening language")
        
        if "SEXUALLY_EXPLICIT" in flagged_categories:
            suggestions.append("Content may be too sexually explicit for general audiences")
        
        if "IDENTITY_ATTACK" in flagged_categories:
            suggestions.append("Avoid content that attacks specific identity groups")
        
        # Positive suggestions
        if not flagged_categories:
            suggestions.append("Content appears safe and appropriate")
        else:
            suggestions.append("Focus on observational humor, wordplay, or self-deprecating jokes")
            suggestions.append("Consider family-friendly alternatives that maintain the humor")
        
        return suggestions
    
    async def _sanitize_content(self, text: str, flagged_categories: List[str]) -> str:
        """Sanitize content by applying replacement rules"""
        sanitized = text
        
        # Apply sanitization rules
        for pattern, replacement in self.sanitization_rules.items():
            sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)
        
        # Additional category-specific sanitization
        if "THREAT" in flagged_categories:
            # Remove violent language
            threat_patterns = [
                (r"\bkill\b", "defeat"),
                (r"\bmurder\b", "eliminate"),
                (r"\bdestroy\b", "fix"),
                (r"\battack\b", "approach")
            ]
            
            for pattern, replacement in threat_patterns:
                sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)
        
        if "SEXUALLY_EXPLICIT" in flagged_categories:
            # Remove sexual content
            sexual_patterns = [
                (r"\bsexy?\b", "attractive"),
                (r"\bhot\b", "appealing"),
                (r"\bnaked\b", "exposed"),
                (r"\bnude\b", "bare")
            ]
            
            for pattern, replacement in sexual_patterns:
                sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)
        
        return sanitized
    
    def suggest_safe_alternatives(self, original_text: str, context: str = "general") -> List[str]:
        """
        Suggest safe humor alternatives based on context
        Literature: "CleanComedy found cleaned dataset produced higher-rated humor"
        """
        alternatives = []
        
        # Extract topic/theme from original text
        topic_keywords = ["work", "family", "food", "technology", "life", "adult", "child"]
        detected_topic = "general"
        
        original_lower = original_text.lower()
        for topic in topic_keywords:
            if topic in original_lower:
                detected_topic = topic
                break
        
        # Generate safe alternatives based on topic
        if detected_topic == "work":
            alternatives = [
                "Something about meetings that goes on forever",
                "The universal truth about Monday mornings",
                "Why coffee is actually a food group at work",
                "The mystery of office printers and their timing"
            ]
        elif detected_topic == "family":
            alternatives = [
                "Something relatable about family dinners",
                "The universal experience of explaining technology to relatives",
                "Why family photos always have one person blinking",
                "The genetic programming of dad jokes"
            ]
        elif detected_topic == "food":
            alternatives = [
                "The eternal struggle of healthy eating",
                "Why vegetables are expensive and fast food isn't",
                "The optimism of buying bananas",
                "Cooking shows vs. actual cooking reality"
            ]
        else:
            alternatives = [
                "Something unexpectedly relatable about everyday life",
                "The uncomfortable truth everyone thinks but doesn't say",
                "A surprisingly accurate observation about modern life",
                "The kind of thing that makes you laugh then nod in agreement"
            ]
        
        return alternatives[:3]  # Return top 3 alternatives
    
    def get_safety_stats(self) -> Dict[str, Any]:
        """Get statistics about content safety filtering"""
        return {
            "filter_categories": [cat.value for cat in ToxicityCategory],
            "sanitization_rules": len(self.sanitization_rules),
            "toxic_patterns": {k: len(v) for k, v in self.toxic_patterns.items()},
            "safe_humor_categories": list(self.safe_humor_keywords.keys()),
            "perspective_api_enabled": self.perspective_api_key is not None,
            "default_thresholds": {
                "toxicity": 0.7,
                "severe_toxicity": 0.5,
                "threat": 0.3,
                "profanity": 0.8
            }
        }

# Global instance
enhanced_content_filter = EnhancedContentFilter() 