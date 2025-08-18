#!/usr/bin/env python3
"""
Controlled Humor Generation
Implements PPLM-style controlled generation as discussed in literature (Tian et al.)
"""

import asyncio
import random
import math
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum

class ControlAttribute(Enum):
    """Attributes that can be controlled during generation"""
    HUMOR = "humor"
    APPROPRIATENESS = "appropriateness" 
    CREATIVITY = "creativity"
    SURPRISE = "surprise"
    SAFETY = "safety"
    STYLE_WITTY = "witty"
    STYLE_SARCASTIC = "sarcastic"
    STYLE_WHOLESOME = "wholesome"

@dataclass
class ControlVector:
    """Control vector for steering generation"""
    attribute: ControlAttribute
    strength: float  # -1.0 to 1.0
    weight: float    # 0.0 to 1.0

@dataclass
class GenerationConstraints:
    """Constraints for controlled generation"""
    max_length: int = 100
    min_humor_score: float = 0.6
    max_toxicity: float = 0.3
    required_style: Optional[str] = None
    forbidden_words: List[str] = None
    required_sentiment: Optional[str] = None  # positive, negative, neutral

class ControlledHumorGenerator:
    """
    PPLM-style controlled generation for humor
    Literature: Tian et al. - "PPLM uses gradient-based guidance at each stage"
    """
    
    def __init__(self):
        # Attribute classifiers (simplified - would be trained models)
        self.humor_keywords = {
            "high": ["hilarious", "funny", "ridiculous", "absurd", "unexpected", "ironic"],
            "medium": ["amusing", "clever", "witty", "silly", "quirky"],
            "low": ["boring", "predictable", "obvious", "standard"]
        }
        
        self.appropriateness_keywords = {
            "safe": ["family-friendly", "wholesome", "clean", "appropriate", "polite"],
            "edgy": ["inappropriate", "shocking", "controversial", "bold"],
            "unsafe": ["offensive", "disturbing", "explicit", "harmful"]
        }
        
        self.style_keywords = {
            "witty": ["clever", "sharp", "intelligent", "sophisticated"],
            "sarcastic": ["ironic", "mocking", "sardonic", "dry"],
            "wholesome": ["sweet", "innocent", "pure", "heartwarming"]
        }
        
        # Control templates for steering
        self.control_templates = {
            ControlAttribute.HUMOR: {
                "enhance": "Make this funnier by adding {enhancement}",
                "suppress": "Make this less funny by {suppression}",
                "keywords": ["unexpected twist", "ironic observation", "absurd comparison"]
            },
            ControlAttribute.APPROPRIATENESS: {
                "enhance": "Make this more appropriate by {enhancement}",
                "suppress": "Make this edgier by {suppression}",
                "keywords": ["family-friendly language", "wholesome content", "clean humor"]
            },
            ControlAttribute.CREATIVITY: {
                "enhance": "Make this more creative by {enhancement}",
                "suppress": "Make this more conventional by {suppression}", 
                "keywords": ["original perspective", "unique angle", "fresh take"]
            }
        }
    
    def calculate_attribute_score(self, text: str, attribute: ControlAttribute) -> float:
        """
        Calculate score for a given attribute (simplified classifier)
        In practice, would use trained neural classifiers
        """
        text_lower = text.lower()
        
        if attribute == ControlAttribute.HUMOR:
            high_count = sum(1 for word in self.humor_keywords["high"] if word in text_lower)
            medium_count = sum(1 for word in self.humor_keywords["medium"] if word in text_lower)
            low_count = sum(1 for word in self.humor_keywords["low"] if word in text_lower)
            
            score = (high_count * 1.0 + medium_count * 0.6 - low_count * 0.3)
            return max(0.0, min(1.0, score / 3.0 + 0.5))
            
        elif attribute == ControlAttribute.APPROPRIATENESS:
            safe_count = sum(1 for word in self.appropriateness_keywords["safe"] if word in text_lower)
            edgy_count = sum(1 for word in self.appropriateness_keywords["edgy"] if word in text_lower)
            unsafe_count = sum(1 for word in self.appropriateness_keywords["unsafe"] if word in text_lower)
            
            score = (safe_count * 1.0 - edgy_count * 0.3 - unsafe_count * 1.0)
            return max(0.0, min(1.0, score / 2.0 + 0.7))
            
        elif attribute == ControlAttribute.SURPRISE:
            # Measure unexpectedness through uncommon words/phrases
            uncommon_patterns = ["quantum", "existential", "metaphysical", "paradox", "irony"]
            surprise_count = sum(1 for pattern in uncommon_patterns if pattern in text_lower)
            return min(1.0, surprise_count / 2.0 + 0.3)
            
        elif attribute in [ControlAttribute.STYLE_WITTY, ControlAttribute.STYLE_SARCASTIC, ControlAttribute.STYLE_WHOLESOME]:
            style_name = attribute.value.replace("style_", "")
            if style_name in self.style_keywords:
                style_count = sum(1 for word in self.style_keywords[style_name] if word in text_lower)
                return min(1.0, style_count / 2.0 + 0.4)
        
        return 0.5  # Default neutral score
    
    async def apply_control_vectors(
        self, 
        base_text: str, 
        control_vectors: List[ControlVector]
    ) -> str:
        """
        Apply control vectors to steer generation
        Literature: "PPLM's attribute steering improved the fraction of 'funny' outputs"
        """
        current_text = base_text
        
        for vector in control_vectors:
            if vector.weight > 0.1:  # Only apply significant weights
                current_text = await self._apply_single_control(current_text, vector)
        
        return current_text
    
    async def _apply_single_control(self, text: str, control_vector: ControlVector) -> str:
        """Apply a single control vector to text"""
        attribute = control_vector.attribute
        strength = control_vector.strength
        
        # Get current score
        current_score = self.calculate_attribute_score(text, attribute)
        
        # Determine if we need to enhance or suppress
        target_score = current_score + (strength * 0.3)  # Adjust by up to 30%
        target_score = max(0.0, min(1.0, target_score))
        
        if target_score > current_score:
            # Need to enhance this attribute
            return await self._enhance_attribute(text, attribute, target_score - current_score)
        elif target_score < current_score:
            # Need to suppress this attribute
            return await self._suppress_attribute(text, attribute, current_score - target_score)
        
        return text
    
    async def _enhance_attribute(self, text: str, attribute: ControlAttribute, enhancement_needed: float) -> str:
        """Enhance specific attribute in text"""
        
        if attribute == ControlAttribute.HUMOR:
            # Add humor-enhancing elements
            humor_boosters = [
                "unexpectedly", "ironically", "surprisingly", "absurdly",
                "hilariously", "ridiculously", "comically"
            ]
            if enhancement_needed > 0.2:
                booster = random.choice(humor_boosters)
                # Insert booster at appropriate position
                words = text.split()
                if len(words) > 2:
                    insert_pos = len(words) // 2
                    words.insert(insert_pos, booster)
                    return " ".join(words)
        
        elif attribute == ControlAttribute.APPROPRIATENESS:
            # Make more family-friendly
            replacements = {
                "damn": "darn", "hell": "heck", "crap": "nonsense",
                "stupid": "silly", "idiot": "person", "sucks": "is unfortunate"
            }
            
            modified_text = text
            for inappropriate, appropriate in replacements.items():
                modified_text = modified_text.replace(inappropriate, appropriate)
            
            if modified_text != text:
                return modified_text
        
        elif attribute == ControlAttribute.CREATIVITY:
            # Add creative elements
            creative_connectors = [
                "like a", "as if", "reminiscent of", "comparable to", 
                "in the style of", "with the energy of"
            ]
            
            creative_objects = [
                "surreal art piece", "fever dream", "philosophical puzzle",
                "cosmic joke", "social experiment", "existential crisis"
            ]
            
            if enhancement_needed > 0.3:
                connector = random.choice(creative_connectors)
                obj = random.choice(creative_objects)
                return f"{text} {connector} {obj}"
        
        return text
    
    async def _suppress_attribute(self, text: str, attribute: ControlAttribute, suppression_needed: float) -> str:
        """Suppress specific attribute in text"""
        
        if attribute == ControlAttribute.HUMOR:
            # Remove humor-enhancing words
            humor_words = self.humor_keywords["high"] + self.humor_keywords["medium"]
            words = text.split()
            filtered_words = [word for word in words if word.lower() not in humor_words]
            
            if len(filtered_words) < len(words):
                return " ".join(filtered_words)
        
        elif attribute == ControlAttribute.SURPRISE:
            # Make more predictable
            words = text.split()
            # Remove unusual/surprising words
            common_words = ["the", "a", "an", "is", "was", "has", "have", "very", "really", "quite"]
            if len(words) > 3:
                # Replace one unusual word with common one
                for i, word in enumerate(words):
                    if len(word) > 6 and word.lower() not in common_words:
                        words[i] = "really"
                        break
                return " ".join(words)
        
        return text
    
    async def controlled_generate(
        self,
        prompt: str,
        control_vectors: List[ControlVector],
        constraints: GenerationConstraints = None
    ) -> Dict[str, Any]:
        """
        Generate humor with controlled attributes
        
        Process:
        1. Generate base response
        2. Apply control vectors iteratively
        3. Check constraints
        4. Refine if needed
        """
        
        if constraints is None:
            constraints = GenerationConstraints()
        
        # Step 1: Base generation (simplified)
        base_candidates = await self._generate_base_candidates(prompt)
        
        # Step 2: Apply controls to each candidate
        controlled_candidates = []
        for candidate in base_candidates:
            controlled_text = await self.apply_control_vectors(candidate, control_vectors)
            
            # Calculate scores for all attributes
            scores = {}
            for attr in ControlAttribute:
                scores[attr.value] = self.calculate_attribute_score(controlled_text, attr)
            
            controlled_candidates.append({
                "text": controlled_text,
                "original": candidate,
                "scores": scores,
                "meets_constraints": self._check_constraints(controlled_text, constraints)
            })
        
        # Step 3: Select best candidate
        best_candidate = self._select_best_candidate(controlled_candidates, control_vectors, constraints)
        
        return {
            "generated_text": best_candidate["text"],
            "original_text": best_candidate["original"],
            "attribute_scores": best_candidate["scores"],
            "control_vectors_applied": [{"attribute": cv.attribute.value, "strength": cv.strength, "weight": cv.weight} for cv in control_vectors],
            "meets_constraints": best_candidate["meets_constraints"],
            "generation_method": "controlled_generation",
            "candidates_considered": len(controlled_candidates)
        }
    
    async def _generate_base_candidates(self, prompt: str) -> List[str]:
        """Generate base candidates before control application"""
        # Simplified base generation - in practice would use LLM
        templates = [
            "Something hilariously {adjective} about {topic}",
            "The {emotion} truth about {topic} that everyone knows",
            "{topic} is basically just {comparison} with extra steps",
            "What they don't tell you about {topic} is {revelation}"
        ]
        
        # Extract topic from prompt
        topic = "life"  # Simplified extraction
        if "work" in prompt.lower():
            topic = "work"
        elif "family" in prompt.lower():
            topic = "family"
        elif "adult" in prompt.lower():
            topic = "adulthood"
        
        candidates = []
        for template in templates[:3]:  # Generate 3 candidates
            if "{adjective}" in template:
                adjective = random.choice(["unexpected", "concerning", "revealing", "absurd"])
                text = template.format(adjective=adjective, topic=topic)
            elif "{emotion}" in template:
                emotion = random.choice(["uncomfortable", "surprising", "obvious", "hidden"])
                text = template.format(emotion=emotion, topic=topic)
            elif "{comparison}" in template:
                comparison = random.choice(["a video game", "performance art", "a social experiment", "organized chaos"])
                text = template.format(topic=topic, comparison=comparison)
            elif "{revelation}" in template:
                revelation = random.choice(["it's mostly improvisation", "nobody knows what they're doing", "it's more expensive than expected"])
                text = template.format(topic=topic, revelation=revelation)
            else:
                text = template.format(topic=topic)
            
            candidates.append(text)
        
        return candidates
    
    def _check_constraints(self, text: str, constraints: GenerationConstraints) -> bool:
        """Check if text meets all constraints"""
        # Length constraint
        if len(text) > constraints.max_length:
            return False
        
        # Humor score constraint
        humor_score = self.calculate_attribute_score(text, ControlAttribute.HUMOR)
        if humor_score < constraints.min_humor_score:
            return False
        
        # Safety constraint (simplified toxicity check)
        safety_score = self.calculate_attribute_score(text, ControlAttribute.APPROPRIATENESS)
        if safety_score < (1.0 - constraints.max_toxicity):
            return False
        
        # Forbidden words constraint
        if constraints.forbidden_words:
            text_lower = text.lower()
            for word in constraints.forbidden_words:
                if word.lower() in text_lower:
                    return False
        
        return True
    
    def _select_best_candidate(
        self, 
        candidates: List[Dict[str, Any]], 
        control_vectors: List[ControlVector],
        constraints: GenerationConstraints
    ) -> Dict[str, Any]:
        """Select best candidate based on control objectives and constraints"""
        
        # First, filter candidates that meet constraints
        valid_candidates = [c for c in candidates if c["meets_constraints"]]
        
        if not valid_candidates:
            # If no candidates meet constraints, select best constraint-violating one
            valid_candidates = candidates
        
        # Score candidates based on control vectors
        scored_candidates = []
        for candidate in valid_candidates:
            total_score = 0.0
            
            for control_vector in control_vectors:
                attr_score = candidate["scores"][control_vector.attribute.value]
                
                # Calculate how well this matches the control vector
                if control_vector.strength > 0:
                    # Want high score for this attribute
                    match_score = attr_score
                else:
                    # Want low score for this attribute
                    match_score = 1.0 - attr_score
                
                weighted_score = match_score * control_vector.weight
                total_score += weighted_score
            
            scored_candidates.append((candidate, total_score))
        
        # Select highest scoring candidate
        if scored_candidates:
            return max(scored_candidates, key=lambda x: x[1])[0]
        
        return candidates[0]  # Fallback
    
    def get_control_stats(self) -> Dict[str, Any]:
        """Get statistics about control capabilities"""
        return {
            "available_attributes": [attr.value for attr in ControlAttribute],
            "control_templates": len(self.control_templates),
            "humor_keywords": {k: len(v) for k, v in self.humor_keywords.items()},
            "style_options": list(self.style_keywords.keys()),
            "appropriateness_levels": list(self.appropriateness_keywords.keys())
        }

# Global instance
controlled_humor_generator = ControlledHumorGenerator() 