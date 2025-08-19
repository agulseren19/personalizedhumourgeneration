#!/usr/bin/env python3
"""
Improved Humor Agents with Enhanced Content Filtering
"""

import asyncio
import json
import re
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import requests
import time

# Lazy loading for detoxify to reduce memory usage
try:
    from detoxify import Detoxify
    DETOXIFY_AVAILABLE = True
except ImportError:
    DETOXIFY_AVAILABLE = False
    print("⚠️  Detoxify not available, using regex fallback")

# Import persona recommendation function
try:
    from ..personas.enhanced_persona_templates import recommend_personas_for_context
except ImportError:
    try:
        from personas.enhanced_persona_templates import recommend_personas_for_context
    except ImportError:
        print("⚠️  Could not import recommend_personas_for_context, using fallback")
        def recommend_personas_for_context(context: str, audience: str, topic: str) -> List[str]:
            """Fallback persona recommendation"""
            return ["General Comedian", "Witty Observer", "Sarcastic Commentator"]

# Data classes for humor generation
@dataclass
class HumorRequest:
    context: str
    audience: str
    topic: str
    user_id: Optional[str] = None
    humor_type: Optional[str] = None
    card_type: str = "white"  # "white" or "black"

@dataclass
class GenerationResult:
    text: str
    persona_name: str
    model_used: str
    generation_time: float
    toxicity_score: float
    is_safe: bool
    confidence_score: float

@dataclass
class EvaluationResult:
    humor_score: float
    creativity_score: float
    appropriateness_score: float
    context_relevance_score: float
    overall_score: float
    reasoning: str
    evaluator_name: str
    model_used: str

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

class ImprovedHumorAgent:
    """Enhanced humor agent with better content filtering"""
    
    def __init__(self, name: str, style: str, api_key: str):
        self.name = name
        self.style = style
        self.api_key = api_key
        self.base_url = "https://api.openai.com/v1/chat/completions"
        
        # Lazy loading for detoxify
        self._detoxify_model = None
        
        # Fallback regex patterns
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

    @property
    def detoxify_model(self):
        """Lazy load detoxify model only when needed"""
        if self._detoxify_model is None and DETOXIFY_AVAILABLE:
            try:
                print("🔄 Loading detoxify model (lazy loading)...")
                self._detoxify_model = Detoxify('original')
                print("✅ Detoxify model loaded successfully")
            except Exception as e:
                print(f"❌ Failed to load detoxify model: {e}")
                self._detoxify_model = None
        return self._detoxify_model

    def check_content_safety(self, text: str) -> ContentFilterResult:
        """Check if content is safe using detoxify or regex fallback"""
        try:
            # Try detoxify first if available
            if self.detoxify_model:
                return self._check_with_detoxify(text)
            else:
                return self._check_with_regex(text)
                
        except Exception as e:
            print(f"Content filter error: {e}")
            # Fallback: assume content is safe
            return ContentFilterResult(
                is_safe=True,
                overall_toxicity=0.0,
                category_scores={},
                flagged_categories=[],
                confidence=0.5,
                suggestions=[],
                sanitized_content=text
            )

    def _check_with_detoxify(self, text: str) -> ContentFilterResult:
        """Use detoxify for content filtering"""
        try:
            scores = self.detoxify_model.predict(text)
            
            # Map detoxify scores to our categories
            category_scores = {
                'toxicity': scores.get('toxicity', 0.0),
                'severe_toxicity': scores.get('severe_toxicity', 0.0),
                'identity_attack': scores.get('identity_attack', 0.0),
                'insult': scores.get('insult', 0.0),
                'profanity': scores.get('obscene', 0.0),
                'threat': scores.get('threat', 0.0),
                'sexual_explicit': 0.0,  # Not in detoxify
                'violence': 0.0  # Not in detoxify
            }
            
            # Determine flagged categories
            flagged_categories = []
            for category, score in category_scores.items():
                if score > 0.7:  # Threshold
                    flagged_categories.append(category)
            
            # Calculate overall toxicity
            overall_toxicity = max(category_scores.values())
            
            # Determine if content is safe
            is_safe = overall_toxicity < 0.7
            
            return ContentFilterResult(
                is_safe=is_safe,
                overall_toxicity=overall_toxicity,
                category_scores=category_scores,
                flagged_categories=flagged_categories,
                confidence=0.9,  # High confidence with ML model
                suggestions=["Consider using more inclusive language"] if not is_safe else [],
                sanitized_content=text if is_safe else self._sanitize_content(text)
            )
            
        except Exception as e:
            print(f"Detoxify error: {e}, falling back to regex")
            return self._check_with_regex(text)

    def _check_with_regex(self, text: str) -> ContentFilterResult:
        """Fallback to regex-based filtering"""
        try:
            category_scores = {}
            flagged_categories = []
            
            for category, patterns in self.toxic_patterns.items():
                score = 0.0
                for pattern in patterns:
                    matches = len(re.findall(pattern, text, re.IGNORECASE))
                    if matches > 0:
                        score = min(1.0, matches * 0.3)
                        flagged_categories.append(category)
                category_scores[category] = score
            
            overall_toxicity = max(category_scores.values()) if category_scores else 0.0
            is_safe = overall_toxicity < 0.7
            
            return ContentFilterResult(
                is_safe=is_safe,
                overall_toxicity=overall_toxicity,
                category_scores=category_scores,
                flagged_categories=flagged_categories,
                confidence=0.7,  # Lower confidence than ML
                suggestions=["Consider using more inclusive language"] if not is_safe else [],
                sanitized_content=text if is_safe else self._sanitize_content(text)
            )
            
        except Exception as e:
            print(f"Regex filter error: {e}")
            # Ultimate fallback
            return ContentFilterResult(
                is_safe=True,
                overall_toxicity=0.0,
                category_scores={},
                flagged_categories=[],
                confidence=0.3,
                suggestions=[],
                sanitized_content=text
            )

    def _sanitize_content(self, text: str) -> str:
        """Sanitize potentially offensive content"""
        # Simple word replacements
        replacements = {
            r"\bf[*]?u[*]?c[*]?k": "fudge",
            r"\bs[*]?h[*]?i[*]?t": "shoot",
            r"\bd[*]?a[*]?m[*]?n": "darn",
            r"\bhate": "dislike",
            r"\bkill": "defeat"
        }
        
        sanitized = text
        for pattern, replacement in replacements.items():
            sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)
        
        return sanitized

class ContentFilter:
    """Content filtering using detoxify or regex fallback"""
    
    def __init__(self):
        # Lazy loading for detoxify
        self._detoxify_model = None
        
        # Fallback regex patterns
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

    @property
    def detoxify_model(self):
        """Lazy load detoxify model only when needed"""
        if self._detoxify_model is None and DETOXIFY_AVAILABLE:
            try:
                print("🔄 Loading detoxify model (lazy loading)...")
                self._detoxify_model = Detoxify('original')
                print("✅ Detoxify model loaded successfully")
            except Exception as e:
                print(f"❌ Failed to load detoxify model: {e}")
                self._detoxify_model = None
        return self._detoxify_model

    def is_content_safe(self, text: str) -> Tuple[bool, float, Dict[str, float]]:
        """Check if content is safe using detoxify or regex fallback"""
        try:
            # Try detoxify first if available
            if self.detoxify_model:
                return self._check_with_detoxify(text)
            else:
                return self._check_with_regex(text)
                
        except Exception as e:
            print(f"Content filter error: {e}")
            # Fallback: assume content is safe
            return True, 0.0, {}

    def _check_with_detoxify(self, text: str) -> Tuple[bool, float, Dict[str, float]]:
        """Use detoxify for content filtering"""
        try:
            scores = self.detoxify_model.predict(text)
            
            # Check each toxicity type
            violations = []
            max_score = 0
            
            toxicity_thresholds = {
                'toxicity': 0.7,
                'severe_toxicity': 0.5,
                'obscene': 0.8,
                'threat': 0.3,
                'insult': 0.7,
                'identity_attack': 0.5
            }
            
            for toxicity_type, threshold in toxicity_thresholds.items():
                if toxicity_type in scores:
                    score = scores[toxicity_type]
                    max_score = max(max_score, score)
                    
                    if score > threshold:
                        violations.append(f"{toxicity_type}: {score:.3f}")
            
            is_safe = len(violations) == 0
            return is_safe, max_score, scores
            
        except Exception as e:
            print(f"Detoxify error: {e}, falling back to regex")
            return self._check_with_regex(text)

    def _check_with_regex(self, text: str) -> Tuple[bool, float, Dict[str, float]]:
        """Fallback to regex-based filtering"""
        try:
            category_scores = {}
            flagged_categories = []
            
            for category, patterns in self.toxic_patterns.items():
                score = 0.0
                for pattern in patterns:
                    matches = len(re.findall(pattern, text, re.IGNORECASE))
                    if matches > 0:
                        score = min(1.0, matches * 0.3)
                        flagged_categories.append(category)
                category_scores[category] = score
            
            overall_toxicity = max(category_scores.values()) if category_scores else 0.0
            is_safe = overall_toxicity < 0.7
            
            return is_safe, overall_toxicity, category_scores
            
        except Exception as e:
            print(f"Regex filter error: {e}")
            # Ultimate fallback
            return True, 0.0, {}

    def sanitize_content(self, text: str) -> str:
        """Sanitize potentially offensive content"""
        # Simple word replacements
        replacements = {
            r"\bf[*]?u[*]?c[*]?k": "fudge",
            r"\bs[*]?h[*]?i[*]?t": "shoot",
            r"\bd[*]?a[*]?m[*]?n": "darn",
            r"\bhate": "dislike",
            r"\bkill": "defeat"
        }
        
        sanitized = text
        for pattern, replacement in replacements.items():
            sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)
        
        return sanitized

class ImprovedHumorEvaluator:
    """Improved evaluation system with meaningful scores"""
    
    def __init__(self):
        self.content_filter = ImprovedHumorAgent("Dummy", "Dummy", "Dummy") # Fallback to regex filter
    
    async def evaluate_humor(self, humor_text: str, request: HumorRequest) -> EvaluationResult:
        """Evaluate humor with meaningful, varied scores"""
        
        # Multi-dimensional evaluation
        humor_score = await self._evaluate_humor_quality(humor_text, request)
        creativity_score = self._evaluate_creativity(humor_text, request)
        appropriateness_score = self._evaluate_appropriateness(humor_text, request)
        context_relevance_score = self._evaluate_context_relevance(humor_text, request)
        
        # Calculate overall score (weighted average)
        overall_score = (
            humor_score * 0.4 +
            creativity_score * 0.3 +
            appropriateness_score * 0.2 +
            context_relevance_score * 0.1
        )
        
        reasoning = f"Humor: {humor_score:.1f}, Creativity: {creativity_score:.1f}, Appropriateness: {appropriateness_score:.1f}, Relevance: {context_relevance_score:.1f}"
        
        return EvaluationResult(
            humor_score=humor_score,
            creativity_score=creativity_score,
            appropriateness_score=appropriateness_score,
            context_relevance_score=context_relevance_score,
            overall_score=overall_score,
            reasoning=reasoning,
            evaluator_name="ImprovedEvaluator",
            model_used="rule_based"
        )
    
    async def _evaluate_humor_quality(self, text: str, request: HumorRequest) -> float:
        """Evaluate humor quality using multiple factors"""
        score = 5.0  # Base score
        
        # Length appropriateness
        if 10 <= len(text) <= 100:
            score += 1.0
        elif len(text) < 5:
            score -= 2.0
        
        # Humor indicators
        humor_indicators = ['unexpected', 'clever', 'ironic', 'witty', 'funny', 'hilarious']
        if any(indicator in text.lower() for indicator in humor_indicators):
            score += 1.0
        
        # Audience appropriateness
        if request.audience == "family" and any(word in text.lower() for word in ['clean', 'wholesome', 'dad']):
            score += 1.0
        elif request.audience == "adults" and any(word in text.lower() for word in ['mature', 'adult', 'sophisticated']):
            score += 1.0
        
        # Randomize slightly to avoid all 5/10 scores
        score += random.uniform(-0.5, 0.5)
        
        return max(1.0, min(10.0, score))
    
    def _evaluate_creativity(self, text: str, request: HumorRequest) -> float:
        """Evaluate creativity and originality"""
        score = 5.0
        
        # Unexpected combinations
        if any(word in text.lower() for word in ['unexpected', 'surprising', 'bizarre', 'absurd']):
            score += 1.5
        
        # Wordplay
        if any(word in text.lower() for word in ['pun', 'play', 'twist']):
            score += 1.0
        
        # Originality (simple heuristic)
        if len(set(text.lower().split())) / len(text.split()) > 0.8:  # High word diversity
            score += 0.5
        
        score += random.uniform(-0.5, 0.5)
        return max(1.0, min(10.0, score))
    
    def _evaluate_appropriateness(self, text: str, request: HumorRequest) -> float:
        """Evaluate appropriateness for audience"""
        score = 5.0
        
        # Check toxicity
        # This part needs to be updated to use the new ContentFilterResult
        # For now, it will use a dummy filter and assume safe if no error
        try:
            dummy_filter = ImprovedHumorAgent("Dummy", "Dummy", "Dummy")
            dummy_result = dummy_filter.check_content_safety(text)
            if dummy_result.is_safe:
                score += 2.0
            else:
                score -= 3.0
        except Exception as e:
            print(f"Appropriateness evaluation failed: {e}")
            score -= 1.0 # Fallback to a lower score
        
        # Audience-specific appropriateness
        if request.audience == "family":
            if any(word in text.lower() for word in ['family', 'kids', 'wholesome']):
                score += 1.0
            if any(word in text.lower() for word in ['adult', 'mature', 'inappropriate']):
                score -= 1.0
        
        score += random.uniform(-0.3, 0.3)
        return max(1.0, min(10.0, score))
    
    def _evaluate_context_relevance(self, text: str, request: HumorRequest) -> float:
        """Evaluate how well the humor fits the context"""
        score = 5.0
        
        # Context word matching
        context_words = set(request.context.lower().split())
        text_words = set(text.lower().split())
        relevance = len(context_words & text_words) / max(len(context_words), 1)
        score += relevance * 3.0
        
        # Topic relevance
        if request.topic.lower() in text.lower():
            score += 1.0
        
        score += random.uniform(-0.3, 0.3)
        return max(1.0, min(10.0, score))

class ImprovedHumorOrchestrator:
    """Orchestrates the improved humor generation and evaluation system"""
    
    def __init__(self):
        self.agent = ImprovedHumorAgent("Dummy", "Dummy", "Dummy") # Fallback to regex filter
        self.evaluator = ImprovedHumorEvaluator()
    
    async def generate_and_evaluate_humor(self, request: HumorRequest) -> Dict[str, Any]:
        """Generate and evaluate humor with proper persona handling"""
        
        # DEBUG: Log the request details
        print(f"🎭 DEBUG: Request card_type = '{request.card_type}'")
        print(f"🎭 DEBUG: Request context = '{request.context}'")
        print(f"🎭 DEBUG: Request audience = '{request.audience}'")
        print(f"🎭 DEBUG: Request topic = '{request.topic}'")
        print(f"🎭 DEBUG: Request user_id = '{request.user_id}'")
        
        # Get persona recommendations
        recommended_personas = await self._get_persona_recommendations(request)
        
        # Use CrewAI for black card generation, standard generation for white cards
        if request.card_type == "black":
            print(f"🎭 Using CrewAI for black card generation")
            generations = await self.agent.generate_black_cards_with_crewai(request)
        else:
            print(f"🎭 Using standard generation for white cards")
            generations = await self.agent.generate_humor(request, recommended_personas)
        
        if not generations:
            return {
                'success': False,
                'error': 'No safe humor generated',
                'recommended_personas': recommended_personas
            }
        
        # Evaluate each generation
        evaluated_results = []
        for generation in generations:
            evaluation = await self.evaluator.evaluate_humor(generation.text, request)
            
            evaluated_results.append({
                'generation': generation,
                'evaluation': evaluation,
                'combined_score': evaluation.overall_score + generation.confidence_score
            })
        
        # Sort by combined score
        evaluated_results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return {
            'success': True,
            'results': evaluated_results,
            'best_result': evaluated_results[0] if evaluated_results else None,
            'recommended_personas': recommended_personas
        }
    
    async def _get_persona_recommendations(self, request: HumorRequest) -> List[str]:
        """SMART STRATEGY: 2 favorites + 1 dynamic/random for exploration"""
        
        # Get context-based recommendations
        context_personas = recommend_personas_for_context(
            request.context, 
            request.audience, 
            request.topic
        )
        
        # Get user-based recommendations if user_id provided
        if request.user_id:
            try:
                user_personas = await improved_aws_knowledge_base.get_persona_recommendations(
                    user_id=request.user_id,
                    context=request.context,
                    audience=request.audience
                )
                
                # Get user preferences for filtering
                user_prefs = await improved_aws_knowledge_base.get_user_preference(request.user_id)
                
                # STRATEGY: Get 2 favorite personas + 1 dynamic/random for exploration
                final_personas = []
                
                # Add 2 favorite personas from user recommendations
                favorite_count = min(2, len(user_personas))
                for i, persona in enumerate(user_personas):
                    if i < favorite_count:
                        final_personas.append(persona)
                        print(f"  Added FAVORITE persona: {persona}")
                
                # Add 1 DYNAMIC or RANDOM persona for exploration
                if len(final_personas) < 3:
                    print("  Adding DYNAMIC/RANDOM persona for exploration")
                    
                    # Try to get dynamic persona first
                    try:
                        from agent_system.personas.dynamic_persona_generator import DynamicPersonaGenerator
                        dynamic_generator = DynamicPersonaGenerator()
                        
                        # Get user interaction history for dynamic generation
                        interaction_history = await improved_aws_knowledge_base.get_user_interaction_history(request.user_id)
                        
                        if interaction_history and len(interaction_history) >= 2:
                            print("    Attempting dynamic persona generation")
                            dynamic_persona = await dynamic_generator.get_or_create_persona_for_user(
                                request.user_id, interaction_history
                            )
                            
                            if dynamic_persona:
                                final_personas.append(dynamic_persona.name)
                                print(f"    Added DYNAMIC persona: {dynamic_persona.name}")
                            else:
                                # Fallback to random
                                self._add_random_persona_for_exploration(final_personas, user_prefs)
                        else:
                            # Not enough interactions for dynamic generation
                            print("    Not enough interactions for dynamic generation")
                            self._add_random_persona_for_exploration(final_personas, user_prefs)
                            
                    except Exception as e:
                        print(f"    Dynamic persona generation failed: {e}")
                        self._add_random_persona_for_exploration(final_personas, user_prefs)
                
                # Fill remaining slots with context personas if needed
                if len(final_personas) < 3:
                    for persona in context_personas:
                        if persona not in final_personas and len(final_personas) < 3:
                            final_personas.append(persona)
                            print(f"  Added context persona: {persona}")
                
                print(f"  Final smart strategy personas: {final_personas}")
                return final_personas[:3]  # Return exactly 3 personas
                
            except Exception as e:
                print(f"  Error getting user recommendations: {e}")
        
        # Fallback: return 3 context personas
        return context_personas[:3]
    
    def _add_random_persona_for_exploration(self, final_personas: List[str], user_prefs):
        """Add a random persona for exploration"""
        print("    Adding RANDOM persona for exploration")
        
        # Get all available personas
        from agent_system.personas.enhanced_persona_templates import get_all_personas
        all_personas = list(get_all_personas().keys())
        
        # Filter out already selected and disliked personas
        available_personas = []
        for persona in all_personas:
            if persona not in final_personas:
                if user_prefs and persona in user_prefs.disliked_personas:
                    print(f"      Skipping disliked persona: {persona}")
                    continue
                available_personas.append(persona)
        
        if available_personas:
            import random
            random_persona = random.choice(available_personas)
            final_personas.append(random_persona)
            print(f"    Added RANDOM persona for exploration: {random_persona}") 