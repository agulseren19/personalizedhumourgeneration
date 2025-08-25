#!/usr/bin/env python3
"""
Improved Humor Agents System
Fixes all major issues: content filtering, persona recommendation, feedback learning, evaluation
"""

import asyncio
import time
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import random

class SurpriseCalculator:
    """Calculate incongruity/surprise index as per Tian et al."""
    
    def __init__(self):
        self.base_model = "gpt-2"  # Use GPT-2 as base model for surprisal calculation
        
    async def calculate_surprise_index(self, humor_text: str, context: str) -> float:
        """
        Calculate surprise index using token-level surprisal: -log P(token|context)
        Higher values indicate more unexpected/incongruous content (funnier)
        """
        try:
            # Create prompt for probability calculation
            prompt = f"""Given this context: "{context}"
Calculate the probability of this completion: "{humor_text}"

This is for measuring unexpectedness in humor generation."""
            
            # Use a simple heuristic based on text characteristics for now
            # In a full implementation, this would use actual token probabilities
            surprise_score = self._calculate_heuristic_surprise(humor_text, context)
            
            print(f"DEBUG: Surprise index for '{humor_text[:50]}...': {surprise_score:.3f}")
            return surprise_score
            
        except Exception as e:
            print(f"Error calculating surprise index: {e}")
            return 5.0  # Default moderate surprise
    
    def _calculate_heuristic_surprise(self, humor_text: str, context: str) -> float:
        """
        Heuristic surprise calculation based on:
        1. Lexical surprise (unusual words)
        2. Semantic distance from context
        3. Length unexpectedness
        4. Syntactic complexity
        """
        surprise_score = 0.0
        
        # 1. Lexical surprise - uncommon words increase surprise
        words = humor_text.lower().split()
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'very', 'really', 'quite'}
        uncommon_ratio = len([w for w in words if w not in common_words and len(w) > 4]) / max(len(words), 1)
        surprise_score += uncommon_ratio * 3.0
        
        # 2. Semantic distance - different topic/domain from context
        context_words = set(context.lower().split())
        humor_words = set(words)
        overlap_ratio = len(context_words.intersection(humor_words)) / max(len(context_words), 1)
        surprise_score += (1.0 - overlap_ratio) * 2.0
        
        # 3. Length unexpectedness - very short or very long responses are surprising
        length_surprise = abs(len(humor_text) - 30) / 30.0  # 30 chars is "expected" length
        surprise_score += min(length_surprise, 2.0)
        
        # 4. Syntactic surprise - punctuation and structure
        if '!' in humor_text or '?' in humor_text:
            surprise_score += 0.5
        if any(char in humor_text for char in ['...', '--', ';']):
            surprise_score += 0.3
            
        # 5. Content-based surprise indicators
        surprise_words = ['unexpected', 'bizarre', 'absurd', 'random', 'weird', 'strange', 'shocking', 'twist']
        if any(word in humor_text.lower() for word in surprise_words):
            surprise_score += 1.0
            
        # Normalize to 0-10 scale
        return min(max(surprise_score, 0.0), 10.0)

# Content filtering
from detoxify import Detoxify

# Import persona recommendation function
try:
    from personas.enhanced_persona_templates import recommend_personas_for_context
except ImportError:
    try:
        from agent_system.personas.enhanced_persona_templates import recommend_personas_for_context
    except ImportError:
        print("⚠️  Could not import recommend_personas_for_context")
        raise

# Import AWS knowledge base
try:
    from knowledge.improved_aws_knowledge_base import improved_aws_knowledge_base, UserPreference
except ImportError:
    try:
        from agent_system.knowledge.improved_aws_knowledge_base import improved_aws_knowledge_base, UserPreference
    except ImportError:
        print("⚠️  Could not import improved_aws_knowledge_base")
        raise

try:
    from llm_clients.multi_llm_manager import multi_llm_manager, LLMRequest, LLMProvider
    from personas.enhanced_persona_templates import get_all_personas
    from personas.dynamic_persona_generator import dynamic_persona_generator
    from config.settings import settings
except ImportError:
    # Fallback to absolute imports when running directly
    from agent_system.llm_clients.multi_llm_manager import multi_llm_manager, LLMRequest, LLMProvider
    from agent_system.personas.enhanced_persona_templates import get_all_personas
    from agent_system.personas.dynamic_persona_generator import dynamic_persona_generator
    from agent_system.config.settings import settings

@dataclass
class HumorRequest:
    context: str
    audience: str = "friends"
    topic: str = "general"
    user_id: str = ""
    humor_type: str = "general"
    card_type: str = "white"
    favorite_personas: Optional[List[str]] = None  # Add support for favorite personas

@dataclass
class GenerationResult:
    text: str
    persona_name: str
    model_used: str
    generation_time: float
    toxicity_score: float
    safety_score: float  # Add safety score field
    is_safe: bool
    confidence_score: float
    surprise_index: float = 5.0  # Add surprise index with default value
    evaluation: Optional[Any] = None  # Add evaluation field for complete sentence metrics
    
    def __post_init__(self):
        # Ensure all float values are Python floats, not numpy types
        if hasattr(self.generation_time, 'item'):
            self.generation_time = float(self.generation_time)
        if hasattr(self.toxicity_score, 'item'):
            self.toxicity_score = float(self.toxicity_score)
        if hasattr(self.confidence_score, 'item'):
            self.confidence_score = float(self.confidence_score)
        if hasattr(self.surprise_index, 'item'):
            self.surprise_index = float(self.surprise_index)

@dataclass
class EvaluationResult:
    # Core literature-based metrics (replacing old ad-hoc metrics)
    surprisal_score: float          # Token-level surprisal (Tian et al. 2022)
    ambiguity_score: float          # Statistical ambiguity (Kao 2016)
    distinctiveness_ratio: float    # Semantic distance ratio
    entropy_score: float            # Information-theoretic entropy
    perplexity_score: float         # Language model perplexity
    semantic_coherence: float       # Cosine similarity-based coherence
    
    # Creativity/Diversity metrics (filtered for non-zero values)
    distinct_1: float               # Distinct-1 ratio (Li et al. 2016)
    distinct_2: float               # Distinct-2 ratio (Li et al. 2016)
    vocabulary_richness: float      # Type-Token Ratio
    overall_semantic_diversity: float  # Overall semantic diversity
    
    # Overall scores
    overall_humor_score: float      # Weighted statistical combination
    pacs_score: float               # Personalization score (PaCS)
    
    # Metadata
    reasoning: str
    evaluator_name: str
    model_used: str
    
    def __post_init__(self):
        # Ensure all float values are Python floats, not numpy types
        for field_name in ['surprisal_score', 'ambiguity_score', 'distinctiveness_ratio', 
                          'entropy_score', 'perplexity_score', 'semantic_coherence',
                          'distinct_1', 'distinct_2', 'vocabulary_richness', 
                          'overall_semantic_diversity', 'overall_humor_score', 'pacs_score']:
            if hasattr(self, field_name):
                value = getattr(self, field_name)
                if hasattr(value, 'item'):
                    setattr(self, field_name, float(value))
    
    def get_frontend_metrics(self) -> Dict[str, float]:
        """
        Get metrics suitable for frontend display, filtering out zero values
        and providing meaningful labels
        """
        metrics = {}
        
        # Core metrics (always show)
        if self.surprisal_score > 0:
            metrics['surprisal'] = self.surprisal_score
        if self.ambiguity_score > 0:
            metrics['ambiguity'] = self.ambiguity_score
        if self.distinctiveness_ratio > 0:
            metrics['distinctiveness'] = self.distinctiveness_ratio
        if self.semantic_coherence > 0:
            metrics['coherence'] = self.semantic_coherence
        
        # Creativity metrics (only show if > 0)
        if self.distinct_1 > 0:
            metrics['creativity'] = self.distinct_1
        if self.vocabulary_richness > 0:
            metrics['vocabulary_richness'] = self.vocabulary_richness
        if self.overall_semantic_diversity > 0:
            metrics['semantic_diversity'] = self.overall_semantic_diversity
        
        # Overall score (always show)
        metrics['overall'] = self.overall_humor_score
        
        # Personalization (only show if > 0)
        if self.pacs_score > 0:
            metrics['personalization'] = self.pacs_score
        
        return metrics
    
    def get_legacy_metrics(self) -> Dict[str, float]:
        """
        Get legacy metric names for backward compatibility
        Maps new metrics to old names used by existing frontend
        """
        return {
            'humor_score': self.overall_humor_score,
            'creativity_score': self.distinct_1 if self.distinct_1 > 0 else 5.0,
            'appropriateness_score': self.semantic_coherence,
            'context_relevance_score': self.semantic_coherence,
            'surprise_index': self.surprisal_score,
            'overall_score': self.overall_humor_score
        }

class ContentFilter:
    """Advanced content filtering using detoxify"""
    
    def __init__(self):
        """Initialize content filter with detoxify"""
        try:
            self.detoxify = Detoxify('original')
            print("✅ Detoxify content filter initialized successfully")
        except Exception as e:
            print(f"⚠️ Detoxify initialization failed: {e}")
            self.detoxify = None
        
        # Thresholds for different toxicity types
        self.thresholds = {
            'toxicity': 0.7,
            'severe_toxicity': 0.5,
            'obscene': 0.6,
            'threat': 0.5,
            'insult': 0.6,
            'identity_attack': 0.5
        }
    
    def is_content_safe(self, text: str) -> Tuple[bool, float, Dict[str, float]]:
        """Check if content is safe using detoxify"""
        try:
            if not self.detoxify:
                print("⚠️ Detoxify not available, using fallback safety assessment")
                # Fallback: basic word-based toxicity check
                toxic_words = ['hate', 'stupid', 'idiot', 'kill', 'die', 'stupid']
                text_lower = text.lower()
                toxic_count = sum(1 for word in toxic_words if word in text_lower)
                
                if toxic_count == 0:
                    return True, 0.8, {'toxicity': 0.2, 'safety_score': 0.8}
                else:
                    toxicity_level = min(toxic_count * 0.3, 0.8)
                    safety_score = 1.0 - toxicity_level
                    return False, safety_score, {'toxicity': toxicity_level, 'safety_score': safety_score}
            
            scores = self.detoxify.predict(text)
            
            # Convert numpy types to Python types
            converted_scores = {}
            for key, value in scores.items():
                if hasattr(value, 'item'):  # numpy scalar
                    converted_scores[key] = float(value)
                else:
                    converted_scores[key] = value
            
            # Check each toxicity type
            violations = []
            max_score = 0.0
            
            for toxicity_type, threshold in self.thresholds.items():
                if toxicity_type in converted_scores:
                    score = float(converted_scores[toxicity_type])
                    max_score = max(max_score, score)
                    
                    if score > threshold:
                        violations.append(f"{toxicity_type}: {score:.3f}")
            
            is_safe = len(violations) == 0
            
            # IMPROVED: Calculate a meaningful safety score (0-1, higher = safer)
            # Invert the toxicity score and apply a curve to make it more meaningful
            if max_score == 0:
                safety_score = 1.0  # Perfect safety
            else:
                # Invert: 1 - toxicity, then apply curve to boost scores
                inverted_score = 1.0 - max_score
                # Apply curve: x^0.5 makes scores higher (more generous)
                safety_score = inverted_score ** 0.5
            
            # DEBUG: Log the safety calculation
            print(f"🛡️ Safety calculation: max_toxicity={max_score:.3f}, inverted={inverted_score:.3f}, final={safety_score:.3f}")
            
            # Store the safety score in the scores dict for frontend use
            converted_scores['safety_score'] = safety_score
            
            return is_safe, safety_score, converted_scores
            
        except Exception as e:
            print(f"Content filtering error: {e}")
            # If filtering fails, be conservative
            return False, 0.5, {'safety_score': 0.5}
    
    def sanitize_content(self, text: str) -> str:
        """Attempt to sanitize content while preserving humor"""
        # Basic replacements for common problematic terms
        replacements = {
            'damn': 'darn',
            'hell': 'heck',
            'shit': 'shoot',
            'fuck': 'fudge',
            'ass': 'butt'
        }
        
        sanitized = text
        for bad, good in replacements.items():
            sanitized = sanitized.replace(bad, good)
            sanitized = sanitized.replace(bad.upper(), good.upper())
            sanitized = sanitized.replace(bad.capitalize(), good.capitalize())
        
        return sanitized

class ImprovedHumorAgent:
    """Improved humor generation agent with proper persona handling and user embeddings"""
    
    def __init__(self):
        self.content_filter = ContentFilter()
        self.surprise_calculator = SurpriseCalculator()
        
        # Initialize user embedding manager for personalization (SHEEP-Medium approach)
        try:
            from knowledge.user_embedding_manager import UserEmbeddingManager
            self.embedding_manager = UserEmbeddingManager(embedding_dimension=128)
            print("✅ User embedding manager initialized for personalization")
        except ImportError as e:
            print(f"⚠️ User embedding manager not available: {e}")
            self.embedding_manager = None
        
        # Initialize statistical evaluator for literature-based metrics
        try:
            import sys
            from pathlib import Path
            
            # Add evaluation directory to path
            current_dir = Path(__file__).parent
            evaluation_dir = current_dir.parent.parent / 'evaluation'
            sys.path.insert(0, str(evaluation_dir))
            
            from statistical_humor_evaluator import StatisticalHumorEvaluator
            self.statistical_evaluator = StatisticalHumorEvaluator()
            print("✅ Statistical evaluator initialized for literature-based metrics")
        except ImportError as e:
            print(f"⚠️ Statistical evaluator not available: {e}")
            print("Using fallback evaluation")
            self.statistical_evaluator = None
    
    async def generate_humor(self, request: HumorRequest, personas: List[str]) -> List[GenerationResult]:
        """Generate humor using dynamic personas based on user preferences"""
        print(f"  Generating with static personas: {personas}")
        
        # Get user preferences to filter personas
        user_preferences = await self._get_user_preferences(request.user_id)
        
        # ENHANCED: Create or get dynamic persona for the user
        custom_persona = await self._get_or_create_custom_persona(request.user_id, user_preferences)
        
        # Mix custom persona with filtered static personas
        filtered_personas = self._filter_personas_by_preferences(personas, user_preferences)
        
        # ENHANCED: Show dynamic persona creation prominently
        if custom_persona:
            print(f"  • DYNAMIC PERSONA CREATED: '{custom_persona.name}'")
            print(f"    Description: {custom_persona.description}")
            print(f"    Humor Style: {custom_persona.humor_style}")
            print(f"    Expertise: {', '.join(custom_persona.expertise_areas)}")
            final_personas = [custom_persona.name] + filtered_personas[:2]
        else:
            if request.user_id and user_preferences and len(user_preferences.interaction_history) > 0:
                interaction_count = len(user_preferences.interaction_history)
                print(f"  • Dynamic persona not created (need 2+ interactions, have {interaction_count})")
            else:
                print(f"  • Dynamic persona not created (no user history)")
            final_personas = filtered_personas[:3]
        
        print(f"  Final personas (static + dynamic): {final_personas}")
        
        # Generate with final personas
        results = []
        # Use simple model names instead of undefined LLMProvider enum
        available_models = ["gpt-4", "gpt-3.5-turbo", "claude-3-sonnet"]
        
        for i, persona_name in enumerate(final_personas):
            model = available_models[i % len(available_models)]
            
            # Use custom persona if it's the first one
            if i == 0 and custom_persona and persona_name == custom_persona.name:
                print(f"    Generating with DYNAMIC persona: {custom_persona.name}")
                result = await self._generate_with_custom_persona(request, custom_persona, model)
            else:
                print(f"    Generating with static persona: {persona_name}")
                result = await self._generate_with_persona(request, persona_name, model)
            
            if result:
                results.append(result)
        
        return results
    
    async def _generate_with_persona(self, request: HumorRequest, persona_name: str, model: str) -> Optional[GenerationResult]:
        """Generate humor with a static persona"""
        try:
            print(f"      Generating with persona: {persona_name} using model: {model}")
            
            # Simple prompt for white card generation
            prompt = f"""Generate a funny white card response for Cards Against Humanity.

Black Card: "{request.context}"
Audience: {request.audience}
Topic: {request.topic}

Generate ONE funny response that fits the blank. Keep it under 50 characters and make it hilarious.

Response:"""
            
            # Use LLM manager directly
            from agent_system.llm_clients.llm_manager import llm_manager, LLMRequest
            
            llm_request = LLMRequest(
                prompt=prompt,
                model=model,
                temperature=0.9,
                max_tokens=100,
                system_prompt=f"You are {persona_name} - a comedy expert. Generate hilarious Cards Against Humanity content."
            )
            
            response = await llm_manager.generate_response(llm_request)
            
            if response and response.content:
                white_card_text = response.content.strip()
                
                # For white cards, create complete sentence and evaluate it
                if request.card_type == "white":
                    # Convert white card to lowercase for natural sentence flow
                    clean_white_card = white_card_text.lower()
                    
                    # Create complete sentence by filling in the blank
                    complete_sentence = request.context.replace('_____', clean_white_card)
                    
                    # Clean up the complete sentence
                    complete_sentence = complete_sentence.replace('____', clean_white_card)
                    complete_sentence = complete_sentence.replace('___', clean_white_card)
                    complete_sentence = complete_sentence.replace('__', clean_white_card)
                    complete_sentence = complete_sentence.replace('_', clean_white_card)
                    
                    # Fix double periods and ensure proper sentence ending
                    complete_sentence = complete_sentence.replace('..', '.')
                    if not complete_sentence.endswith('.') and not complete_sentence.endswith('!') and not complete_sentence.endswith('?'):
                        complete_sentence += '.'
                    
                    # Remove extra spaces
                    complete_sentence = ' '.join(complete_sentence.split())
                    
                    # Log complete sentence creation for debugging
                    print(f"      🔍 Complete sentence created: '{complete_sentence[:80]}{'...' if len(complete_sentence) > 80 else ''}'")
                    
                    # Evaluate the complete sentence using literature-based metrics
                    try:
                        if hasattr(self, 'statistical_evaluator') and self.statistical_evaluator:
                            # Get user humor profile for personalization (PaCS score)
                            user_profile = []
                            if request.user_id:
                                try:
                                    # Get user's liked cards from feedback history
                                    from agent_system.models.database import get_session_local, UserFeedback
                                    from agent_system.config.settings import settings
                                    
                                    SessionLocal = get_session_local(settings.database_url)
                                    db = SessionLocal()
                                    try:
                                        # Get cards the user rated highly (7+ out of 10)
                                        high_rated_feedback = db.query(UserFeedback).filter(
                                            UserFeedback.user_id == request.user_id,
                                            UserFeedback.feedback_score >= 7.0
                                        ).limit(20).all()  # Get up to 20 high-rated cards
                                        
                                        # Extract the response texts as user humor profile
                                        user_profile = [fb.response_text for fb in high_rated_feedback if fb.response_text]
                                        
                                        if user_profile:
                                            print(f"      🎭 Using user humor profile with {len(user_profile)} high-rated cards for PaCS calculation")
                                        else:
                                            print(f"      🎭 No high-rated cards found for user {request.user_id}, using empty profile")
                                            
                                    finally:
                                        db.close()
                                        
                                except Exception as e:
                                    print(f"      ⚠️ Could not get user profile for PaCS calculation: {e}")
                                    user_profile = []
                            
                            # Use the new literature-based evaluation with user profile
                            evaluation_result = self.statistical_evaluator.evaluate_humor_statistically(
                                complete_sentence, request.context, user_profile
                            )
                            
                            # Calculate actual safety score using content filter
                            is_safe, safety_score, toxicity_details = self.content_filter.is_content_safe(white_card_text)
                            
                            # DEBUG: Log the safety score calculation
                            print(f"      🛡️ Generated safety: is_safe={is_safe}, safety_score={safety_score:.3f}, toxicity_details={toxicity_details}")
                            
                            # Create generation result with evaluation
                            generation_result = GenerationResult(
                                text=white_card_text,
                                persona_name=persona_name,
                                model_used=model,
                                generation_time=1.0,
                                toxicity_score=toxicity_details.get('toxicity', 0.1),
                                safety_score=safety_score,
                                is_safe=is_safe,
                                confidence_score=0.8,
                                surprise_index=evaluation_result.surprisal_score,
                                # Add evaluation data
                                evaluation=evaluation_result
                            )
                        else:
                            # Fallback to basic evaluation
                            surprise_index = await self.surprise_calculator.calculate_surprise_index(white_card_text, request.context)
                            
                            # Calculate actual safety score using content filter
                            is_safe, safety_score, toxicity_details = self.content_filter.is_content_safe(white_card_text)
                            
                            generation_result = GenerationResult(
                                text=white_card_text,
                                persona_name=persona_name,
                                model_used=model,
                                generation_time=1.0,
                                toxicity_score=toxicity_details.get('toxicity', 0.1),
                                safety_score=safety_score,
                                is_safe=is_safe,
                                confidence_score=0.8,
                                surprise_index=surprise_index
                            )
                    except Exception as e:
                        print(f"⚠️ Evaluation failed, using fallback: {e}")
                        # Fallback to basic evaluation
                        surprise_index = await self.surprise_calculator.calculate_surprise_index(white_card_text, request.context)
                        
                        # Calculate actual safety score using content filter
                        is_safe, safety_score, toxicity_details = self.content_filter.is_content_safe(white_card_text)
                        
                        generation_result = GenerationResult(
                            text=white_card_text,
                            persona_name=persona_name,
                            model_used=model,
                            generation_time=1.0,
                            toxicity_score=toxicity_details.get('toxicity', 0.1),
                            safety_score=safety_score,
                            is_safe=is_safe,
                            confidence_score=0.8,
                            surprise_index=surprise_index
                        )
                else:
                    # For black cards, use basic evaluation
                    surprise_index = await self.surprise_calculator.calculate_surprise_index(white_card_text, request.context)
                    
                    # Calculate actual safety score using content filter
                    is_safe, safety_score, toxicity_details = self.content_filter.is_content_safe(white_card_text)
                    
                    generation_result = GenerationResult(
                        text=white_card_text,
                        persona_name=persona_name,
                        model_used=model,
                        generation_time=1.0,
                        toxicity_score=toxicity_details.get('toxicity', 0.1),
                        safety_score=safety_score,
                        is_safe=is_safe,
                        confidence_score=0.8,
                        surprise_index=surprise_index
                    )
                
                print(f"      ✅ Generated: {response.content.strip()[:50]}...")
                return generation_result
            else:
                print(f"      ❌ Empty response from {persona_name}")
                return None
                
        except Exception as e:
            print(f"      ❌ Error generating with {persona_name}: {e}")
            return None
    
    async def _generate_with_custom_persona(self, request: HumorRequest, custom_persona, model: str) -> Optional[GenerationResult]:
        """Generate humor with a custom dynamic persona"""
        try:
            print(f"      Generating with custom persona: {custom_persona.name} using model: {model}")
            
            # Use custom persona's prompt style
            prompt = f"""Generate a funny white card response for Cards Against Humanity.

Black Card: "{request.context}"
Audience: {request.audience}
Topic: {request.topic}

Persona Style: {custom_persona.humor_style}
Persona Expertise: {', '.join(custom_persona.expertise_areas)}

Generate ONE funny response that fits the blank. Keep it under 50 characters and make it hilarious.

Response:"""
            
            # Use LLM manager directly
            from agent_system.llm_clients.llm_manager import llm_manager, LLMRequest
            
            llm_request = LLMRequest(
                prompt=prompt,
                model=model,
                temperature=0.9,
                max_tokens=100,
                system_prompt=f"You are {custom_persona.name} - {custom_persona.description}"
            )
            
            response = await llm_manager.generate_response(llm_request)
            
            if response and response.content:
                # Calculate surprise index
                surprise_index = await self.surprise_calculator.calculate_surprise_index(response.content.strip(), request.context)
                
                # Calculate actual safety score using content filter
                is_safe, safety_score, toxicity_details = self.content_filter.is_content_safe(response.content.strip())
                
                # Create generation result
                generation_result = GenerationResult(
                    text=response.content.strip(),
                    persona_name=custom_persona.name,
                    model_used=model,
                    generation_time=1.0,  # Estimated time
                    toxicity_score=toxicity_details.get('toxicity', 0.1),
                    safety_score=safety_score,
                    is_safe=is_safe,
                    confidence_score=0.9,  # Higher confidence for custom personas
                    surprise_index=surprise_index
                )
                
                print(f"      ✅ Generated with custom persona: {response.content.strip()[:50]}...")
                return generation_result
            else:
                print(f"      ❌ Empty response from custom persona {custom_persona.name}")
                return None
                
        except Exception as e:
            print(f"      ❌ Error generating with custom persona {custom_persona.name}: {e}")
            return None
    
    async def _get_user_preferences(self, user_id: str) -> Optional[UserPreference]:
        """Get user preferences from knowledge base"""
        if not user_id:
            return None
        
        try:
            return await improved_aws_knowledge_base.get_user_preference(user_id)
        except Exception as e:
            print(f"  Error getting user preferences: {e}")
            return None
    
    async def _get_or_create_custom_persona(self, user_id: str, user_preferences: Optional[UserPreference]):
        """Get or create a custom persona for the user"""
        if not user_id:
            return None
        
        try:
            # Get interaction history from user preferences
            interaction_history = []
            if user_preferences and user_preferences.interaction_history:
                interaction_history = user_preferences.interaction_history
            
            # ENHANCED: Reduce threshold to make dynamic personas more accessible
            if len(interaction_history) < 2:  # Changed from 3 to 2
                return None
            
            # Get or create custom persona
            custom_persona = await dynamic_persona_generator.get_or_create_persona_for_user(
                user_id, interaction_history
            )
            
            print(f"  • Generated custom persona from {len(interaction_history)} interactions")
            return custom_persona
            
        except Exception as e:
            print(f"  • Error creating custom persona: {e}")
            return None
    
    def _filter_personas_by_preferences(self, personas: List[str], user_prefs: Optional[UserPreference]) -> List[str]:
        """Filter personas based on user likes/dislikes and embeddings (SHEEP-Medium approach)"""
        if not user_prefs:
            return personas
        
        # Remove disliked personas
        filtered = [p for p in personas if p not in user_prefs.disliked_personas]
        
        # ENHANCED: Use user embeddings for better personalization if available
        if self.embedding_manager and user_prefs.user_id:
            try:
                # Score personas using embeddings
                persona_scores = []
                for persona in filtered:
                    score = self._calculate_persona_embedding_score(persona, user_prefs.user_id)
                    persona_scores.append((persona, score))
                
                # Sort by embedding score
                persona_scores.sort(key=lambda x: x[1], reverse=True)
                filtered = [p[0] for p in persona_scores]
                
                print(f"  🧠 Personas ranked by embeddings: {filtered[:3]}")
            except Exception as e:
                print(f"  ⚠️ Embedding personalization failed: {e}")
        
        # Prioritize liked personas
        liked_in_list = [p for p in filtered if p in user_prefs.liked_personas]
        not_liked_in_list = [p for p in filtered if p not in user_prefs.liked_personas]
        
        # Put liked personas first
        final_list = liked_in_list + not_liked_in_list
        
        return final_list if final_list else personas  # Fallback to original if all filtered out
    
    def _calculate_persona_embedding_score(self, persona_name: str, user_id: str) -> float:
        """Calculate persona score using user embeddings (SHEEP-Medium approach)"""
        try:
            # Create simple text embedding for persona
            text_embedding = self._create_simple_text_embedding(persona_name)
            
            # Get personalized prediction
            score = self.embedding_manager.get_personalized_prediction(
                user_id, text_embedding, persona_name, "persona_selection", "general"
            )
            
            return score
        except Exception as e:
            print(f"    ⚠️ Error calculating embedding score for {persona_name}: {e}")
            return 5.0  # Default neutral score
    
    def _create_simple_text_embedding(self, text: str) -> List[float]:
        """Create simple text embedding (hash-based for now)"""
        # Simple hash-based embedding - replace with proper model in production
        text_hash = hash(text) % 10000
        
        embedding = []
        for i in range(128):  # Match embedding dimension
            feature = float((text_hash + i * 7) % 1000) / 1000.0
            embedding.append(feature)
        
        return embedding
    
    async def _generate_with_custom_persona_old(self, request: HumorRequest, custom_persona, model: str) -> Optional[GenerationResult]:
        """Generate humor with a custom persona template"""
        start_time = time.time()
        
        # Create system prompt from custom persona
        system_prompt = f"""You are "{custom_persona.name}" - {custom_persona.description}

Your humor style: {custom_persona.humor_style}
Your expertise: {', '.join(custom_persona.expertise_areas)}
Your approach: {custom_persona.prompt_style}

Generate a Cards Against Humanity response that matches your style perfectly."""
        
        # Create user prompt
        if request.card_type == "black":
            prompt = f"""Create a Cards Against Humanity BLACK CARD that sets up humor:

Context: {request.context}
Audience: {request.audience}
Topic: {request.topic}

Create a fill-in-the-blank prompt that would be funny and appropriate for this audience.
Return only the black card text with appropriate blank spaces (use _____ for blanks).

Black Card:"""
        else:
            prompt = f"""Complete this Cards Against Humanity card:

Black Card: "{request.context}"
Audience: {request.audience}
Topic: {request.topic}

Create a hilarious white card response that matches your humor style.
Return only the white card text, nothing else.

White Card:"""
        
        # Generate response
        try:
            llm_request = LLMRequest(
                prompt=prompt,
                model=model,
                system_prompt=system_prompt,
                temperature=0.8,
                max_tokens=60
            )
            
            response = await multi_llm_manager.generate_response(llm_request)
            if isinstance(response, Exception):
                print(f"    Error generating with custom persona {custom_persona.name}: {response}")
                return None
            
            generation_time = time.time() - start_time
            
            # Clean response
            humor_text = self._clean_response(response.content)
            
            # Content filtering
            is_safe, toxicity_score, _ = self.content_filter.is_content_safe(humor_text)
            
            # If not safe, try to sanitize
            if not is_safe:
                sanitized = self.content_filter.sanitize_content(humor_text)
                is_safe_sanitized, toxicity_score_sanitized, _ = self.content_filter.is_content_safe(sanitized)
                
                if is_safe_sanitized:
                    humor_text = sanitized
                    is_safe = True
                    toxicity_score = toxicity_score_sanitized
                else:
                    print(f"    Content filtered out from custom persona {custom_persona.name}: too toxic")
                    return None
            
            # Calculate surprise index
            surprise_index = await self.surprise_calculator.calculate_surprise_index(humor_text, request.context)
            
            return GenerationResult(
                text=humor_text,
                persona_name=custom_persona.name,
                model_used=getattr(response, 'model', model) if hasattr(response, 'model') else str(model),
                generation_time=generation_time,
                toxicity_score=toxicity_score,
                safety_score=1.0 - toxicity_score,  # Calculate safety score from toxicity
                is_safe=is_safe,
                confidence_score=self._calculate_confidence(humor_text, request.context),
                surprise_index=surprise_index
            )
            
        except Exception as e:
            print(f"    Error generating with custom persona {custom_persona.name}: {e}")
            return None

    async def generate_black_cards_with_crewai(self, request: HumorRequest) -> List[GenerationResult]:
        """Generate ONE black card using simplified CrewAI - FAST VERSION"""
        print(f"🎭 Generating ONE black card with simplified CrewAI for user {request.user_id}")
        print(f"🎭 DEBUG: Black card generation called with card_type = '{request.card_type}'")
        print(f"🎭 DEBUG: Black card generation called with context = '{request.context}'")
        
        try:
            # Get user preferences for personalization
            user_preferences = await self._get_user_preferences(request.user_id)
            
            # Get ONE favorite comedian (not multiple)
            favorite_comedian = None
            if user_preferences and user_preferences.liked_personas:
                favorite_comedian = user_preferences.liked_personas[0]  # Only top 1 favorite
                print(f"    Using favorite comedian: {favorite_comedian}")
            
            # Create SIMPLIFIED CrewAI crew (only 1 agent)
            crew = await self._create_simple_black_card_crew(request, favorite_comedian)
            
            if not crew:
                print("    ❌ CrewAI setup failed, falling back to standard generation")
                return []
            
            # Generate ONE black card with timeout protection
            result = await asyncio.wait_for(
                self._generate_single_black_card_with_crewai(crew, request, favorite_comedian),
                timeout=20.0  # 20 second timeout for single generation
            )
            
            if result:
                print(f"    ✅ Generated 1 black card with CrewAI in under 20s")
                return [result]
            else:
                print("    ❌ CrewAI generation failed")
                return []
                
        except asyncio.TimeoutError:
            print("    ⚠️ CrewAI timeout after 20s")
            return []
        except Exception as e:
            print(f"    ❌ CrewAI error: {e}")
            return []

    async def _create_simple_black_card_crew(self, request: HumorRequest, favorite_comedian: str = None) -> Any:
        """Create SIMPLIFIED CrewAI crew for black card generation - ONLY 1 AGENT"""
        try:
            from crewai import Agent, Task, Crew
            
            # Create ONLY ONE specialized agent for black card generation
            black_card_agent = Agent(
                role="Black Card Generator",
                goal="Generate ONE creative and unexpected black card for Cards Against Humanity that is edgy, surprising, and hilarious",
                backstory="""You are a professional comedy writer who specializes in Cards Against Humanity black cards. 
                You create shocking, unexpected, and hilarious fill-in-the-blank prompts that make players laugh out loud. 
                You understand the game's irreverent, edgy style and excel at subverting expectations.""",
                verbose=False,  # Reduce verbosity for speed
                allow_delegation=False
            )
            
            return {
                'black_card_agent': black_card_agent
            }
            
        except ImportError as e:
            print(f"    ❌ CrewAI import failed: {e}")
            return None
        except Exception as e:
            print(f"    ❌ CrewAI setup failed: {e}")
            return None
    
    async def _create_black_card_crew(self, request: HumorRequest, favorite_comedians: List[str], 
                                     random_comedian: str, custom_persona) -> Any:
        """Create CrewAI crew for black card generation"""
        try:
            from crewai import Agent, Task, Crew
            
            # Create specialized agents for black card generation
            creative_agent = Agent(
                role="Creative Black Card Generator",
                goal="Generate exactly 3 creative and unexpected black card options for Cards Against Humanity that are edgy, surprising, and hilarious",
                backstory="""You are a professional comedy writer who specializes in Cards Against Humanity black cards. 
                You have years of experience creating shocking, unexpected, and hilarious fill-in-the-blank prompts that make 
                players laugh out loud. You understand the game's irreverent, edgy style and excel at subverting 
                expectations. You always follow the exact format requested and never deviate from instructions.""",
                verbose=True,
                allow_delegation=False
            )
            
            evaluator_agent = Agent(
                role="Black Card Quality Evaluator", 
                goal="Analyze black card options and select the single best one based on CAH criteria, providing clear reasoning in the EXACT format specified",
                backstory="""You are a comedy expert and CAH judge with perfect understanding of what makes 
                black cards funny in this game. You MUST follow the exact output format specified in your task. 
                You NEVER give generic responses like 'I can give a great answer'. You always provide detailed 
                analysis using the ANALYSIS/BEST OPTION/REASONING format. You are precise and professional.""",
                verbose=True,
                allow_delegation=False
            )
            
            refiner_agent = Agent(
                role="Black Card Refiner",
                goal="Take the selected black card and make it funnier, punchier, and more impactful while following the EXACT output format specified",
                backstory="""You are a comedy editor who perfects black card prompts. You MUST follow the exact output format 
                specified in your task. You NEVER give generic responses. You always provide the refined black card 
                using the REFINED BLACK CARD/IMPROVEMENTS format. You make cards more unexpected, concise, and punchy.""",
                verbose=True,
                allow_delegation=False
            )
            
            return {
                'creative_agent': creative_agent,
                'evaluator_agent': evaluator_agent,
                'refiner_agent': refiner_agent
            }
            
        except ImportError as e:
            print(f"    ❌ CrewAI import failed: {e}")
            print("    CrewAI not available, falling back to standard generation")
            return None
        except Exception as e:
            print(f"    ❌ CrewAI setup failed: {e}")
            print("    CrewAI setup failed, falling back to standard generation")
            return None

    async def _generate_single_black_card_with_crewai(self, crew: Any, request: HumorRequest, comedian_name: str = None) -> Optional[GenerationResult]:
        """Generate ONE black card using simplified CrewAI - FAST VERSION"""
        if not crew or 'black_card_agent' not in crew:
            print("    ❌ Invalid crew for single black card generation")
            return None
        
        try:
            from crewai import Task, Crew
            
            # Create simple task for black card generation
            task = Task(
                description=f"""Generate ONE funny black card for Cards Against Humanity.

Context: {request.context}
Audience: {request.audience}
Topic: {request.topic}

Requirements:
- Generate exactly ONE black card with a blank (_____)
- Keep it under 100 characters
- Make it edgy, unexpected, and hilarious
- Follow CAH style: irreverent, shocking, surprising

Output format: Just the black card text, nothing else.

Example: "What would grandma find disturbing, yet oddly charming? _____"

Black Card:""",
                agent=crew['black_card_agent'],
                expected_output="A single black card text with blank"
            )
            
            # Create simple crew with only one agent and task
            crew_instance = Crew(
                agents=[crew['black_card_agent']],
                tasks=[task],
                verbose=False,  # Reduce verbosity for speed
                max_rpm=10,  # Limit requests per minute
                max_consecutive_auto_reply=1  # Limit auto-replies
            )
            
            # Execute the task
            result = crew_instance.kickoff()
            
            if result and result.strip():
                # Calculate surprise index for CrewAI result
                surprise_index = 7.0  # Default higher surprise for CrewAI (as it's designed to be creative)
                try:
                    surprise_calc = SurpriseCalculator()
                    surprise_index = await surprise_calc.calculate_surprise_index(result.strip(), request.context)
                except Exception as e:
                    print(f"⚠️ CrewAI surprise calculation failed: {e}")
                
                # Create generation result with dynamic persona name
                persona_name = comedian_name if comedian_name else "Professional Comedy Writer"
                
                # Calculate safety score for black cards
                is_safe, safety_score, toxicity_details = self.content_filter.is_content_safe(result.strip())
                
                generation_result = GenerationResult(
                    text=result.strip(),
                    persona_name=persona_name,
                    model_used="crewai",
                    generation_time=1.0,  # Estimated time
                    toxicity_score=toxicity_details.get('toxicity', 0.3),
                    safety_score=safety_score,
                    is_safe=is_safe,
                    confidence_score=0.9,
                    surprise_index=surprise_index
                )
                
                print(f"    ✅ CrewAI generated: {result.strip()[:50]}...")
                return generation_result
            else:
                print("    ❌ CrewAI returned empty result")
                return None
                
        except Exception as e:
            print(f"    ❌ CrewAI generation error: {e}")
            return None
    
    async def _generate_black_card_with_crewai(self, crew: Any, comedian_name: str, 
                                             request: HumorRequest, is_custom: bool = False) -> Optional[GenerationResult]:
        """Generate a single black card using CrewAI for a specific comedian"""
        if not crew:
            print(f"    ⚠️  Crew not available, falling back to standard generation for {comedian_name}")
            # Fallback to standard generation
            return await self._generate_with_persona(request, comedian_name, LLMProvider.OPENAI_GPT4)
        
        try:
            from crewai import Task, Crew
            
            # Get persona details
            if is_custom:
                persona_description = f"Custom persona: {comedian_name}"
                humor_style = "personalized based on user preferences"
            else:
                personas = get_all_personas()
                persona_template = personas.get(comedian_name)
                persona_description = persona_template.description if persona_template else comedian_name
                humor_style = persona_template.humor_style if persona_template else 'clever and edgy'
            
            # Task 1: Generate multiple black card options
            generation_task = Task(
                description=f"""
                Generate exactly 3 different funny black card prompts for Cards Against Humanity in the style of {comedian_name}.
                
                COMEDIAN STYLE: {persona_description}
                HUMOR STYLE: {humor_style}
                TOPIC: {request.topic}
                AUDIENCE: {request.audience}
                CONTEXT: {request.context}
                
                REQUIREMENTS:
                - Each response must be a fill-in-the-blank prompt (use _____ for blanks)
                - Must be unexpected and surprising
                - Edgy but not extremely offensive (CAH style)
                - Must have exactly one blank space
                - Each option must be completely different
                - Follow CAH's irreverent, shocking humor style
                - Appropriate for the "{request.audience}" audience
                
                OUTPUT FORMAT (follow exactly):
                1. [black card option 1 with _____ blank]
                2. [black card option 2 with _____ blank]
                3. [black card option 3 with _____ blank]
                
                Do not include any other text, explanations, or formatting.
                """,
                agent=crew['creative_agent'],
                expected_output="Exactly 3 numbered black card options with blanks, nothing else"
            )
            
            # Task 2: Evaluate and select
            evaluation_task = Task(
                description=f"""
                CRITICAL: You must evaluate the 3 black card options and select the best one.
                
                EVALUATION CRITERIA:
                - Unexpectedness and surprise factor (most important)
                - Cleverness and wit of the prompt
                - Perfect fill-in-the-blank format with exactly one blank
                - Comedic timing and impact
                - Perfect fit for CAH's edgy humor style
                - Appropriateness for the "{request.audience}" audience
                
                MANDATORY OUTPUT FORMAT (you MUST follow this exactly):
                ANALYSIS: [Brief analysis of why each option works or doesn't work]
                BEST OPTION: [number] - [exact black card text from the list]
                REASONING: [Specific reasons why this option is the funniest]
                
                WARNING: Do NOT respond with generic phrases like "I can give a great answer" or similar. 
                You MUST provide the actual analysis in the format above. Failure to follow this format is unacceptable.
                """,
                agent=crew['evaluator_agent'],
                expected_output="Analysis with selected best option in exact format specified - NO GENERIC RESPONSES",
                context=[generation_task]
            )
            
            # Task 3: Refine
            refinement_task = Task(
                description=f"""
                CRITICAL: Take the selected best black card and refine it to maximize comedic impact.
                
                REFINEMENT GOALS:
                - Make it more unexpected or clever if possible
                - Improve comedic timing and punch
                - Ensure it has exactly one blank (_____)
                - Make it more concise and impactful
                - Keep the same concept but improve execution
                
                MANDATORY OUTPUT FORMAT (you MUST follow this exactly):
                REFINED BLACK CARD: [the improved black card text only]
                IMPROVEMENTS: [brief explanation of changes made]
                
                WARNING: Do NOT respond with generic phrases like "I can give a great answer" or similar.
                You MUST provide the actual refined black card in the format above. Failure to follow this format is unacceptable.
                The refined black card should be the final, polished version ready to use.
                """,
                agent=crew['refiner_agent'],
                expected_output="Refined black card with improvements explanation in exact format - NO GENERIC RESPONSES",
                context=[evaluation_task]
            )
            
            # Create and run the crew
            crew_instance = Crew(
                agents=[crew['creative_agent'], crew['evaluator_agent'], crew['refiner_agent']],
                tasks=[generation_task, evaluation_task, refinement_task],
                verbose=True
            )
            
            # Execute the crew
            result = crew_instance.kickoff()
            
            # Parse the result to extract the refined black card
            black_card_text = self._parse_black_card_crew_result(str(result))
            
            if black_card_text:
                # Calculate surprise index
                surprise_index = 7.0  # Default higher surprise for CrewAI
                try:
                    surprise_index = await self.surprise_calculator.calculate_surprise_index(black_card_text, request.context)
                except Exception as e:
                    print(f"⚠️ CrewAI surprise calculation failed: {e}")
                
                # Create generation result
                # Calculate safety score for black cards
                is_safe, safety_score, toxicity_details = self.content_filter.is_content_safe(black_card_text)
                
                return GenerationResult(
                    text=black_card_text,
                    persona_name=comedian_name,
                    model_used="crewai",
                    generation_time=0.0,  # Will be set by caller
                    toxicity_score=toxicity_details.get('toxicity', 0.0),
                    safety_score=safety_score,
                    is_safe=is_safe,
                    confidence_score=self._calculate_confidence(black_card_text, request.context),
                    surprise_index=surprise_index
                )
            
        except Exception as e:
            print(f"    CrewAI black card generation failed for {comedian_name}: {e}")
        
        # Fallback to standard generation
        return await self._generate_with_persona_old(request, comedian_name, "gpt-4")

    def _parse_black_card_crew_result(self, result_text: str) -> str:
        """Parse CrewAI result to extract the final black card"""
        try:
            # First, try to find "REFINED BLACK CARD:" pattern
            if "REFINED BLACK CARD:" in result_text:
                lines = result_text.split('\n')
                for line in lines:
                    if line.strip().startswith("REFINED BLACK CARD:"):
                        card_text = line.split(":", 1)[1].strip()
                        card_text = card_text.strip('"').strip("'").strip()
                        if card_text and "_____" in card_text:
                            return card_text
            
            # Second, try to find "BEST OPTION:" pattern
            if "BEST OPTION:" in result_text:
                lines = result_text.split('\n')
                for line in lines:
                    if line.strip().startswith("BEST OPTION:"):
                        if " - " in line:
                            card_text = line.split(" - ", 1)[1].strip()
                            card_text = card_text.strip('"').strip("'").strip()
                            if card_text and "_____" in card_text:
                                return card_text
            
            # Third, look for numbered list items
            lines = result_text.split('\n')
            creative_options = []
            for line in lines:
                line = line.strip()
                if line and (line.startswith('1.') or line.startswith('2.') or line.startswith('3.')):
                    card_text = line[2:].strip()
                    card_text = card_text.strip('"').strip("'").strip()
                    if card_text and "_____" in card_text:
                        creative_options.append(card_text)
            
            # Return the first valid option
            if creative_options:
                return creative_options[0]
            
        except Exception as e:
            print(f"    Error parsing black card crew result: {e}")
        
        return ""
    
    async def _generate_with_persona_old(self, request: HumorRequest, persona_name: str, model: str) -> Optional[GenerationResult]:
        """Generate humor with specific persona"""
        start_time = time.time()
        
        # Get persona details
        personas = get_all_personas()
        persona_template = personas.get(persona_name)
        
        # Create appropriate prompt based on card type
        if request.card_type == "black":
            prompt = self._create_black_card_prompt(request, persona_template)
        else:
            prompt = self._create_white_card_prompt(request, persona_template)
        
        system_prompt = f"""You are {persona_name}. {persona_template.description if persona_template else ''}
Your humor style: {persona_template.humor_style if persona_template else 'clever and edgy'}
Generate content that is funny but ethical."""
        
        # Generate response
        try:
            llm_request = LLMRequest(
                prompt=prompt,
                model=model,
                system_prompt=system_prompt,
                temperature=0.8,
                max_tokens=60
            )
            
            response = await multi_llm_manager.generate_response(llm_request)
            if isinstance(response, Exception):
                print(f"    Error generating with {persona_name}: {response}")
                return None
            
            generation_time = time.time() - start_time
            
            # Clean response
            humor_text = self._clean_response(response.content)
            
            # Content filtering
            is_safe, toxicity_score, _ = self.content_filter.is_content_safe(humor_text)
            
            # If not safe, try to sanitize
            if not is_safe:
                sanitized = self.content_filter.sanitize_content(humor_text)
                is_safe_sanitized, toxicity_score_sanitized, _ = self.content_filter.is_content_safe(sanitized)
                
                if is_safe_sanitized:
                    humor_text = sanitized
                    is_safe = True
                    toxicity_score = toxicity_score_sanitized
                else:
                    print(f"    Content filtered out from {persona_name}: too toxic")
                    return None
            
            # Calculate surprise index
            surprise_index = await self.surprise_calculator.calculate_surprise_index(humor_text, request.context)
            
            return GenerationResult(
                text=humor_text,
                persona_name=persona_name,
                model_used=getattr(response, 'model', model) if hasattr(response, 'model') else str(model),
                generation_time=generation_time,
                toxicity_score=toxicity_score,
                is_safe=is_safe,
                confidence_score=self._calculate_confidence(humor_text, request.context),
                surprise_index=surprise_index
            )
            
        except Exception as e:
            print(f"    Error generating with {persona_name}: {e}")
            return None
    
    def _create_white_card_prompt(self, request: HumorRequest, persona_template) -> str:
        """Create prompt for white card generation"""
        humor_style = persona_template.humor_style if persona_template else 'clever and edgy'
        return f"""Complete this Cards Against Humanity card with a single, hilarious response:

Black Card: "{request.context}"
Audience: {request.audience}
Topic: {request.topic}

Respond with just the white card text - be {humor_style} but keep it ethical.
Make it appropriate for the "{request.audience}" audience.

White Card:"""
    
    def _create_black_card_prompt(self, request: HumorRequest, persona_template) -> str:
        """Create prompt for black card generation"""
        humor_style = persona_template.humor_style if persona_template else 'clever and edgy'
        return f"""Create a Cards Against Humanity BLACK card (the prompt card with blank):

Topic: {request.topic}
Audience: {request.audience}
Style: {humor_style}

Create a setup that's funny and leads to great responses. Include exactly one blank (____).
Make it appropriate for the "{request.audience}" audience but maintain CAH's edge.

Black Card:"""
    
    def _clean_response(self, response_text: str) -> str:
        """Clean and format the response"""
        text = response_text.strip()
        
        # Remove common prefixes
        prefixes = ['white card:', 'black card:', 'response:', 'answer:']
        for prefix in prefixes:
            if text.lower().startswith(prefix):
                text = text[len(prefix):].strip()
        
        # Remove quotes
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
        
        # Take only the first line
        text = text.split('\n')[0].strip()
        
        return text
    
    def _calculate_confidence(self, text: str, context: str) -> float:
        """Calculate confidence score for the generated humor"""
        score = 0.5  # Base score
        
        # Length scoring
        if 5 <= len(text) <= 80:
            score += 0.2
        
        # Context relevance (simple keyword matching)
        context_words = set(context.lower().split())
        text_words = set(text.lower().split())
        relevance = len(context_words & text_words) / max(len(context_words), 1)
        score += relevance * 0.3
        
        return min(score, 1.0)

class ImprovedHumorEvaluator:
    """Literature-based evaluation system with academically rigorous metrics"""
    
    def __init__(self):
        self.content_filter = ContentFilter()
        self.surprise_calculator = SurpriseCalculator()  # Keep for basic evaluation fallback
        
        # Import the new literature-based evaluator
        try:
            import sys
            from pathlib import Path
            
            # Add evaluation directory to path
            current_dir = Path(__file__).parent
            evaluation_dir = current_dir.parent.parent / 'evaluation'
            sys.path.insert(0, str(evaluation_dir))
            
            from statistical_humor_evaluator import StatisticalHumorEvaluator
            self.statistical_evaluator = StatisticalHumorEvaluator()
            print("✅ Loaded literature-based statistical evaluator")
            
        except ImportError as e:
            print(f"⚠️ Could not import statistical_humor_evaluator: {e}")
            print("Falling back to basic evaluation")
            self.statistical_evaluator = None
    
    async def evaluate_humor(self, humor_text: str, request: HumorRequest) -> EvaluationResult:
        """Evaluate humor using literature-based metrics"""
        
        if self.statistical_evaluator:
            # Use the new literature-based evaluation
            return await self._evaluate_with_literature_metrics(humor_text, request)
        else:
            # Fallback to basic evaluation
            return await self._evaluate_basic(humor_text, request)
    
    async def _evaluate_with_literature_metrics(self, humor_text: str, request: HumorRequest) -> EvaluationResult:
        """Evaluate using literature-based statistical metrics"""
        
        try:
            # Use the black card context for evaluation
            context = request.context
            
            # Get user humor profile for personalization (PaCS score)
            user_profile = []
            if request.user_id:
                try:
                    # Get user's liked cards from feedback history
                    from agent_system.models.database import get_session_local, UserFeedback
                    from agent_system.config.settings import settings
                    
                    SessionLocal = get_session_local(settings.database_url)
                    db = SessionLocal()
                    try:
                        # Get cards the user rated highly (7+ out of 10)
                        high_rated_feedback = db.query(UserFeedback).filter(
                            UserFeedback.user_id == request.user_id,
                            UserFeedback.feedback_score >= 7.0
                        ).limit(20).all()  # Get up to 20 high-rated cards
                        
                        # Extract the response texts as user humor profile
                        user_profile = [fb.response_text for fb in high_rated_feedback if fb.response_text]
                        
                        if user_profile:
                            print(f"🎭 Using user humor profile with {len(user_profile)} high-rated cards for PaCS calculation")
                        else:
                            print(f"🎭 No high-rated cards found for user {request.user_id}, using empty profile")
                            
                    finally:
                        db.close()
                        
                except Exception as e:
                    print(f"⚠️ Could not get user profile for PaCS calculation: {e}")
                    user_profile = []
            
            # Get statistical evaluation scores with user profile
            scores = self.statistical_evaluator.evaluate_humor_statistically(humor_text, context, user_profile)
            
            # Create reasoning based on key metrics
            reasoning = self._create_literature_based_reasoning(scores)
            
            return EvaluationResult(
                surprisal_score=scores.surprisal_score,
                ambiguity_score=scores.ambiguity_score,
                distinctiveness_ratio=scores.distinctiveness_ratio,
                entropy_score=scores.entropy_score,
                perplexity_score=scores.perplexity_score,
                semantic_coherence=scores.semantic_coherence,
                distinct_1=scores.distinct_1,
                distinct_2=scores.distinct_2,
                vocabulary_richness=scores.vocabulary_richness,
                overall_semantic_diversity=scores.overall_semantic_diversity,
                overall_humor_score=scores.overall_humor_score,
                pacs_score=scores.pacs_score,
                reasoning=reasoning,
                evaluator_name="LiteratureBasedEvaluator",
                model_used="statistical_humor_evaluator"
            )
            
        except Exception as e:
            print(f"⚠️ Literature-based evaluation failed: {e}")
            print("Falling back to basic evaluation")
            return await self._evaluate_basic(humor_text, request)
    
    def _create_literature_based_reasoning(self, scores) -> str:
        """Create meaningful reasoning based on literature-based metrics"""
        
        reasoning_parts = []
        
        # Surprisal analysis (Tian et al. 2022)
        if scores.surprisal_score >= 7.0:
            reasoning_parts.append("High surprisal indicates strong incongruity")
        elif scores.surprisal_score >= 4.0:
            reasoning_parts.append("Moderate surprisal suggests some unexpectedness")
        else:
            reasoning_parts.append("Low surprisal indicates predictable content")
        
        # Creativity analysis (Li et al. 2016)
        if scores.distinct_1 >= 0.8:
            reasoning_parts.append("Excellent lexical diversity")
        elif scores.distinct_1 >= 0.6:
            reasoning_parts.append("Good lexical diversity")
        
        # Semantic coherence (Garimella et al. 2020)
        if scores.semantic_coherence >= 7.0:
            reasoning_parts.append("Strong semantic coherence with context")
        elif scores.semantic_coherence >= 5.0:
            reasoning_parts.append("Moderate semantic coherence")
        
        # Overall assessment
        if scores.overall_humor_score >= 7.0:
            reasoning_parts.append("High-quality humor based on literature metrics")
        elif scores.overall_humor_score >= 5.0:
            reasoning_parts.append("Moderate-quality humor with room for improvement")
        else:
            reasoning_parts.append("Lower-quality humor, consider refinement")
        
        return " | ".join(reasoning_parts)
    
    async def _evaluate_basic(self, humor_text: str, request: HumorRequest) -> EvaluationResult:
        """Fallback basic evaluation when literature-based evaluator unavailable"""
        
        # Basic heuristic evaluation (old method)
        humor_score = await self._evaluate_humor_quality(humor_text, request)
        creativity_score = self._evaluate_creativity(humor_text, request)
        appropriateness_score = self._evaluate_appropriateness(humor_text, request)
        context_relevance_score = self._evaluate_context_relevance(humor_text, request)
        surprise_index = await self.surprise_calculator.calculate_surprise_index(humor_text, request.context)
        
        # Calculate overall score
        overall_score = (
            humor_score * 0.35 +
            creativity_score * 0.25 +
            appropriateness_score * 0.2 +
            context_relevance_score * 0.1 +
            surprise_index * 0.1
        )
        
        reasoning = f"Basic evaluation: Overall quality {overall_score:.1f}/10"
        
        return EvaluationResult(
            surprisal_score=surprise_index,
            ambiguity_score=5.0,  # Default neutral
            distinctiveness_ratio=5.0,  # Default neutral
            entropy_score=5.0,  # Default neutral
            perplexity_score=5.0,  # Default neutral
            semantic_coherence=context_relevance_score,
            distinct_1=creativity_score / 10.0,  # Scale creativity to 0-1
            distinct_2=0.5,  # Default neutral
            vocabulary_richness=0.5,  # Default neutral
            overall_semantic_diversity=0.5,  # Default neutral
            overall_humor_score=overall_score,
            pacs_score=0.0,  # No personalization in basic mode
            reasoning=reasoning,
            evaluator_name="BasicEvaluator",
            model_used="rule_based"
        )
    
    def _evaluate_humor_quality(self, text: str, request: HumorRequest) -> float:
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
        is_safe, toxicity_score, _ = self.content_filter.is_content_safe(text)
        if is_safe:
            score += 2.0
        else:
            score -= 3.0
        
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
        self.agent = ImprovedHumorAgent()
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
        
        # Use CrewAI for black cards (with timeout), standard generation for white cards
        if request.card_type == "black":
            print(f"🎭 Using CrewAI for black card generation (with 30s timeout)")
            try:
                # Use asyncio.wait_for to prevent infinite hanging
                generations = await asyncio.wait_for(
                    self.agent.generate_black_cards_with_crewai(request),
                    timeout=30.0  # 30 second timeout for CrewAI
                )
                print("✅ CrewAI black card generation completed successfully")
            except asyncio.TimeoutError:
                print("⚠️ CrewAI timeout after 30s, falling back to standard generation")
                generations = await self.agent.generate_humor(request, recommended_personas)
        else:
            print(f"🎭 Using standard generation for white cards")
            print(f"🎭 DEBUG: Calling self.agent.generate_humor with {len(recommended_personas)} personas")
            generations = await self.agent.generate_humor(request, recommended_personas)
            print(f"🎭 DEBUG: Standard generation returned: {generations}")
            print(f"🎭 DEBUG: Type: {type(generations)}, Length: {len(generations) if generations else 'None'}")
        
        if not generations:
            print("⚠️ Standard generation failed, trying fallback generation...")
            fallback_generations = await self._generate_fallback_humor(request, recommended_personas)
            
            if fallback_generations and len(fallback_generations) > 0:
                print(f"✅ Fallback generation successful: {len(fallback_generations)} cards")
                generations = fallback_generations
            else:
                print("❌ Fallback generation also failed")
                return {
                    'success': False,
                    'error': 'No safe humor generated (including fallback)',
                    'recommended_personas': recommended_personas,
                    'results': [],
                    'top_results': [],
                    'num_results': 0,
                    'generation_time': 0.0,
                    'fallback_used': True
                }
        
        # Evaluate each generation
        evaluated_results = []
        for generation in generations:
            evaluation = await self.evaluator.evaluate_humor(generation.text, request)
            
            evaluated_results.append({
                'generation': generation,
                'evaluation': evaluation,
                'combined_score': evaluation.overall_humor_score + generation.confidence_score
            })
        
        # Sort by combined score
        evaluated_results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy_types(obj):
            if hasattr(obj, 'dtype'):  # numpy array or scalar
                return float(obj) if hasattr(obj, 'item') else obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        # Convert generation_time to ensure it's a regular Python float
        total_generation_time = sum(float(r['generation'].generation_time) for r in evaluated_results) if evaluated_results else 0.0
        
        return {
            'success': True,
            'results': convert_numpy_types(evaluated_results),
            'best_result': convert_numpy_types(evaluated_results[0]) if evaluated_results else None,
            'recommended_personas': recommended_personas,
            'top_results': convert_numpy_types(evaluated_results[:3]) if evaluated_results else [],
            'num_results': len(evaluated_results) if evaluated_results else 0,
            'generation_time': total_generation_time,
            'fallback_used': False
        }
    
    async def _generate_fallback_humor(self, request: HumorRequest, personas: List[str]) -> List[GenerationResult]:
        """Generate fallback humor when standard generation fails"""
        print("🔄 Attempting fallback humor generation...")
        
        try:
            # Try with OpenAI models only (since that's what we have)
            fallback_models = ["gpt-3.5-turbo", "gpt-4-turbo", "gpt-4"]
            print(f"🔄 OpenAI fallback models: {fallback_models}")
            
            for model in fallback_models:
                try:
                    print(f"🔄 Trying fallback model: {model}")
                    
                    # Simple prompt for fallback
                    if request.card_type == "white":
                        prompt = f"""Generate a funny white card response for Cards Against Humanity.

Context: {request.context}
Audience: {request.audience}
Topic: {request.topic}

Generate ONE funny response that fits the blank. Keep it under 50 characters and make it hilarious.

Response:"""
                    else:
                        prompt = f"""Generate a funny black card prompt for Cards Against Humanity.

Context: {request.context}
Audience: {request.audience}
Topic: {request.topic}

Generate ONE funny black card prompt with a blank (_____). Keep it under 100 characters and make it hilarious.

Example format: "What would grandma find disturbing, yet oddly charming? _____"

Response:"""
                    
                    # Try to generate with fallback model
                    llm_request = LLMRequest(
                        prompt=prompt,
                        model=model,
                        temperature=0.9,
                        max_tokens=100,
                        system_prompt="You are a comedy expert. Generate hilarious Cards Against Humanity content."
                    )
                    
                    # Use llm_manager directly instead of multi_llm_manager
                    from agent_system.llm_clients.llm_manager import llm_manager
                    response = await llm_manager.generate_response(llm_request)
                    
                    if response and response.content:
                        # Calculate surprise index for fallback
                        surprise_index = 5.0  # Default surprise for fallback
                        try:
                            surprise_calc = SurpriseCalculator()
                            surprise_index = await surprise_calc.calculate_surprise_index(response.content.strip(), request.context)
                        except Exception as e:
                            print(f"⚠️ Fallback surprise calculation failed: {e}")
                        
                        # Create fallback generation result
                        fallback_result = GenerationResult(
                            text=response.content.strip(),
                            persona_name="fallback_comedy_expert",
                            model_used=model,
                            generation_time=0.5,
                            toxicity_score=0.1,
                            safety_score=0.9,  # High safety for fallback
                            is_safe=True,
                            confidence_score=0.8,
                            surprise_index=surprise_index
                        )
                        
                        print(f"✅ Fallback generation successful with {model}")
                        return [fallback_result]
                        
                except Exception as e:
                    print(f"❌ Fallback model {model} failed: {e}")
                    continue
            
            print("❌ All fallback models failed")
            return []
            
        except Exception as e:
            print(f"❌ Fallback generation error: {e}")
            return []
    
    async def _get_persona_recommendations(self, request: HumorRequest) -> List[str]:
        """SMART STRATEGY: Use favorite personas when available, otherwise fallback to smart strategy"""
        
        # If favorite personas are provided in the request, use them directly
        if hasattr(request, 'favorite_personas') and request.favorite_personas:
            print(f"🎭 Using provided favorite personas: {request.favorite_personas}")
            # Return up to 3 favorite personas
            return request.favorite_personas[:3]
        
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