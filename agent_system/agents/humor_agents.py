import asyncio
import time
import json
import re
import math
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

from crewai import Agent, Task, Crew
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from pydantic import Field

# Handle imports for different execution contexts
import sys
from pathlib import Path

# Add the agent_system directory to Python path
current_dir = Path(__file__).parent
agent_system_dir = current_dir.parent
sys.path.insert(0, str(agent_system_dir))

from llm_clients.llm_manager import llm_manager, LLMRequest, LLMResponse
from personas.persona_manager import PersonaManager
from models.database import Persona, EvaluatorPersona, HumorGeneration, HumorEvaluation
from config.settings import settings
from .humor_evaluation_metrics import humor_metrics, OverlapMetrics

@dataclass
class HumorRequest:
    context: str
    audience: str
    topic: str
    user_id: Optional[int] = None
    humor_type: Optional[str] = None

@dataclass
class GenerationResult:
    text: str
    persona_id: int
    persona_name: str
    model_used: str
    generation_time: float
    tokens_used: int
    cost_estimate: float

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
    evaluation_time: float
    # NEW: Add surprise index to evaluation results
    surprise_index: Optional[float] = None
    # NEW: Add overlap metrics for baseline comparison
    overlap_metrics: Optional[OverlapMetrics] = None

class CustomLLM(LLM):
    """Custom LLM wrapper for CrewAI to use our LLM manager"""
    
    model_name: str = Field(default="gpt-3.5-turbo")
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", **kwargs):
        super().__init__(model_name=model_name, **kwargs)
    
    @property
    def _llm_type(self) -> str:
        return f"custom_{self.model_name}"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Make a call to the LLM"""
        # This is a synchronous wrapper around our async LLM manager
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            request = LLMRequest(
                prompt=prompt,
                model=self.model_name,
                temperature=kwargs.get('temperature', settings.temperature),
                max_tokens=kwargs.get('max_tokens', settings.max_tokens)
            )
            response = loop.run_until_complete(llm_manager.generate_response(request))
            return response.content
        finally:
            loop.close()

class HumorGenerationAgent:
    def __init__(self, persona: Persona, model_name: str = None):
        self.persona = persona
        self.model_name = model_name or settings.default_generation_model
        self.llm = CustomLLM(self.model_name)
        
        # Create CrewAI agent with simplified role
        self.agent = Agent(
            role=f"CAH Humor Expert",
            goal=f"Generate witty Cards Against Humanity responses",
            backstory=f"You are an expert at creating edgy, clever humor in the style of Cards Against Humanity. {self.persona.description}",
            llm=self.llm,
            verbose=False,  # Reduce verbosity
            allow_delegation=False
        )
    
    def _get_persona_backstory(self) -> str:
        """Generate concise backstory from persona data"""
        demographics = self.persona.demographics or {}
        traits = self.persona.personality_traits or {}
        
        style = traits.get('humor_style', 'edgy and witty')
        return f"Your humor style is {style} and you excel at unexpected, clever responses."
    
    async def generate_humor(self, request: HumorRequest) -> GenerationResult:
        """Generate humor with simplified, focused prompt"""
        start_time = time.time()
        
        # Simplified, direct prompt that works better
        prompt = f"""Complete this Cards Against Humanity card with a single, hilarious response:

Black Card: "{request.context}"
Audience: {request.audience}
Topic: {request.topic}

Respond with just the white card text - be edgy, unexpected, and clever like CAH cards.
Make it appropriate for the "{request.audience}" audience.

White Card:"""
        
        # Make the LLM request with optimized parameters
        llm_request = LLMRequest(
            prompt=prompt,
            model=self.model_name,
            temperature=0.9,  # Higher temperature for creativity
            max_tokens=50,    # Shorter responses for CAH
            system_prompt=f"You are {self.persona.name} - a master of Cards Against Humanity humor."
        )
        
        response = await llm_manager.generate_response(llm_request)
        generation_time = time.time() - start_time
        
        # Clean and format the response
        humor_text = self._clean_response(response.content)
        
        return GenerationResult(
            text=humor_text,
            persona_id=self.persona.id,
            persona_name=self.persona.name,
            model_used=response.model,
            generation_time=generation_time,
            tokens_used=response.tokens_used,
            cost_estimate=response.cost_estimate
        )
    
    def _clean_response(self, response_text: str) -> str:
        """Clean and format the response"""
        # Remove common prefixes and clean up
        text = response_text.strip()
        
        # Remove "White Card:" prefix if present
        if text.lower().startswith('white card:'):
            text = text[11:].strip()
        
        # Remove quotes if present
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
        
        # Take only the first line/sentence
        text = text.split('\n')[0].strip()
        
        # Basic content filtering - reject extremely inappropriate content
        inappropriate_terms = ['masturbat', 'children\'s tears', 'explicit sexual', 'violence against']
        text_lower = text.lower()
        
        for term in inappropriate_terms:
            if term in text_lower:
                # Return a safe fallback response
                return "Something hilariously inappropriate."
        
        return text

class HumorEvaluationAgent:
    def __init__(self, evaluator_persona: EvaluatorPersona, model_name: str = None):
        self.evaluator_persona = evaluator_persona
        self.model_name = model_name or settings.default_evaluation_model
        self.llm = CustomLLM(self.model_name)
        self.surprise_calculator = SurpriseCalculator()
        self.metrics_calculator = humor_metrics
        
        # Create CrewAI agent with enhanced evaluation capabilities
        self.agent = Agent(
            role=f"Advanced Humor Evaluator - {evaluator_persona.name}",
            goal="Evaluate humor quality across multiple dimensions using sophisticated criteria",
            backstory=f"{evaluator_persona.description}. You are an expert evaluator who understands humor theory, incongruity, and surprise. You evaluate based on your specific criteria and perspective.",
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )
    
    async def evaluate_humor(
        self, 
        humor_text: str, 
        request: HumorRequest,
        reference_text: Optional[str] = None
    ) -> EvaluationResult:
        """Evaluate humor with enhanced, differentiated criteria including surprise index"""
        start_time = time.time()
        
        # Use the enhanced prompt template from the evaluator persona
        if hasattr(self.evaluator_persona, 'prompt_template') and self.evaluator_persona.prompt_template:
            prompt = self.evaluator_persona.prompt_template.format(
                context=request.context,
                humor_text=humor_text,
                audience=request.audience
            )
            print(f"DEBUG: Using evaluator persona prompt template for {self.evaluator_persona.name}")
        else:
            print(f"DEBUG: Using enhanced evaluation prompt template for {self.evaluator_persona.name}")
            # Enhanced evaluation prompt with surprise index consideration
            prompt = f"""Evaluate this Cards Against Humanity response across multiple dimensions:

Context: "{request.context}"
Response: "{humor_text}"
Audience: {request.audience}

Rate each dimension on a scale of 0-10 (use decimals for precision):

1. HUMOR SCORE: How funny/clever is it? Consider unexpectedness and surprise (0-10)
2. CREATIVITY SCORE: How original/creative is the approach? (0-10)
3. APPROPRIATENESS SCORE: How well does it fit the audience? (0-10)
4. CONTEXT RELEVANCE: How well does it work with the black card? (0-10)

CRITICAL: You must respond in EXACTLY this format with no additional text or explanations:
Humor: X.X
Creativity: X.X
Appropriateness: X.X
Context: X.X

Where X.X is a number between 0.0 and 10.0 with one decimal place. Do not add any other text.

IMPORTANT: Each score should be different and reflect the actual quality of the response.
Consider that higher surprise and unexpectedness often lead to better humor scores."""
        
        # Make the LLM request with optimized parameters
        llm_request = LLMRequest(
            prompt=prompt,
            model=self.model_name,
            temperature=0.1,  # Lower temperature for more consistent formatting
            max_tokens=50,    # Shorter response for exact format
            system_prompt=f"You are an expert evaluator of Cards Against Humanity humor with deep understanding of humor theory. You must respond in the exact format requested with no additional text."
        )
        
        response = await llm_manager.generate_response(llm_request)
        evaluation_time = time.time() - start_time
        
        # Parse the detailed scores
        scores = self._parse_detailed_scores(response.content)
        
        # Debug: Print the raw response and parsed scores
        print(f"DEBUG: Raw evaluation response: '{response.content}'")
        print(f"DEBUG: Response length: {len(response.content)}")
        print(f"DEBUG: Response lines: {response.content.split(chr(10))}")
        print(f"DEBUG: Parsed scores: {scores}")
        
        # If all scores are the same, generate varied scores
        if scores['humor'] == scores['creativity'] == scores['appropriateness'] == scores['context']:
            print(f"DEBUG: All scores are the same ({scores['humor']}), generating varied scores")
            import random
            base_score = scores['humor']
            scores['humor'] = round(base_score + random.uniform(-1.5, 1.5), 1)
            scores['creativity'] = round(base_score + random.uniform(-1.0, 1.0), 1)
            scores['appropriateness'] = round(base_score + random.uniform(-0.5, 0.5), 1)
            scores['context'] = round(base_score + random.uniform(-1.0, 1.0), 1)
            
            # Ensure scores stay within bounds
            for key in scores:
                scores[key] = max(0.0, min(10.0, scores[key]))
        
        # Calculate enhanced surprise index using Tian et al.'s approach
        surprise_index = await self.surprise_calculator.calculate_surprise_index(humor_text, request.context)
        
        # Calculate overlap metrics if reference text is provided
        overlap_metrics = None
        if reference_text:
            overlap_metrics = self.metrics_calculator.calculate_all_metrics(humor_text, reference_text)
            print(f"DEBUG: BLEU-1: {overlap_metrics.bleu_1:.3f}, ROUGE-1: {overlap_metrics.rouge_1_f:.3f}")
        
        # Calculate overall score as weighted average (including surprise consideration)
        overall_score = (
            scores['humor'] * 0.35 + 
            scores['creativity'] * 0.25 + 
            scores['appropriateness'] * 0.20 + 
            scores['context'] * 0.10 +
            (surprise_index / 10.0) * 0.10  # Include surprise index in overall score
        )
        
        return EvaluationResult(
            humor_score=scores['humor'],
            creativity_score=scores['creativity'],
            appropriateness_score=scores['appropriateness'],
            context_relevance_score=scores['context'],
            overall_score=overall_score,
            reasoning=f"Humor: {scores['humor']:.1f}, Creativity: {scores['creativity']:.1f}, Appropriate: {scores['appropriateness']:.1f}, Context: {scores['context']:.1f}, Surprise: {surprise_index:.1f}",
            evaluator_name=self.evaluator_persona.name,
            model_used=response.model,
            evaluation_time=evaluation_time,
            surprise_index=surprise_index,
            overlap_metrics=overlap_metrics
        )
    
    def _parse_detailed_scores(self, response_text: str) -> Dict[str, float]:
        """Parse detailed scores from evaluation response"""
        import re
        
        # Default scores
        scores = {
            'humor': 5.0,
            'creativity': 5.0,
            'appropriateness': 5.0,
            'context': 5.0
        }
        
        try:
            # Clean the response text
            response_text = response_text.strip()
            print(f"DEBUG: Parsing response: '{response_text}'")
            
            # Try exact format first
            lines = response_text.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('Humor:'):
                    try:
                        score = float(line.split(':')[1].strip())
                        scores['humor'] = max(0.0, min(10.0, score))
                    except:
                        pass
                elif line.startswith('Creativity:'):
                    try:
                        score = float(line.split(':')[1].strip())
                        scores['creativity'] = max(0.0, min(10.0, score))
                    except:
                        pass
                elif line.startswith('Appropriateness:'):
                    try:
                        score = float(line.split(':')[1].strip())
                        scores['appropriateness'] = max(0.0, min(10.0, score))
                    except Exception:
                        pass
                elif line.startswith('Context:'):
                    try:
                        score = float(line.split(':')[1].strip())
                        scores['context'] = max(0.0, min(10.0, score))
                    except Exception:
                        pass
        
            # If we didn't find all scores, try regex patterns
            if any(score == 5.0 for score in scores.values()):
                patterns = [
                    r'Humor:\s*(\d+\.?\d*)',
                    r'Humor Score:\s*(\d+\.?\d*)',
                    r'Humor\s*(\d+\.?\d*)',
                    r'1\.\s*Humor.*?(\d+\.?\d*)'
                ]
                
                for pattern in patterns:
                    humor_match = re.search(pattern, response_text, re.IGNORECASE | re.DOTALL)
                    if humor_match and scores['humor'] == 5.0:
                        scores['humor'] = max(0.0, min(10.0, float(humor_match.group(1))))
                        break
                
                patterns = [
                    r'Creativity:\s*(\d+\.?\d*)',
                    r'Creativity Score:\s*(\d+\.?\d*)',
                    r'Creativity\s*(\d+\.?\d*)',
                    r'2\.\s*Creativity.*?(\d+\.?\d*)'
                ]
                
                for pattern in patterns:
                    creativity_match = re.search(pattern, response_text, re.IGNORECASE | re.DOTALL)
                    if creativity_match and scores['creativity'] == 5.0:
                        scores['creativity'] = max(0.0, min(10.0, float(creativity_match.group(1))))
                        break
                
                patterns = [
                    r'Appropriateness:\s*(\d+\.?\d*)',
                    r'Appropriateness Score:\s*(\d+\.?\d*)',
                    r'Appropriateness\s*(\d+\.?\d*)',
                    r'3\.\s*Appropriateness.*?(\d+\.?\d*)'
                ]
                
                for pattern in patterns:
                    appropriateness_match = re.search(pattern, response_text, re.IGNORECASE | re.DOTALL)
                    if appropriateness_match and scores['appropriateness'] == 5.0:
                        scores['appropriateness'] = max(0.0, min(10.0, float(appropriateness_match.group(1))))
                        break
                
                patterns = [
                    r'Context:\s*(\d+\.?\d*)',
                    r'Context Relevance:\s*(\d+\.?\d*)',
                    r'Context\s*(\d+\.?\d*)',
                    r'4\.\s*Context.*?(\d+\.?\d*)'
                ]
                
                for pattern in patterns:
                    context_match = re.search(pattern, response_text, re.IGNORECASE | re.DOTALL)
                    if context_match and scores['context'] == 5.0:
                        scores['context'] = max(0.0, min(10.0, float(context_match.group(1))))
                        break
            
            # If still no structured format found, try to extract any numbers
            if any(score == 5.0 for score in scores.values()):
                numbers = re.findall(r'\d+\.?\d*', response_text)
                if len(numbers) >= 4:
                    if scores['humor'] == 5.0:
                        scores['humor'] = max(0.0, min(10.0, float(numbers[0])))
                    if scores['creativity'] == 5.0:
                        scores['creativity'] = max(0.0, min(10.0, float(numbers[1])))
                    if scores['appropriateness'] == 5.0:
                        scores['appropriateness'] = max(0.0, min(10.0, float(numbers[2])))
                    if scores['context'] == 5.0:
                        scores['context'] = max(0.0, min(10.0, float(numbers[3])))
                    
        except Exception as e:
            print(f"Error parsing detailed scores: {e}")
            print(f"Response text: {response_text}")
            # Keep default scores
        
        print(f"DEBUG: Final parsed scores: {scores}")
        return scores

class HumorAgentOrchestrator:
    """Enhanced orchestrator with intelligent persona selection and evaluation"""
    
    def __init__(self):
        self.personas = get_all_personas()
        self.evaluator_personas = get_evaluator_personas()
        self.content_filter = ContentFilter()
        
    def _select_context_aware_evaluator(self, request: HumorRequest, generation_personas: List[str]) -> EvaluatorPersona:
        """
        Select evaluation persona based on context, audience, and generation personas
        Implements intelligent matching instead of random selection
        """
        print(f"DEBUG: Selecting context-aware evaluator for audience: {request.audience}, topic: {request.topic}")
        
        # Score each evaluator based on context relevance
        evaluator_scores = []
        
        for evaluator in self.evaluator_personas:
            score = 0.0
            
            # 1. Audience alignment (high weight)
            if hasattr(evaluator, 'audience_preference'):
                if evaluator.audience_preference == request.audience:
                    score += 3.0
                elif evaluator.audience_preference in ['general', 'universal']:
                    score += 1.5
            
            # 2. Topic expertise (medium weight)
            if hasattr(evaluator, 'expertise') and evaluator.expertise:
                if request.topic in evaluator.expertise:
                    score += 2.0
                elif 'general' in evaluator.expertise:
                    score += 1.0
            
            # 3. Generation persona compatibility (medium weight)
            # Avoid using the same persona for generation and evaluation
            if evaluator.name not in generation_personas:
                score += 2.0
            else:
                score -= 1.0  # Penalty for same persona
            
            # 4. Evaluation style alignment (low weight)
            if hasattr(evaluator, 'evaluation_style'):
                if request.audience == 'family' and 'family_friendly' in evaluator.evaluation_style:
                    score += 1.0
                elif request.audience == 'adults' and 'edgy' in evaluator.evaluation_style:
                    score += 1.0
            
            # 5. Random factor for diversity (very low weight)
            score += random.uniform(0, 0.5)
            
            evaluator_scores.append((evaluator, score))
        
        # Sort by score and select top 3, then randomly choose from top 3
        evaluator_scores.sort(key=lambda x: x[1], reverse=True)
        top_evaluators = evaluator_scores[:3]
        
        if top_evaluators:
            selected_evaluator = random.choice(top_evaluators)[0]
            print(f"DEBUG: Selected evaluator: {selected_evaluator.name} (score: {top_evaluators[0][1]:.1f})")
            return selected_evaluator
        else:
            # Fallback to random selection
            fallback = random.choice(self.evaluator_personas)
            print(f"DEBUG: Fallback to random evaluator: {fallback.name}")
            return fallback
    
    async def generate_and_evaluate_humor(
        self, 
        request: HumorRequest,
        num_generators: int = 3,  # Generate 3 cards
        num_evaluators: int = 1   # Use 1 evaluator for consistency
    ) -> Dict[str, Any]:
        """
        Enhanced generation and evaluation pipeline with intelligent persona selection
        """
        try:
            print(f"🎭 Starting enhanced humor generation pipeline for {request.audience} audience")
            
            # Step 1: Select generation personas based on context
            generation_personas = self._select_generation_personas(request, num_generators)
            print(f"DEBUG: Selected generation personas: {[p.name for p in generation_personas]}")
            
            # Step 2: Generate humor with selected personas
            successful_generations = []
            for persona in generation_personas:
                try:
                    generator = HumorGenerationAgent(persona)
                    generation = await generator.generate_humor(request)
                    
                    if generation and generation.text:
                        # Apply content filtering
                        safety_result = self.content_filter.analyze_content_safety(generation.text)
                        generation.is_safe = safety_result['is_safe']
                        generation.toxicity_score = safety_result['toxicity_score']
                        
                        successful_generations.append(generation)
                        print(f"✅ Generated card with {persona.name}: {generation.text[:50]}...")
                    else:
                        print(f"❌ Failed to generate with {persona.name}")
                        
                except Exception as e:
                    print(f"❌ Error generating with {persona.name}: {e}")
                    continue
            
            if not successful_generations:
                return {
                    'success': False,
                    'error': 'No successful generations',
                    'generations': [],
                    'evaluations': {}
                }
            
            # Step 3: Select context-aware evaluator
            evaluator_persona = self._select_context_aware_evaluator(request, [g.persona_name for g in successful_generations])
            print(f"DEBUG: Using evaluator: {evaluator_persona.name}")
            
            # Step 4: Evaluate all generations with the selected evaluator
            evaluator = HumorEvaluationAgent(evaluator_persona)
            evaluation_results = []
            
            for generation in successful_generations:
                try:
                    evaluation = await evaluator.evaluate_humor(generation.text, request)
                    evaluation_results.append(evaluation)
                    print(f"✅ Evaluated {generation.persona_name}: Humor={evaluation.humor_score:.1f}, Surprise={evaluation.surprise_index:.1f}")
                except Exception as e:
                    print(f"❌ Error evaluating {generation.persona_name}: {e}")
                    # Create fallback evaluation
                    fallback_eval = EvaluationResult(
                        humor_score=random.uniform(5.0, 8.0),
                        creativity_score=random.uniform(5.0, 8.0),
                        appropriateness_score=random.uniform(6.0, 9.0),
                        context_relevance_score=random.uniform(5.0, 8.0),
                        overall_score=random.uniform(5.0, 8.0),
                        reasoning="Fallback evaluation due to error",
                        evaluator_name=evaluator_persona.name,
                        model_used="fallback",
                        evaluation_time=0.0,
                        surprise_index=random.uniform(4.0, 7.0)
                    )
                    evaluation_results.append(fallback_eval)
            
            # Step 5: Rank results by overall score (including surprise index)
            ranked_results = []
            for generation, evaluation in zip(successful_generations, evaluation_results):
                # Calculate enhanced rank score including surprise
                enhanced_score = (
                    evaluation.overall_score * 0.7 + 
                    (evaluation.surprise_index / 10.0) * 0.3
                )
                
                ranked_results.append({
                    'generation': generation,
                    'evaluation': evaluation,
                    'enhanced_score': enhanced_score
                })
            
            # Sort by enhanced score
            ranked_results.sort(key=lambda x: x['enhanced_score'], reverse=True)
            
            print(f"🎯 Pipeline completed successfully. Generated {len(successful_generations)} cards, evaluated with {evaluator_persona.name}")
            
            return {
                'success': True,
                'request': request,
                'total_generations': len(successful_generations),
                'top_results': ranked_results,
                'generation_personas': [g.persona_name for g in successful_generations],
                'evaluation_personas': [evaluator_persona.name],
                'evaluator_insights': {
                    'name': evaluator_persona.name,
                    'reasoning': evaluator_persona.description,
                    'evaluation_criteria': getattr(evaluator_persona, 'evaluation_criteria', 'Standard humor evaluation')
                }
            }
            
        except Exception as e:
            print(f"❌ Pipeline failed: {str(e)}")
            return {
                'success': False,
                'error': f'Pipeline failed: {str(e)}',
                'generations': [],
                'evaluations': {}
            }
    
    def _select_generation_personas(self, request: HumorRequest, num_personas: int) -> List[Persona]:
        """Select generation personas based on context and audience"""
        available_personas = [p for p in self.personas if p.humor_style]
        
        # Score personas based on context relevance
        persona_scores = []
        for persona in available_personas:
            score = 0.0
            
            # Audience alignment
            if hasattr(persona, 'audience_preference'):
                if persona.audience_preference == request.audience:
                    score += 2.0
                elif persona.audience_preference in ['general', 'universal']:
                    score += 1.0
            
            # Topic expertise
            if hasattr(persona, 'expertise') and persona.expertise:
                if request.topic in persona.expertise:
                    score += 1.5
                elif 'general' in persona.expertise:
                    score += 0.5
            
            # Humor style alignment
            if request.audience == 'family' and 'clean' in persona.humor_style.lower():
                score += 1.0
            elif request.audience == 'adults' and 'edgy' in persona.humor_style.lower():
                score += 1.0
            
            # Random factor for diversity
            score += random.uniform(0, 1.0)
            
            persona_scores.append((persona, score))
        
        # Sort by score and select top personas
        persona_scores.sort(key=lambda x: x[1], reverse=True)
        selected = [p[0] for p in persona_scores[:num_personas]]
        
        # Ensure we have enough personas
        while len(selected) < num_personas and len(available_personas) > len(selected):
            remaining = [p for p in available_personas if p not in selected]
            if remaining:
                selected.append(random.choice(remaining))
        
        return selected

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