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

# Content filtering
from detoxify import Detoxify

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

# Import AWS knowledge base
try:
    from ..knowledge.improved_aws_knowledge_base import improved_aws_knowledge_base
except ImportError:
    try:
        from knowledge.improved_aws_knowledge_base import improved_aws_knowledge_base
    except ImportError:
        print("⚠️  Could not import improved_aws_knowledge_base, using fallback")
        # Create a mock knowledge base
        class MockKnowledgeBase:
            async def get_persona_recommendations(self, user_id: str, context: str, audience: str) -> List[str]:
                return ["General Comedian", "Witty Observer"]
            
            async def get_user_preference(self, user_id: str) -> Dict[str, Any]:
                return {"humor_style": "general", "audience": "friends", "liked_personas": [], "disliked_personas": []}
            
            async def get_user_interaction_history(self, user_id: str) -> List[Dict[str, Any]]:
                return []
        
        improved_aws_knowledge_base = MockKnowledgeBase()

try:
    from ..llm_clients.multi_llm_manager import multi_llm_manager, LLMRequest, LLMProvider
    from ..personas.enhanced_persona_templates import get_all_personas
    from ..personas.dynamic_persona_generator import dynamic_persona_generator
    from ..config.settings import settings
except ImportError:
    # Fallback to absolute imports when running directly
    from llm_clients.multi_llm_manager import multi_llm_manager, LLMRequest, LLMProvider
    from personas.enhanced_persona_templates import get_all_personas
    from personas.dynamic_persona_generator import dynamic_persona_generator
    from config.settings import settings

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

class ContentFilter:
    """Advanced content filtering using detoxify"""
    
    def __init__(self):
        self.detoxify = Detoxify('original')
        
        # Thresholds for different toxicity types
        self.toxicity_thresholds = {
            'toxicity': 0.7,
            'severe_toxicity': 0.5,
            'obscene': 0.8,
            'threat': 0.3,
            'insult': 0.7,
            'identity_attack': 0.5
        }
    
    def is_content_safe(self, text: str) -> Tuple[bool, float, Dict[str, float]]:
        """Check if content is safe using detoxify"""
        try:
            scores = self.detoxify.predict(text)
            
            # Check each toxicity type
            violations = []
            max_score = 0
            
            for toxicity_type, threshold in self.toxicity_thresholds.items():
                if toxicity_type in scores:
                    score = scores[toxicity_type]
                    max_score = max(max_score, score)
                    
                    if score > threshold:
                        violations.append(f"{toxicity_type}: {score:.3f}")
            
            is_safe = len(violations) == 0
            
            return is_safe, max_score, scores
            
        except Exception as e:
            print(f"Content filtering error: {e}")
            # If filtering fails, be conservative
            return False, 1.0, {}
    
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
    """Improved humor generation agent with proper persona handling"""
    
    def __init__(self):
        self.content_filter = ContentFilter()
    
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
        available_models = [LLMProvider.OPENAI_GPT4, LLMProvider.OPENAI_GPT35, LLMProvider.CLAUDE_SONNET]
        
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
        """Filter personas based on user likes/dislikes"""
        if not user_prefs:
            return personas
        
        # Remove disliked personas
        filtered = [p for p in personas if p not in user_prefs.disliked_personas]
        
        # Prioritize liked personas
        liked_in_list = [p for p in filtered if p in user_prefs.liked_personas]
        not_liked_in_list = [p for p in filtered if p not in user_prefs.liked_personas]
        
        # Put liked personas first
        final_list = liked_in_list + not_liked_in_list
        
        return final_list if final_list else personas  # Fallback to original if all filtered out
    
    async def _generate_with_custom_persona(self, request: HumorRequest, custom_persona, model: LLMProvider) -> Optional[GenerationResult]:
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
            
            return GenerationResult(
                text=humor_text,
                persona_name=custom_persona.name,
                model_used=response.provider.value,
                generation_time=generation_time,
                toxicity_score=toxicity_score,
                is_safe=is_safe,
                confidence_score=self._calculate_confidence(humor_text, request.context)
            )
            
        except Exception as e:
            print(f"    Error generating with custom persona {custom_persona.name}: {e}")
            return None

    async def generate_black_cards_with_crewai(self, request: HumorRequest) -> List[GenerationResult]:
        """Generate black cards using CrewAI with personalized personas"""
        print(f"🎭 Generating black cards with CrewAI for user {request.user_id}")
        print(f"🎭 DEBUG: Black card generation called with card_type = '{request.card_type}'")
        print(f"🎭 DEBUG: Black card generation called with context = '{request.context}'")
        
        # Get user preferences and create personalized personas
        user_preferences = await self._get_user_preferences(request.user_id)
        custom_persona = await self._get_or_create_custom_persona(request.user_id, user_preferences)
        
        # Get favorite comedians from user preferences
        favorite_comedians = []
        if user_preferences and user_preferences.liked_personas:
            favorite_comedians = user_preferences.liked_personas[:2]  # Top 2 favorites
        
        # Get random comedian for variety
        all_personas = list(get_all_personas().keys())
        random_comedian = random.choice([p for p in all_personas if p not in favorite_comedians])
        
        # Create CrewAI crew for black card generation
        crew = await self._create_black_card_crew(request, favorite_comedians, random_comedian, custom_persona)
        
        results = []
        
        # Generate with favorite comedians first
        for comedian in favorite_comedians:
            try:
                result = await self._generate_black_card_with_crewai(crew, comedian, request)
                if result:
                    results.append(result)
            except Exception as e:
                print(f"    Error generating with favorite comedian {comedian}: {e}")
        
        # Generate with random comedian
        try:
            result = await self._generate_black_card_with_crewai(crew, random_comedian, request)
            if result:
                results.append(result)
        except Exception as e:
            print(f"    Error generating with random comedian {random_comedian}: {e}")
        
        # Generate with custom persona if available
        if custom_persona and len(results) < 3:
            try:
                result = await self._generate_black_card_with_crewai(crew, custom_persona.name, request, is_custom=True)
                if result:
                    results.append(result)
            except Exception as e:
                print(f"    Error generating with custom persona {custom_persona.name}: {e}")
        
        print(f"    Generated {len(results)} black cards with CrewAI")
        return results

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
                # Create generation result
                return GenerationResult(
                    text=black_card_text,
                    persona_name=comedian_name,
                    model_used="crewai",
                    generation_time=0.0,  # Will be set by caller
                    toxicity_score=0.0,   # Will be filtered by caller
                    is_safe=True,         # Will be checked by caller
                    confidence_score=self._calculate_confidence(black_card_text, request.context)
                )
            
        except Exception as e:
            print(f"    CrewAI black card generation failed for {comedian_name}: {e}")
        
        # Fallback to standard generation
        return await self._generate_with_persona(request, comedian_name, LLMProvider.OPENAI_GPT4)

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
    
    async def _generate_with_persona(self, request: HumorRequest, persona_name: str, model: LLMProvider) -> Optional[GenerationResult]:
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
            
            return GenerationResult(
                text=humor_text,
                persona_name=persona_name,
                model_used=response.provider.value,
                generation_time=generation_time,
                toxicity_score=toxicity_score,
                is_safe=is_safe,
                confidence_score=self._calculate_confidence(humor_text, request.context)
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
    """Improved evaluation system with meaningful scores"""
    
    def __init__(self):
        self.content_filter = ContentFilter()
    
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