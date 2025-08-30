import json
import random
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from sqlalchemy.orm import Session

# Handle imports for different execution contexts
import sys
from pathlib import Path

# Add the agent_system directory to Python path
current_dir = Path(__file__).parent
agent_system_dir = current_dir.parent
sys.path.insert(0, str(agent_system_dir))

try:
    from models.database import Persona, EvaluatorPersona, PersonaPreference, User, UserFeedback
    from config.settings import settings
    from personas.persona_templates import (
        get_all_personas, 
        get_persona_template, 
        recommend_personas_for_context,
        PersonaTemplate
    )
    from personas.dynamic_persona_generator import DynamicPersonaGenerator
except ImportError as e:
    print(f"❌ Import failed in persona_manager: {e}")
    raise

@dataclass
class PersonaProfile:
    name: str
    description: str
    demographics: Dict[str, Any]
    personality_traits: Dict[str, Any]
    expertise_areas: List[str]
    prompt_template: str

class PersonaManager:
    def __init__(self, db: Session):
        self.db = db
        self.dynamic_generator = DynamicPersonaGenerator()
        self._initialize_personas()
    
    def _initialize_personas(self):
        """Initialize personas from templates if they don't exist"""
        existing_personas = self.db.query(Persona).all()
        existing_names = {p.name for p in existing_personas}
        
        # Create personas from templates
        all_templates = get_all_personas()
        
        for persona_key, template in all_templates.items():
            if template.name not in existing_names:
                self._create_persona_from_template(persona_key, template)
        
        # Create default evaluator personas if none exist
        existing_evaluators = self.db.query(EvaluatorPersona).all()
        if not existing_evaluators:
            self._create_default_evaluators()
        else:
            # Update existing evaluator personas with new prompt templates
            self._update_existing_evaluators()
        
        self.db.commit()
    
    def _create_persona_from_template(self, persona_key: str, template: PersonaTemplate):
        """Create a database persona from a template"""
        # Create optimized prompt template for CAH
        prompt_template = f"""You are {template.name} - {template.description}

Style: {template.humor_style}
Expertise: {', '.join(template.expertise_areas)}

{template.prompt_style}

Generate a single hilarious Cards Against Humanity white card response for:
Context: {{context}}
Audience: {{audience}}
Topic: {{topic}}

Examples of your style:
{chr(10).join('• ' + ex for ex in template.examples[:3])}

White Card Response:"""
        
        persona = Persona(
            name=template.name,
            description=template.description,
            demographics=template.demographic_hints,
            personality_traits={
                "humor_style": template.humor_style,
                "expertise_areas": template.expertise_areas
            },
            expertise_areas=template.expertise_areas,
            prompt_template=prompt_template,
            is_active=True
        )
        
        self.db.add(persona)
    
    def _create_default_evaluators(self):
        """Create default evaluator personas"""
        evaluators = [
            {
                "name": "CAH Comedy Judge",
                "description": "Expert at evaluating Cards Against Humanity humor quality",
                "criteria": ["humor", "creativity", "appropriateness", "context_fit"],
                "prompt_template": """Evaluate this Cards Against Humanity response across multiple dimensions:

Context: "{context}"
Response: "{humor_text}"
Audience: {audience}

Rate each dimension on a scale of 0-10 (use decimals for precision):

1. HUMOR SCORE: How funny/clever is it? (0-10)
2. CREATIVITY SCORE: How original/creative is the approach? (0-10)
3. APPROPRIATENESS SCORE: How well does it fit the audience? (0-10)
4. CONTEXT RELEVANCE: How well does it work with the black card? (0-10)

CRITICAL: You must respond in EXACTLY this format with no additional text or explanations:
Humor: X.X
Creativity: X.X
Appropriateness: X.X
Context: X.X

Where X.X is a number between 0.0 and 10.0 with one decimal place. Do not add any other text.

IMPORTANT: Each score should be different and reflect the actual quality of the response."""
            },
            {
                "name": "Humor Quality Expert",
                "description": "Specialist in objective humor quality assessment",
                "criteria": ["humor", "creativity", "appropriateness", "context_fit"],
                "prompt_template": """Evaluate this Cards Against Humanity response across multiple dimensions:

Context: "{context}"
Response: "{humor_text}"
Audience: {audience}

Rate each dimension on a scale of 0-10 (use decimals for precision):

1. HUMOR SCORE: How funny/clever is it? (0-10)
2. CREATIVITY SCORE: How original/creative is the approach? (0-10)
3. APPROPRIATENESS SCORE: How well does it fit the audience? (0-10)
4. CONTEXT RELEVANCE: How well does it work with the black card? (0-10)

CRITICAL: You must respond in EXACTLY this format with no additional text or explanations:
Humor: X.X
Creativity: X.X
Appropriateness: X.X
Context: X.X

Where X.X is a number between 0.0 and 10.0 with one decimal place. Do not add any other text.

IMPORTANT: Each score should be different and reflect the actual quality of the response."""
            }
        ]
        
        for eval_data in evaluators:
            evaluator = EvaluatorPersona(
                name=eval_data["name"],
                description=eval_data["description"],
                evaluation_criteria=eval_data["criteria"],
                prompt_template=eval_data["prompt_template"],
                is_active=True
            )
            self.db.add(evaluator)
    
    def _update_existing_evaluators(self):
        """Update existing evaluator personas with new prompt templates"""
        # Get the new prompt templates
        new_evaluators = [
            {
                "name": "CAH Comedy Judge",
                "description": "Expert at evaluating Cards Against Humanity humor quality",
                "criteria": ["humor", "creativity", "appropriateness", "context_fit"],
                "prompt_template": """Evaluate this Cards Against Humanity response across multiple dimensions:

Context: "{context}"
Response: "{humor_text}"
Audience: {audience}

Rate each dimension on a scale of 0-10 (use decimals for precision):

1. HUMOR SCORE: How funny/clever is it? (0-10)
2. CREATIVITY SCORE: How original/creative is the approach? (0-10)
3. APPROPRIATENESS SCORE: How well does it fit the audience? (0-10)
4. CONTEXT RELEVANCE: How well does it work with the black card? (0-10)

CRITICAL: You must respond in EXACTLY this format with no additional text or explanations:
Humor: X.X
Creativity: X.X
Appropriateness: X.X
Context: X.X

Where X.X is a number between 0.0 and 10.0 with one decimal place. Do not add any other text.

IMPORTANT: Each score should be different and reflect the actual quality of the response."""
            },
            {
                "name": "Humor Quality Expert",
                "description": "Specialist in objective humor quality assessment",
                "criteria": ["humor", "creativity", "appropriateness", "context_fit"],
                "prompt_template": """Evaluate this Cards Against Humanity response across multiple dimensions:

Context: "{context}"
Response: "{humor_text}"
Audience: {audience}

Rate each dimension on a scale of 0-10 (use decimals for precision):

1. HUMOR SCORE: How funny/clever is it? (0-10)
2. CREATIVITY SCORE: How original/creative is the approach? (0-10)
3. APPROPRIATENESS SCORE: How well does it fit the audience? (0-10)
4. CONTEXT RELEVANCE: How well does it work with the black card? (0-10)

CRITICAL: You must respond in EXACTLY this format with no additional text or explanations:
Humor: X.X
Creativity: X.X
Appropriateness: X.X
Context: X.X

Where X.X is a number between 0.0 and 10.0 with one decimal place. Do not add any other text.

IMPORTANT: Each score should be different and reflect the actual quality of the response."""
            }
        ]
        
        # Update existing evaluators
        for eval_data in new_evaluators:
            existing_evaluator = self.db.query(EvaluatorPersona).filter(
                EvaluatorPersona.name == eval_data["name"]
            ).first()
            
            if existing_evaluator:
                existing_evaluator.prompt_template = eval_data["prompt_template"]
                existing_evaluator.evaluation_criteria = eval_data["criteria"]
                print(f"Updated evaluator: {existing_evaluator.name}")
    
    def get_persona_by_key(self, persona_key: str) -> Optional[Persona]:
        """Get persona by template key"""
        template = get_persona_template(persona_key)
        if template:
            return self.db.query(Persona).filter(Persona.name == template.name).first()
        return None
    
    def get_recommended_personas(self, context: str, audience: str, topic: str, count: int = 3) -> List[Persona]:
        """Get recommended personas based on context"""
        recommended_keys = recommend_personas_for_context(context, audience, topic)
        
        personas = []
        for key in recommended_keys[:count]:
            persona = self.get_persona_by_key(key)
            if persona and persona.is_active:
                personas.append(persona)
        
        # Fill up to count with random personas if needed
        if len(personas) < count:
            additional = self.get_random_personas(count - len(personas))
            personas.extend([p for p in additional if p not in personas])
        
        return personas[:count]
    
    def get_random_personas(self, count: int = 3) -> List[Persona]:
        """Get random active personas"""
        all_personas = self.db.query(Persona).filter(Persona.is_active == True).all()
        return random.sample(all_personas, min(count, len(all_personas)))
    
    def get_personas_by_style(self, humor_style: str, count: int = 2) -> List[Persona]:
        """Get personas that match a specific humor style"""
        # Since humor_type doesn't exist in the model, search in personality_traits instead
        matching_personas = self.db.query(Persona).filter(
            Persona.is_active == True
        ).all()
        
        # Filter by humor style in personality_traits
        style_matched = []
        for persona in matching_personas:
            if (persona.personality_traits and 
                isinstance(persona.personality_traits, dict) and
                persona.personality_traits.get('humor_style', '').lower().find(humor_style.lower()) != -1):
                style_matched.append(persona)
        
        if len(style_matched) >= count:
            return random.sample(style_matched, count)
        else:
            # Add random personas to fill the count
            additional_needed = count - len(style_matched)
            additional = self.get_random_personas(additional_needed)
            return style_matched + additional
    
    def get_persona_by_interest(self, interest: str) -> Optional[Persona]:
        """Get a persona that specializes in a specific interest"""
        interest_mapping = {
            "marvel": "marvel_fanatic",
            "superhero": "marvel_fanatic", 
            "comics": "marvel_fanatic",
            "office": "office_worker",
            "work": "office_worker",
            "workplace": "office_worker",
            "food": "foodie_comedian",
            "cooking": "foodie_comedian",
            "gaming": "gaming_guru",
            "games": "gaming_guru",
            "dad jokes": "dad_humor_enthusiast",
            "puns": "wordplay_master",
            "dark": "dark_humor_specialist",
            "absurd": "absurdist_artist"
        }
        
        persona_key = interest_mapping.get(interest.lower())
        if persona_key:
            return self.get_persona_by_key(persona_key)
        return None
    
    def get_evaluator_personas(self, count: int = 1) -> List[EvaluatorPersona]:
        """Get evaluator personas"""
        evaluators = self.db.query(EvaluatorPersona).filter(
            EvaluatorPersona.is_active == True
        ).all()
        return random.sample(evaluators, min(count, len(evaluators)))
    
    def get_generation_personas(self) -> List[Persona]:
        """Get all generation personas"""
        return self.db.query(Persona).filter(Persona.is_active == True).all()
    
    async def get_personalized_personas(self, user_id: int, context: str, count: int = 3) -> List[Persona]:
        """SMART STRATEGY: 2 favorites + 1 dynamic/random for exploration"""
        user_id_str = str(user_id)
        
        # Store user_id for use in _add_random_persona method
        self._current_user_id = user_id
        
        # Get user's persona preferences from database
        from agent_system.models.database import PersonaPreference
        
        persona_preferences = self.db.query(PersonaPreference).filter(
            PersonaPreference.user_id == user_id_str
        ).order_by(PersonaPreference.preference_score.desc()).all()
        
        print(f"DEBUG: Found {len(persona_preferences)} persona preferences for user {user_id_str}")
        
        # Also get interaction history for fallback
        user_interactions = self.db.query(UserFeedback).filter(
            UserFeedback.user_id == user_id_str
        ).order_by(UserFeedback.created_at.desc()).limit(50).all()
        
        print(f"DEBUG: Found {len(user_interactions)} total interactions for user {user_id_str}")
        
        selected_personas = []
        
        # STRATEGY: Get 2 favorite personas (or as many as available)
        favorite_count = min(2, count)
        
        # Get favorite personas from preferences
        if persona_preferences:
            print("DEBUG: Selecting favorite personas from preferences")
            for pref in persona_preferences:
                if len(selected_personas) >= favorite_count:
                    break
                    
                # Get the persona object
                persona = self.db.query(Persona).filter(
                    Persona.id == pref.persona_id,
                    Persona.is_active == True
                ).first()
                
                if persona and pref.preference_score >= 6.0:  # Only use well-rated personas
                    selected_personas.append(persona)
                    print(f"DEBUG: Selected FAVORITE persona: {persona.name} "
                          f"(score: {pref.preference_score:.1f}, interactions: {pref.interaction_count})")
        
        # If we need more favorites, use interaction history analysis
        if len(selected_personas) < favorite_count and user_interactions and len(user_interactions) >= 3:
            print("DEBUG: Supplementing favorites with interaction history analysis")
            
            persona_analysis = {}
            
            for interaction in user_interactions:
                if interaction.feedback_score and interaction.persona_name:
                    score = interaction.feedback_score
                    persona_name = interaction.persona_name
                    
                    # Skip if already selected from preferences
                    if any(p.name == persona_name for p in selected_personas):
                        continue
                    
                    if persona_name not in persona_analysis:
                        persona_analysis[persona_name] = {
                            'scores': [],
                            'total_interactions': 0,
                            'recent_score': 0,
                            'avg_score': 0
                        }
                    
                    persona_analysis[persona_name]['scores'].append(score)
                    persona_analysis[persona_name]['total_interactions'] += 1
                    
                    if len(persona_analysis[persona_name]['scores']) <= 5:
                        persona_analysis[persona_name]['recent_score'] = score
            
            # Calculate weighted preferences from interaction history
            preferred_personas = []
            for persona_name, analysis in persona_analysis.items():
                scores = analysis['scores']
                avg_score = sum(scores) / len(scores)
                total_interactions = analysis['total_interactions']
                recent_score = analysis['recent_score']
                
                frequency_bonus = min(total_interactions / 10.0, 2.0)
                recent_bonus = max(0, (recent_score - 6.0) / 4.0) if recent_score > 0 else 0
                weighted_score = avg_score + frequency_bonus + recent_bonus
                
                if avg_score >= 5.5 or total_interactions >= 3:
                    preferred_personas.append({
                        'name': persona_name,
                        'weighted_score': weighted_score,
                        'avg_score': avg_score,
                        'interactions': total_interactions
                    })
            
            # Sort by weighted score and add to selection
            preferred_personas.sort(key=lambda x: x['weighted_score'], reverse=True)
            
            for persona_data in preferred_personas:
                if len(selected_personas) >= favorite_count:
                    break
                    
                persona = self.db.query(Persona).filter(
                    Persona.name == persona_data['name'],
                    Persona.is_active == True
                ).first()
                
                if persona:
                    selected_personas.append(persona)
                    print(f"DEBUG: Added FAVORITE from history: {persona.name} "
                         f"(weighted: {persona_data['weighted_score']:.2f})")
        
        # STRATEGY: Add 1 DYNAMIC or RANDOM persona for exploration (if we have 3 slots)
        if count >= 3 and len(selected_personas) < count:
            print("DEBUG: Adding DYNAMIC/RANDOM persona for exploration")
            
            # Check if user has enough interactions for dynamic persona generation
            if len(user_interactions) >= 2:
                print("DEBUG: Attempting dynamic persona generation")
                try:
                    # Convert interactions to format expected by dynamic generator
                    interaction_history = []
                    for interaction in user_interactions:
                        interaction_data = {
                            'feedback_score': interaction.feedback_score,
                            'response_text': interaction.response_text or '',
                            'context': interaction.context or '',
                            'topic': 'general',
                            'audience': 'friends',
                            'persona_name': interaction.persona_name or '',
                            'created_at': interaction.created_at.isoformat() if interaction.created_at else None
                        }
                        interaction_history.append(interaction_data)
                    
                    # Try to generate or get existing dynamic persona
                    dynamic_persona = await self.dynamic_generator.get_or_create_persona_for_user(
                        user_id_str, interaction_history
                    )
                    
                    if dynamic_persona:
                        # Convert dynamic persona template to database persona
                        db_persona = self._create_persona_from_dynamic_template(dynamic_persona)
                        if db_persona:
                            selected_personas.append(db_persona)
                            print(f"DEBUG: Added DYNAMIC persona: {db_persona.name}")
                        else:
                            # Fallback to random if dynamic creation failed
                            self._add_random_persona(selected_personas, count)
                    else:
                        # Fallback to random if no dynamic persona
                        self._add_random_persona(selected_personas, count)
                        
                except Exception as e:
                    print(f"DEBUG: Dynamic persona generation failed: {e}")
                    self._add_random_persona(selected_personas, count)
            else:
                # Not enough interactions for dynamic generation, use random
                print("DEBUG: Not enough interactions for dynamic generation, using random")
                self._add_random_persona(selected_personas, count)
        
        # Fill remaining slots with recommended personas if needed
        if len(selected_personas) < count:
            print(f"DEBUG: Filling {count - len(selected_personas)} remaining slots with recommended personas")
            recommended = self.get_recommended_personas(context, "adults", "humor", count * 2)
            
            selected_names = {p.name for p in selected_personas}
            
            # Get disliked personas for filtering
            user_id = getattr(self, '_current_user_id', None)
            disliked_personas = self._get_disliked_persona_names(user_id)
            
            for persona in recommended:
                if (persona.name not in selected_names and 
                    persona.name not in disliked_personas and 
                    len(selected_personas) < count):
                    selected_personas.append(persona)
                    print(f"DEBUG: Added recommended persona: {persona.name}")
        
        # Final fallback - ensure we have enough personas
        if len(selected_personas) < count:
            all_personas = self.db.query(Persona).filter(Persona.is_active == True).all()
            selected_names = {p.name for p in selected_personas}
            
            # Get disliked personas for filtering
            user_id = getattr(self, '_current_user_id', None)
            disliked_personas = self._get_disliked_persona_names(user_id)
            
            for persona in all_personas:
                if (persona.name not in selected_names and 
                    persona.name not in disliked_personas and 
                    len(selected_personas) < count):
                    selected_personas.append(persona)
                    print(f"DEBUG: Added fallback persona: {persona.name}")
        
        print(f"DEBUG: Final selected personas: {[p.name for p in selected_personas]}")
        return selected_personas[:count]
    
    def _get_disliked_persona_names(self, user_id: str) -> set:
        """Get set of disliked persona names for a user (minimal DB query)"""
        if not user_id:
            return set()
        
        # Single efficient query using JOIN
        from agent_system.models.database import PersonaPreference
        result = self.db.query(Persona.name).join(PersonaPreference).filter(
            PersonaPreference.user_id == str(user_id),
            PersonaPreference.preference_score <= 3.0,
            Persona.is_active == True
        ).all()
        
        return {name[0] for name in result}
    
    def _add_random_persona(self, selected_personas: List[Persona], count: int):
        """Add a random persona for exploration"""
        # Get disliked personas for current user
        user_id = getattr(self, '_current_user_id', None)
        disliked_personas = self._get_disliked_persona_names(user_id)
        selected_names = {p.name for p in selected_personas}
        
        # Get available personas excluding selected and disliked ones
        all_personas = self.db.query(Persona).filter(Persona.is_active == True).all()
        available_personas = [
            p for p in all_personas 
            if p.name not in selected_names and p.name not in disliked_personas
        ]
        
        if available_personas:
            random_persona = random.choice(available_personas)
            selected_personas.append(random_persona)
            print(f"DEBUG: Added RANDOM persona for exploration: {random_persona.name}")
        else:
            print(f"DEBUG: No available personas for random selection")
    
    def _create_persona_from_dynamic_template(self, dynamic_template: PersonaTemplate) -> Optional[Persona]:
        """Create a database persona from a dynamic persona template"""
        try:
            # Check if persona already exists
            existing_persona = self.db.query(Persona).filter(
                Persona.name == dynamic_template.name
            ).first()
            
            if existing_persona:
                print(f"DEBUG: Dynamic persona already exists: {existing_persona.name}")
                return existing_persona
            
            # Create optimized prompt template for CAH
            prompt_template = f"""You are {dynamic_template.name} - {dynamic_template.description}

Style: {dynamic_template.humor_style}
Expertise: {', '.join(dynamic_template.expertise_areas)}

{dynamic_template.prompt_style}

Generate a single hilarious Cards Against Humanity white card response for:
Context: {{context}}
Audience: {{audience}}
Topic: {{topic}}

Examples of your style:
{chr(10).join('• ' + ex for ex in dynamic_template.examples[:3])}

White Card Response:"""
            
            persona = Persona(
                name=dynamic_template.name,
                description=dynamic_template.description,
                demographics=dynamic_template.demographic_hints,
                personality_traits={
                    "humor_style": dynamic_template.humor_style,
                    "expertise_areas": dynamic_template.expertise_areas,
                    "is_dynamic": True,
                    "generated_for_user": True
                },
                expertise_areas=dynamic_template.expertise_areas,
                prompt_template=prompt_template,
                is_active=True
            )
            
            self.db.add(persona)
            self.db.commit()
            
            print(f"DEBUG: Created dynamic persona in database: {persona.name}")
            return persona
            
        except Exception as e:
            print(f"DEBUG: Failed to create dynamic persona: {e}")
            return None
    
    def update_persona_performance(self, persona_id: int, performance_score: float):
        """Update persona performance metrics"""
        persona = self.db.query(Persona).filter(Persona.id == persona_id).first()
        if persona:
            # Simple performance tracking
            if not hasattr(persona, 'performance_score'):
                persona.performance_score = performance_score
            else:
                # Moving average
                persona.performance_score = (
                    persona.performance_score * 0.8 + performance_score * 0.2
                )
            self.db.commit()
    
    def get_best_performing_personas(self, count: int = 3) -> List[Persona]:
        """Get the best performing personas"""
        personas = self.db.query(Persona).filter(
            Persona.is_active == True
        ).order_by(
            getattr(Persona, 'performance_score', 7.0).desc()
        ).limit(count).all()
        
        if len(personas) < count:
            # Fill with random if not enough
            additional = self.get_random_personas(count - len(personas))
            personas.extend(additional)
        
        return personas[:count]
    
    def get_persona_by_id(self, persona_id: int) -> Optional[Persona]:
        """Get a specific persona by ID"""
        return self.db.query(Persona).get(persona_id)
    
    def get_evaluator_by_id(self, evaluator_id: int) -> Optional[EvaluatorPersona]:
        """Get a specific evaluator persona by ID"""
        return self.db.query(EvaluatorPersona).get(evaluator_id)
    
    def update_persona_preference(self, user_id: int, persona_id: int, feedback: float, context: str):
        """Update user preference for a persona based on feedback"""
        # Get or create preference record
        preference = self.db.query(PersonaPreference).filter(
            PersonaPreference.user_id == user_id,
            PersonaPreference.persona_id == persona_id
        ).first()
        
        if not preference:
            preference = PersonaPreference(
                user_id=user_id,
                persona_id=persona_id,
                preference_score=0.5,  # Neutral starting point
                interaction_count=0,
                context_preferences={}
            )
            self.db.add(preference)
        
        # Update preference score using weighted average
        old_score = preference.preference_score
        interaction_count = preference.interaction_count
        
        # Calculate new score (weighted average with more weight on recent feedback)
        weight = min(0.3, 1.0 / (interaction_count + 1))  # Decreasing learning rate
        new_score = old_score * (1 - weight) + feedback * weight
        
        preference.preference_score = max(0.0, min(1.0, new_score))  # Clamp between 0 and 1
        preference.interaction_count += 1
        from sqlalchemy import func
        preference.last_interaction = func.now()
        
        # Update context preferences
        if context:
            if not preference.context_preferences:
                preference.context_preferences = {}
            
            if context not in preference.context_preferences:
                preference.context_preferences[context] = []
            
            preference.context_preferences[context].append(feedback)
            # Keep only last 10 feedback items per context
            preference.context_preferences[context] = preference.context_preferences[context][-10:]
        
        self.db.commit()
    
    def create_custom_persona(self, persona_data: PersonaProfile) -> Persona:
        """Create a new custom persona"""
        persona = Persona(
            name=persona_data.name,
            description=persona_data.description,
            demographics=persona_data.demographics,
            personality_traits=persona_data.personality_traits,
            expertise_areas=persona_data.expertise_areas,
            prompt_template=persona_data.prompt_template
        )
        self.db.add(persona)
        self.db.commit()
        return persona 