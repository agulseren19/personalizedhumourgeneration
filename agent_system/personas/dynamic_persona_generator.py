#!/usr/bin/env python3
"""
Dynamic Persona Generator
Creates new personas based on user preferences and behaviors
"""

import asyncio
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import random

# Try relative imports first
try:
    from .enhanced_persona_templates import PersonaTemplate
    from ..llm_clients.multi_llm_manager import multi_llm_manager, LLMRequest, LLMProvider
except ImportError:
    # When running from agent_system directory
    try:
        from personas.enhanced_persona_templates import PersonaTemplate
        from llm_clients.multi_llm_manager import multi_llm_manager, LLMRequest, LLMProvider
    except ImportError:
        # Final fallback - try absolute imports
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from personas.enhanced_persona_templates import PersonaTemplate
        from llm_clients.multi_llm_manager import multi_llm_manager, LLMRequest, LLMProvider

@dataclass
class UserBehaviorProfile:
    """User's behavior profile for persona generation"""
    user_id: str
    preferred_topics: List[str]
    humor_styles: List[str]
    audience_preferences: List[str]
    high_scoring_responses: List[str]
    low_scoring_responses: List[str]
    demographic_hints: Dict[str, Any]
    context_preferences: Dict[str, float]
    
class DynamicPersonaGenerator:
    """Generates personas dynamically based on user preferences"""
    
    def __init__(self):
        self.generated_personas: Dict[str, PersonaTemplate] = {}
        self.user_profiles: Dict[str, UserBehaviorProfile] = {}
        
    async def analyze_user_behavior(self, user_id: str, interaction_history: List[Dict]) -> UserBehaviorProfile:
        """Analyze user behavior to create a behavioral profile"""
        
        # Initialize counters
        topic_counts = {}
        humor_style_counts = {}
        audience_counts = {}
        high_scoring_responses = []
        low_scoring_responses = []
        context_preferences = {}
        
        # Analyze interaction history
        for interaction in interaction_history:
            # Count topics
            topic = interaction.get('topic', 'general')
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
            
            # Count audiences
            audience = interaction.get('audience', 'general')
            audience_counts[audience] = audience_counts.get(audience, 0) + 1
            
            # Analyze scoring patterns
            score = interaction.get('feedback_score', 5)
            response_text = interaction.get('response_text', '')
            
            if score >= 8:
                high_scoring_responses.append(response_text)
            elif score <= 3:
                low_scoring_responses.append(response_text)
            
            # Context preferences
            context = interaction.get('context', '')
            if context:
                context_preferences[context] = context_preferences.get(context, 0) + score
        
        # Extract preferred topics (top 3)
        preferred_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        preferred_topics = [topic for topic, count in preferred_topics]
        
        # Extract audience preferences
        audience_preferences = sorted(audience_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        audience_preferences = [audience for audience, count in audience_preferences]
        
        # Infer humor styles from high-scoring responses
        humor_styles = await self._infer_humor_styles(high_scoring_responses)
        
        # Infer demographic hints
        demographic_hints = self._infer_demographics(interaction_history)
        
        profile = UserBehaviorProfile(
            user_id=user_id,
            preferred_topics=preferred_topics,
            humor_styles=humor_styles,
            audience_preferences=audience_preferences,
            high_scoring_responses=high_scoring_responses[:10],  # Keep top 10
            low_scoring_responses=low_scoring_responses[:10],
            demographic_hints=demographic_hints,
            context_preferences=context_preferences
        )
        
        self.user_profiles[user_id] = profile
        return profile
    
    async def _infer_humor_styles(self, high_scoring_responses: List[str]) -> List[str]:
        """Use LLM to infer humor styles from high-scoring responses"""
        if not high_scoring_responses:
            return ["witty", "clever"]
        
        # Create analysis prompt
        responses_text = "\n".join([f"- {response}" for response in high_scoring_responses[:5]])
        
        prompt = f"""Analyze these high-scoring humor responses and identify the primary humor styles:

{responses_text}

Based on these responses, what are the 3 main humor styles? Choose from:
- witty, sarcastic, punny, absurd, dark, wholesome, nerdy, pop-culture, 
- wordplay, ironic, observational, self-deprecating, edgy, clever, silly

Return only 3 comma-separated humor styles, no explanation."""
        
        try:
            request = LLMRequest(
                prompt=prompt,
                model=LLMProvider.OPENAI_GPT35,
                temperature=0.3,
                max_tokens=50
            )
            
            response = await multi_llm_manager.generate_response(request)
            styles = response.content.strip().split(',')
            return [style.strip() for style in styles[:3]]
            
        except Exception as e:
            print(f"Error inferring humor styles: {e}")
            return ["witty", "clever", "observational"]
    
    def _infer_demographics(self, interaction_history: List[Dict]) -> Dict[str, Any]:
        """Infer demographic hints from interaction patterns"""
        demographics = {}
        
        # Analyze timing patterns
        times = [interaction.get('timestamp') for interaction in interaction_history if interaction.get('timestamp')]
        
        # Analyze topic preferences for age hints
        topics = [interaction.get('topic') for interaction in interaction_history]
        
        if 'gaming' in topics:
            demographics['age_range'] = '18-35'
            demographics['tech_savvy'] = True
        elif 'family' in topics:
            demographics['age_range'] = '25-45'
            demographics['parental_status'] = 'likely_parent'
        elif 'work' in topics:
            demographics['age_range'] = '25-55'
            demographics['professional'] = True
        
        # Analyze audience preferences
        audiences = [interaction.get('audience') for interaction in interaction_history]
        
        if 'family' in audiences:
            demographics['family_oriented'] = True
        elif 'colleagues' in audiences:
            demographics['professional'] = True
        elif 'friends' in audiences:
            demographics['social'] = True
        
        return demographics
    
    async def generate_custom_persona(self, user_id: str, interaction_history: List[Dict]) -> PersonaTemplate:
        """Generate a custom persona based on user behavior"""
        
        # Analyze user behavior
        profile = await self.analyze_user_behavior(user_id, interaction_history)
        
        # Generate persona name
        persona_name = self._generate_persona_name(profile)
        
        # Generate persona using LLM
        persona_data = await self._generate_persona_with_llm(profile, persona_name)
        
        # Create PersonaTemplate
        persona = PersonaTemplate(
            name=persona_data['name'],
            description=persona_data['description'],
            humor_style=persona_data['humor_style'],
            expertise_areas=persona_data['expertise_areas'],
            demographic_hints=profile.demographic_hints,
            prompt_style=persona_data['prompt_style'],
            examples=persona_data['examples']
        )
        
        # Store the generated persona in memory
        self.generated_personas[persona_name] = persona
        
        # FIXED: Save dynamic persona to database automatically
        await self._save_persona_to_database(persona)
        
        return persona
    
    def _generate_persona_name(self, profile: UserBehaviorProfile) -> str:
        """Generate a unique persona name based on user profile"""
        # Combine user preferences into a unique name
        primary_style = profile.humor_styles[0] if profile.humor_styles else "witty"
        primary_topic = profile.preferred_topics[0] if profile.preferred_topics else "general"
        
        # Create persona name with underscores for consistency
        name_parts = []
        
        if primary_style == "witty":
            name_parts.append("clever")
        elif primary_style == "sarcastic":
            name_parts.append("sarcastic")
        elif primary_style == "punny":
            name_parts.append("wordplay")
        else:
            name_parts.append(primary_style.lower().replace(" ", "_"))
        
        if primary_topic == "gaming":
            name_parts.append("gamer")
        elif primary_topic == "work":
            name_parts.append("professional")
        elif primary_topic == "family":
            name_parts.append("family_person")
        else:
            name_parts.append("enthusiast")
        
        # FIXED: Use consistent underscore format, no user suffix to make it stable
        return "_".join(name_parts)
    
    async def _generate_persona_with_llm(self, profile: UserBehaviorProfile, persona_name: str) -> Dict[str, Any]:
        """Use LLM to generate detailed persona based on user profile"""
        
        # Create examples of what the user likes
        liked_examples = "\n".join([f"- {example}" for example in profile.high_scoring_responses[:3]])
        disliked_examples = "\n".join([f"- {example}" for example in profile.low_scoring_responses[:3]])
        
        prompt = f"""Create a humor persona based on this user's preferences:

User Profile:
- Preferred topics: {', '.join(profile.preferred_topics)}
- Humor styles: {', '.join(profile.humor_styles)}
- Audience preferences: {', '.join(profile.audience_preferences)}
- Demographics: {profile.demographic_hints}

Examples they LIKED:
{liked_examples}

Examples they DISLIKED:
{disliked_examples}

Create a persona named "{persona_name}" that would generate humor this user would enjoy.

Return a JSON object with:
{{
  "name": "Human-readable persona name",
  "description": "2-sentence description of the persona",
  "humor_style": "primary humor style in 3-4 words",
  "expertise_areas": ["area1", "area2", "area3"],
  "prompt_style": "How this persona should generate humor",
  "examples": ["example1", "example2", "example3"]
}}"""
        
        try:
            request = LLMRequest(
                prompt=prompt,
                model=LLMProvider.OPENAI_GPT4,
                temperature=0.7,
                max_tokens=400
            )
            
            response = await multi_llm_manager.generate_response(request)
            
            # Parse JSON response
            persona_data = json.loads(response.content.strip())
            
            # Validate required fields
            required_fields = ['name', 'description', 'humor_style', 'expertise_areas', 'prompt_style', 'examples']
            for field in required_fields:
                if field not in persona_data:
                    persona_data[field] = self._get_default_value(field)
            
            return persona_data
            
        except Exception as e:
            print(f"Error generating persona with LLM: {e}")
            return self._get_fallback_persona(profile, persona_name)
    
    def _get_default_value(self, field: str) -> Any:
        """Get default value for missing persona fields"""
        defaults = {
            'name': 'Custom Humor Expert',
            'description': 'A personalized humor expert tailored to your preferences.',
            'humor_style': 'witty and clever',
            'expertise_areas': ['humor', 'wit', 'entertainment'],
            'prompt_style': 'Generate clever, witty responses that match user preferences',
            'examples': ['Something clever', 'Something witty', 'Something funny']
        }
        return defaults.get(field, '')
    
    def _get_fallback_persona(self, profile: UserBehaviorProfile, persona_name: str) -> Dict[str, Any]:
        """Create fallback persona if LLM generation fails"""
        primary_style = profile.humor_styles[0] if profile.humor_styles else "witty"
        primary_topic = profile.preferred_topics[0] if profile.preferred_topics else "general"
        
        return {
            'name': f'{primary_style.title()} {primary_topic.title()} Expert',
            'description': f'A {primary_style} humor expert who specializes in {primary_topic} content.',
            'humor_style': f'{primary_style} and engaging',
            'expertise_areas': [primary_topic, primary_style, 'humor'],
            'prompt_style': f'Generate {primary_style} humor focused on {primary_topic}',
            'examples': [
                f'Something {primary_style}',
                f'Something about {primary_topic}',
                'Something personalized'
            ]
        }
    
    async def get_or_create_persona_for_user(self, user_id: str, interaction_history: List[Dict]) -> PersonaTemplate:
        """Get existing persona or create new one for user"""
        
        # Check if we already have a persona for this user
        existing_persona_name = None
        for persona_name, persona in self.generated_personas.items():
            if user_id in persona_name:
                existing_persona_name = persona_name
                break
        
        if existing_persona_name:
            return self.generated_personas[existing_persona_name]
        
        # ENHANCED: Create new persona with lower threshold
        if len(interaction_history) >= 2:  # Changed from 3 to 2
            return await self.generate_custom_persona(user_id, interaction_history)
        else:
            # Return a basic persona for new users
            return self._create_basic_persona(user_id)
    
    def _create_basic_persona(self, user_id: str) -> PersonaTemplate:
        """Create a basic persona for new users"""
        return PersonaTemplate(
            name=f"Adaptive Humor Expert for {user_id}",
            description="A versatile humor expert that adapts to your preferences over time.",
            humor_style="adaptive and clever",
            expertise_areas=["humor", "wit", "entertainment"],
            demographic_hints={},
            prompt_style="Generate clever, adaptable humor that learns from feedback",
            examples=[
                "Something unexpectedly clever",
                "Something contextually appropriate",
                "Something surprisingly funny"
            ]
        )
    
    async def evolve_persona(self, persona_name: str, new_interactions: List[Dict]) -> PersonaTemplate:
        """Evolve an existing persona based on new interactions"""
        
        if persona_name not in self.generated_personas:
            return None
        
        current_persona = self.generated_personas[persona_name]
        
        # Analyze new interactions
        user_id = new_interactions[0].get('user_id') if new_interactions else None
        if not user_id:
            return current_persona
        
        # Update user profile with new interactions
        all_interactions = []
        if user_id in self.user_profiles:
            # Would need to get historical interactions from database
            pass
        
        all_interactions.extend(new_interactions)
        
        # Re-analyze and update persona
        updated_profile = await self.analyze_user_behavior(user_id, all_interactions)
        updated_persona_data = await self._generate_persona_with_llm(updated_profile, persona_name)
        
        # Update the persona
        updated_persona = PersonaTemplate(
            name=updated_persona_data['name'],
            description=updated_persona_data['description'],
            humor_style=updated_persona_data['humor_style'],
            expertise_areas=updated_persona_data['expertise_areas'],
            demographic_hints=updated_profile.demographic_hints,
            prompt_style=updated_persona_data['prompt_style'],
            examples=updated_persona_data['examples']
        )
        
        self.generated_personas[persona_name] = updated_persona
        return updated_persona
    
    def get_all_personas(self) -> Dict[str, PersonaTemplate]:
        """Get all generated personas"""
        return self.generated_personas
    
    def get_persona_by_name(self, name: str) -> Optional[PersonaTemplate]:
        """Get specific persona by name"""
        return self.generated_personas.get(name)

    async def _save_persona_to_database(self, persona: PersonaTemplate):
        """Save a dynamically generated persona to the database"""
        try:
            from agent_system.models.database import get_session_local, Persona
            from agent_system.config.settings import settings
            
            SessionLocal = get_session_local(settings.database_url)
            db = SessionLocal()
            
            try:
                # Check if persona already exists
                existing = db.query(Persona).filter(Persona.name == persona.name).first()
                
                if not existing:
                    # Create new persona in database
                    new_persona = Persona(
                        name=persona.name,
                        description=persona.description,
                        demographics=persona.demographic_hints,
                        personality_traits={
                            "humor_style": persona.humor_style,
                            "is_dynamic": True,
                            "is_ai_comedian": True
                        },
                        expertise_areas=persona.expertise_areas,
                        prompt_template=persona.prompt_style,
                        is_active=True
                    )
                    
                    db.add(new_persona)
                    db.commit()
                    print(f"✅ Saved dynamic persona to database: {persona.name}")
                else:
                    print(f"ℹ️  Dynamic persona already exists in database: {persona.name}")
                    
            finally:
                db.close()
                
        except Exception as e:
            print(f"❌ Error saving dynamic persona to database: {e}")

# Global instance
dynamic_persona_generator = DynamicPersonaGenerator() 