#!/usr/bin/env python3
"""
Test Dynamic Persona Generation System
Demonstrates how personas are created dynamically based on user preferences
"""

import asyncio
from datetime import datetime
from typing import Dict, List

from agent_system.personas.dynamic_persona_generator import dynamic_persona_generator
from agent_system.knowledge.improved_aws_knowledge_base import improved_aws_knowledge_base

async def test_dynamic_persona_generation():
    """Test the dynamic persona generation system"""
    
    print("DYNAMIC PERSONA GENERATION TEST")
    print("=" * 50)
    
    # Create sample user interaction history
    user_id = "test_user_123"
    
    # Sample interactions showing user preferences
    interaction_history = [
        {
            'user_id': user_id,
            'persona_name': 'millennial_memer',
            'context': 'What did I do during the pandemic? _____',
            'response_text': 'Learned sourdough and got existential dread',
            'topic': 'lifestyle',
            'audience': 'friends',
            'feedback_score': 9.0,
            'timestamp': datetime.now().isoformat()
        },
        {
            'user_id': user_id,
            'persona_name': 'gaming_guru',
            'context': 'What\'s my secret weapon? _____',
            'response_text': 'Muscle memory from 10,000 hours of Tetris',
            'topic': 'gaming',
            'audience': 'friends',
            'feedback_score': 8.5,
            'timestamp': datetime.now().isoformat()
        },
        {
            'user_id': user_id,
            'persona_name': 'dad_humor_enthusiast',
            'context': 'What makes me laugh? _____',
            'response_text': 'Dad jokes so bad they wrap around to good',
            'topic': 'humor',
            'audience': 'family',
            'feedback_score': 3.0,
            'timestamp': datetime.now().isoformat()
        },
        {
            'user_id': user_id,
            'persona_name': 'millennial_memer',
            'context': 'What\'s my biggest fear? _____',
            'response_text': 'Running out of avocado toast',
            'topic': 'lifestyle',
            'audience': 'friends',
            'feedback_score': 8.0,
            'timestamp': datetime.now().isoformat()
        },
        {
            'user_id': user_id,
            'persona_name': 'office_worker',
            'context': 'What do I do at work? _____',
            'response_text': 'Pretend to understand Excel formulas',
            'topic': 'work',
            'audience': 'colleagues',
            'feedback_score': 7.5,
            'timestamp': datetime.now().isoformat()
        },
        {
            'user_id': user_id,
            'persona_name': 'gaming_guru',
            'context': 'What\'s my superpower? _____',
            'response_text': 'Knowing every Pokémon by their cry',
            'topic': 'gaming',
            'audience': 'friends',
            'feedback_score': 9.5,
            'timestamp': datetime.now().isoformat()
        }
    ]
    
    print(f"User: {user_id}")
    print(f"Interaction history: {len(interaction_history)} interactions")
    print()
    
    # Test 1: Analyze user behavior
    print("1. ANALYZING USER BEHAVIOR")
    print("-" * 30)
    
    behavior_profile = await dynamic_persona_generator.analyze_user_behavior(
        user_id, interaction_history
    )
    
    print(f"Preferred topics: {behavior_profile.preferred_topics}")
    print(f"Humor styles: {behavior_profile.humor_styles}")
    print(f"Audience preferences: {behavior_profile.audience_preferences}")
    print(f"High-scoring responses: {len(behavior_profile.high_scoring_responses)}")
    print(f"Low-scoring responses: {len(behavior_profile.low_scoring_responses)}")
    print(f"Demographic hints: {behavior_profile.demographic_hints}")
    print()
    
    # Test 2: Generate custom persona
    print("2. GENERATING CUSTOM PERSONA")
    print("-" * 30)
    
    custom_persona = await dynamic_persona_generator.generate_custom_persona(
        user_id, interaction_history
    )
    
    print(f"Persona name: {custom_persona.name}")
    print(f"Description: {custom_persona.description}")
    print(f"Humor style: {custom_persona.humor_style}")
    print(f"Expertise areas: {custom_persona.expertise_areas}")
    print(f"Prompt style: {custom_persona.prompt_style}")
    print(f"Example responses: {custom_persona.examples}")
    print()
    
    # Test 3: Show persona evolution
    print("3. PERSONA EVOLUTION")
    print("-" * 30)
    
    # Add more interactions
    new_interactions = [
        {
            'user_id': user_id,
            'persona_name': 'tech_savvy_millennial',
            'context': 'What broke my phone? _____',
            'response_text': 'Trying to take a selfie with my cat',
            'topic': 'technology',
            'audience': 'friends',
            'feedback_score': 8.0,
            'timestamp': datetime.now().isoformat()
        },
        {
            'user_id': user_id,
            'persona_name': 'streaming_enthusiast',
            'context': 'What did I binge-watch last night? _____',
            'response_text': 'TikTok compilations of people falling',
            'topic': 'entertainment',
            'audience': 'friends',
            'feedback_score': 7.0,
            'timestamp': datetime.now().isoformat()
        }
    ]
    
    all_interactions = interaction_history + new_interactions
    
    evolved_persona = await dynamic_persona_generator.generate_custom_persona(
        user_id + "_evolved", all_interactions
    )
    
    print(f"Original persona: {custom_persona.name}")
    print(f"Evolved persona: {evolved_persona.name}")
    print(f"New humor style: {evolved_persona.humor_style}")
    print(f"New expertise: {evolved_persona.expertise_areas}")
    print()
    
    # Test 4: Multiple users with different preferences
    print("4. DIFFERENT USER PREFERENCES")
    print("-" * 30)
    
    # User 2: Professional/Work-focused
    professional_user = "business_user_456"
    professional_history = [
        {
            'user_id': professional_user,
            'persona_name': 'office_worker',
            'context': 'What\'s in my briefcase? _____',
            'response_text': 'Three empty coffee cups and crushed dreams',
            'topic': 'work',
            'audience': 'colleagues',
            'feedback_score': 8.5,
            'timestamp': datetime.now().isoformat()
        },
        {
            'user_id': professional_user,
            'persona_name': 'corporate_ladder_climber',
            'context': 'What\'s my management style? _____',
            'response_text': 'Aggressive delegation and strategic coffee breaks',
            'topic': 'work',
            'audience': 'colleagues',
            'feedback_score': 9.0,
            'timestamp': datetime.now().isoformat()
        },
        {
            'user_id': professional_user,
            'persona_name': 'millennial_memer',
            'context': 'What\'s trending? _____',
            'response_text': 'Memes about student loans',
            'topic': 'culture',
            'audience': 'colleagues',
            'feedback_score': 4.0,
            'timestamp': datetime.now().isoformat()
        }
    ]
    
    professional_persona = await dynamic_persona_generator.generate_custom_persona(
        professional_user, professional_history
    )
    
    print(f"Professional user persona: {professional_persona.name}")
    print(f"Humor style: {professional_persona.humor_style}")
    print(f"Expertise: {professional_persona.expertise_areas}")
    print()
    
    # Test 5: Show all generated personas
    print("5. ALL GENERATED PERSONAS")
    print("-" * 30)
    
    all_personas = dynamic_persona_generator.get_all_personas()
    print(f"Total generated personas: {len(all_personas)}")
    
    for persona_name, persona in all_personas.items():
        print(f"  • {persona.name} - {persona.humor_style}")
    
    print()
    print("DYNAMIC PERSONA GENERATION TEST COMPLETE!")
    print("=" * 50)

async def test_integration_with_knowledge_base():
    """Test integration with the improved knowledge base"""
    
    print("\nINTEGRATION TEST WITH KNOWLEDGE BASE")
    print("=" * 50)
    
    user_id = "integration_test_user"
    
    # Simulate some feedback through the knowledge base
    await improved_aws_knowledge_base.update_user_feedback(
        user_id=user_id,
        persona_name="gaming_guru",
        feedback_score=9.0,
        context="What's my gaming setup? _____",
        response_text="RGB lights and disappointment",
        topic="gaming",
        audience="friends"
    )
    
    await improved_aws_knowledge_base.update_user_feedback(
        user_id=user_id,
        persona_name="tech_enthusiast",
        feedback_score=8.5,
        context="What did I upgrade? _____",
        response_text="My graphics card and my debt",
        topic="technology",
        audience="friends"
    )
    
    await improved_aws_knowledge_base.update_user_feedback(
        user_id=user_id,
        persona_name="dad_humor_enthusiast",
        feedback_score=3.0,
        context="What's funny? _____",
        response_text="Puns about computer programming",
        topic="humor",
        audience="family"
    )
    
    # Get user preferences
    user_pref = await improved_aws_knowledge_base.get_user_preference(user_id)
    
    if user_pref:
        print(f"User preferences for {user_id}:")
        print(f"  Liked personas: {user_pref.liked_personas}")
        print(f"  Disliked personas: {user_pref.disliked_personas}")
        print(f"  Interaction history: {len(user_pref.interaction_history)} interactions")
        
        # Create dynamic persona from knowledge base data
        if len(user_pref.interaction_history) >= 3:
            dynamic_persona = await dynamic_persona_generator.get_or_create_persona_for_user(
                user_id, user_pref.interaction_history
            )
            
            print(f"\nGenerated persona from knowledge base:")
            print(f"  Name: {dynamic_persona.name}")
            print(f"  Style: {dynamic_persona.humor_style}")
            print(f"  Expertise: {dynamic_persona.expertise_areas}")
        else:
            print(f"\nNot enough interactions for dynamic persona generation")
    else:
        print(f"No user preferences found for {user_id}")

async def main():
    """Run all tests"""
    await test_dynamic_persona_generation()
    await test_integration_with_knowledge_base()

if __name__ == "__main__":
    asyncio.run(main()) 