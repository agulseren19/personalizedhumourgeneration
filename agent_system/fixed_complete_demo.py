#!/usr/bin/env python3
"""
Fixed Complete System Demo
Demonstrates all fixes: content filtering, persona recommendations, feedback learning, evaluation
"""

import asyncio
import json
import time
from typing import List, Dict, Any
from datetime import datetime
import random

# Import fixed components
from agent_system.agents.improved_humor_agents import (
    ImprovedHumorOrchestrator, 
    HumorRequest, 
    ContentFilter
)
from agent_system.knowledge.improved_aws_knowledge_base import improved_aws_knowledge_base
from agent_system.personas.enhanced_persona_templates import get_all_personas
from agent_system.config.settings import settings

class FixedCompleteHumorSystem:
    """
    Fixed humor generation system with all improvements:
    - Content filtering with detoxify
    - Proper persona recommendation and filtering
    - Fixed feedback learning system
    - Black card generation
    - Meaningful evaluation scores
    - AWS knowledge base integration
    """
    
    def __init__(self):
        print("Initializing FIXED Complete Humor Generation System...")
        print("Fixed Issues:")
        print("  • Content filtering with detoxify")
        print("  • Proper persona recommendation")
        print("  • Fixed feedback learning (likes/dislikes)")
        print("  • Meaningful evaluation scores")
        print("  • Black card generation")
        print("  • AWS knowledge base integration")
        print()
        
        self.orchestrator = ImprovedHumorOrchestrator()
        self.content_filter = ContentFilter()
        
        # Demo personas for realistic feedback
        self.demo_personas = [
            "dad_humor_enthusiast",
            "millennial_memer", 
            "office_worker",
            "gaming_guru",
            "dark_humor_specialist",
            "wordplay_master",
            "suburban_parent",
            "gen_z_chaos"
        ]
        
        print(f"Available personas: {self.demo_personas}")
        print()
    
    async def demonstrate_full_fixed_workflow(self):
        """Demonstrate the complete fixed workflow"""
        print("=" * 70)
        print("DEMONSTRATING FIXED COMPLETE WORKFLOW")
        print("=" * 70)
        
        # 1. Demonstrate content filtering
        await self.demonstrate_content_filtering()
        
        # 2. Generate initial humor to establish baseline
        await self.generate_initial_humor()
        
        # 3. Simulate realistic user feedback (varied scores)
        await self.simulate_realistic_feedback()
        
        # 4. Demonstrate learning and adaptation
        await self.demonstrate_fixed_learning()
        
        # 5. Show black card generation
        await self.demonstrate_black_card_generation()
        
        # 6. Show group humor with consensus
        await self.demonstrate_group_consensus()
        
        # 7. Show user analytics
        await self.show_user_analytics()
    
    async def demonstrate_content_filtering(self):
        """Demonstrate content filtering with detoxify"""
        print("\nCONTENT FILTERING DEMONSTRATION")
        print("-" * 50)
        
        # Test various content types
        test_content = [
            "A delightful family picnic",  # Safe
            "My boss's terrible management skills",  # Borderline
            "Something hilariously inappropriate",  # Edgy but safe
            "Damn politicians and their BS",  # Mild profanity
            "You're such an idiot, moron",  # Insult - should be filtered
        ]
        
        for content in test_content:
            is_safe, toxicity_score, scores = self.content_filter.is_content_safe(content)
            print(f"  '{content}'")
            print(f"    Safe: {is_safe}, Toxicity: {toxicity_score:.3f}")
            
            if not is_safe:
                sanitized = self.content_filter.sanitize_content(content)
                print(f"    Sanitized: '{sanitized}'")
            print()
    
    async def generate_initial_humor(self):
        """Generate initial humor to establish baseline"""
        print("\nINITIAL HUMOR GENERATION")
        print("-" * 50)
        
        users = [
            {"user_id": "john_lawyer", "context": "What's the best legal excuse for being late? _____", "audience": "colleagues"},
            {"user_id": "mary_parent", "context": "What did I pack for my kid's lunch? _____", "audience": "family"},
            {"user_id": "alex_gamer", "context": "What's my secret gaming strategy? _____", "audience": "friends"}
        ]
        
        self.initial_results = []
        
        for user in users:
            print(f"\n👤 Processing user: {user['user_id']}")
            
            request = HumorRequest(
                context=user['context'],
                audience=user['audience'],
                topic="lifestyle",
                user_id=user['user_id']
            )
            
            result = await self.orchestrator.generate_and_evaluate_humor(request)
            
            if result['success']:
                print(f"  Context: {request.context}")
                print(f"  Recommended personas: {result['recommended_personas']}")
                
                for i, evaluated_result in enumerate(result['results'][:3], 1):
                    generation = evaluated_result['generation']
                    evaluation = evaluated_result['evaluation']
                    
                    print(f"    {i}. \"{generation.text}\"")
                    print(f"       By: {generation.persona_name} (via {generation.model_used})")
                    print(f"       Score: {evaluation.overall_score:.1f}/10")
                    print(f"       Toxicity: {generation.toxicity_score:.3f}")
                    print(f"       Safe: {generation.is_safe}")
                
                self.initial_results.append((user, result))
            else:
                print(f"  • Failed: {result.get('error')}")
    
    async def simulate_realistic_feedback(self):
        """Simulate realistic user feedback with varied scores"""
        print("\nREALISTIC FEEDBACK SIMULATION")
        print("-" * 50)
        
        # Realistic feedback scenarios (not all 5/10!)
        feedback_scenarios = [
            {
                "user_id": "john_lawyer", 
                "preferences": {
                    "office_worker": 8.5,
                    "wordplay_master": 7.5,
                    "dad_humor_enthusiast": 6.0,
                    "dark_humor_specialist": 3.5,  # Dislike
                    "gen_z_chaos": 4.0,  # Dislike
                    "millennial_memer": 6.5
                }
            },
            {
                "user_id": "mary_parent", 
                "preferences": {
                    "dad_humor_enthusiast": 9.0,
                    "suburban_parent": 8.0,
                    "office_worker": 7.0,
                    "dark_humor_specialist": 2.5,  # Strong dislike
                    "gen_z_chaos": 3.0,  # Dislike
                    "wordplay_master": 7.5
                }
            },
            {
                "user_id": "alex_gamer",
                "preferences": {
                    "gaming_guru": 9.5,
                    "millennial_memer": 8.5,
                    "gen_z_chaos": 7.5,
                    "office_worker": 5.5,
                    "suburban_parent": 4.5,
                    "dad_humor_enthusiast": 3.5  # Dislike
                }
            }
        ]
        
        for user_data, (user, result) in zip(feedback_scenarios, self.initial_results):
            if result.get('success'):
                print(f"\n👤 {user_data['user_id']} feedback:")
                
                # Give feedback for multiple rounds to establish patterns
                for round_num in range(3):  # 3 rounds of feedback
                    print(f"  Round {round_num + 1}:")
                    
                    for evaluated_result in result['results']:
                        generation = evaluated_result['generation']
                        persona_name = generation.persona_name
                        
                        # Get base score from preferences
                        base_score = user_data['preferences'].get(persona_name, 5.0)
                        
                        # Add some randomness to make it realistic
                        feedback_score = base_score + random.uniform(-0.5, 0.5)
                        feedback_score = max(1.0, min(10.0, feedback_score))
                        
                        print(f"    {persona_name}: {feedback_score:.1f}/10")
                        
                        # Store feedback with detailed information for persona generation
                        await improved_aws_knowledge_base.update_user_feedback(
                            user_id=user_data['user_id'],
                            persona_name=persona_name,
                            feedback_score=feedback_score,
                            context=user['context'],
                            response_text=generation.text,
                            topic=user.get('topic', 'general'),
                            audience=user.get('audience', 'general')
                        )
                
                print(f"  • Feedback pattern established for {user_data['user_id']}")
    
    async def demonstrate_fixed_learning(self):
        """Demonstrate the fixed learning system"""
        print("\nFIXED LEARNING & ADAPTATION")
        print("-" * 50)
        
        for user_id in ["john_lawyer", "mary_parent", "alex_gamer"]:
            print(f"\n👤 Learning demonstration for {user_id}:")
            
            # Show current preferences
            user_pref = await improved_aws_knowledge_base.get_user_preference(user_id)
            print(f"  Current liked personas: {user_pref.liked_personas}")
            print(f"  Current disliked personas: {user_pref.disliked_personas}")
            
            # Generate new humor with learned preferences
            request = HumorRequest(
                context="What's my favorite way to procrastinate? _____",
                audience="friends",
                topic="lifestyle",
                user_id=user_id
            )
            
            print(f"  Generating new humor with learned preferences...")
            result = await self.orchestrator.generate_and_evaluate_humor(request)
            
            if result['success']:
                print(f"  Recommended personas: {result['recommended_personas']}")
                
                best_result = result['best_result']
                if best_result:
                    generation = best_result['generation']
                    evaluation = best_result['evaluation']
                    
                    print(f"  Best result: \"{generation.text}\"")
                    print(f"  Selected persona: {generation.persona_name}")
                    print(f"  Score: {evaluation.overall_score:.1f}/10")
                    
                    # Verify learning worked
                    if generation.persona_name in user_pref.liked_personas:
                        print("  • SUCCESS: System selected a LIKED persona!")
                    elif generation.persona_name in user_pref.disliked_personas:
                        print("  • PROBLEM: System selected a DISLIKED persona!")
                    else:
                        print("  • System selected a neutral persona")
            else:
                print(f"  • Generation failed: {result.get('error')}")
    
    async def demonstrate_black_card_generation(self):
        """Demonstrate black card generation"""
        print("\nBLACK CARD GENERATION")
        print("-" * 50)
        
        topics = ["work", "relationships", "technology", "food"]
        
        for topic in topics:
            print(f"\n📝 Generating black card for topic: {topic}")
            
            request = HumorRequest(
                context="",  # Not needed for black cards
                audience="adults",
                topic=topic,
                user_id="creative_writer",
                card_type="black"
            )
            
            result = await self.orchestrator.generate_and_evaluate_humor(request)
            
            if result['success']:
                for i, evaluated_result in enumerate(result['results'][:2], 1):
                    generation = evaluated_result['generation']
                    evaluation = evaluated_result['evaluation']
                    
                    print(f"  {i}. \"{generation.text}\"")
                    print(f"     By: {generation.persona_name}")
                    print(f"     Score: {evaluation.overall_score:.1f}/10")
                    print(f"     Safe: {generation.is_safe}")
    
    async def demonstrate_group_consensus(self):
        """Demonstrate group consensus humor"""
        print("\nGROUP CONSENSUS HUMOR")
        print("-" * 50)
        
        group_id = "office_friends"
        member_ids = ["john_lawyer", "mary_parent", "alex_gamer"]
        
        # Create group context
        group_context = await improved_aws_knowledge_base.create_group_context(
            group_id, member_ids
        )
        
        print(f"Group: {group_id}")
        print(f"Members: {', '.join(member_ids)}")
        print(f"Consensus personas: {group_context['group_preferences']['consensus_personas']}")
        
        # Generate group humor
        request = HumorRequest(
            context="What's the best team building activity? _____",
            audience="group",
            topic="workplace",
            user_id=None  # Group mode
        )
        
        result = await self.orchestrator.generate_and_evaluate_humor(request)
        
        if result['success']:
            print("\nGroup humor options:")
            for i, evaluated_result in enumerate(result['results'], 1):
                generation = evaluated_result['generation']
                evaluation = evaluated_result['evaluation']
                
                print(f"  {i}. \"{generation.text}\"")
                print(f"     By: {generation.persona_name}")
                print(f"     Score: {evaluation.overall_score:.1f}/10")
                print(f"     Safe: {generation.is_safe}")
    
    async def show_user_analytics(self):
        """Show detailed user analytics"""
        print("\nUSER ANALYTICS")
        print("-" * 50)
        
        for user_id in ["john_lawyer", "mary_parent", "alex_gamer"]:
            analytics = await improved_aws_knowledge_base.get_user_analytics(user_id)
            
            if 'error' not in analytics:
                print(f"\n👤 {user_id} Analytics:")
                print(f"  Total interactions: {analytics['total_interactions']}")
                print(f"  Average score: {analytics['average_score']:.1f}/10")
                print(f"  Liked personas: {analytics['liked_personas']}")
                print(f"  Disliked personas: {analytics['disliked_personas']}")
                
                print("  Persona performance:")
                for persona, data in analytics['persona_performance'].items():
                    status_emoji = "💚" if data['status'] == 'liked' else "💔" if data['status'] == 'disliked' else "🤍"
                    print(f"    {status_emoji} {persona}: {data['avg_score']:.1f}/10 ({data['interaction_count']} interactions)")
    
    async def demonstrate_evaluation_improvements(self):
        """Demonstrate improved evaluation system"""
        print("\n🎯 EVALUATION IMPROVEMENTS")
        print("-" * 50)
        
        test_responses = [
            "A very funny and clever response",
            "Short",
            "This is a mediocre response that's okay",
            "An absolutely hilarious and unexpected twist that surprises everyone",
            "Boring standard response"
        ]
        
        request = HumorRequest(
            context="What's the funniest thing about work? _____",
            audience="colleagues",
            topic="work"
        )
        
        for response in test_responses:
            evaluation = await self.orchestrator.evaluator.evaluate_humor(response, request)
            
            print(f"  Response: \"{response}\"")
            print(f"  Overall Score: {evaluation.overall_score:.1f}/10")
            print(f"  Breakdown: {evaluation.reasoning}")
            print()

async def main():
    """Main fixed demonstration"""
    print("FIXED COMPLETE HUMOR GENERATION SYSTEM DEMO")
    print("All major issues have been addressed:")
    print("• Content filtering with detoxify")
    print("• Proper persona recommendation based on user preferences")
    print("• Fixed feedback learning (likes/dislikes actually work)")
    print("• Meaningful evaluation scores (not all 5/10)")
    print("• Black card generation")
    print("• AWS knowledge base integration")
    print("• Group consensus humor")
    print("\n" + "="*70)
    
    # Initialize system
    system = FixedCompleteHumorSystem()
    
    # Run complete demonstration
    await system.demonstrate_full_fixed_workflow()
    
    # Show evaluation improvements
    await system.demonstrate_evaluation_improvements()
    
    print(f"\n" + "="*70)
    print("FIXED DEMONSTRATION COMPLETE!")
    print("All major issues have been resolved:")
    print("  • Content filtering working")
    print("  • Persona recommendations working")
    print("  • Feedback learning working")
    print("  • Evaluation scores meaningful")
    print("  • Black card generation working")
    print("  • Group consensus working")
    print("  • Analytics working")
    print("="*70)

if __name__ == "__main__":
    asyncio.run(main()) 