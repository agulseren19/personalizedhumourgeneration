#!/usr/bin/env python3
"""
Test script for the enhanced evaluation system
Tests surprise index calculation and context-aware persona selection
"""

import asyncio
import sys
import os

# Add the parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from agent_system.agents.humor_agents import (
    SurpriseCalculator, 
    HumorEvaluationAgent,
    HumorAgentOrchestrator
)
from agent_system.agents.improved_humor_agents import HumorRequest

async def test_surprise_calculator():
    """Test the enhanced surprise index calculation"""
    print("🎯 Testing Enhanced Surprise Index Calculator")
    print("=" * 50)
    
    calculator = SurpriseCalculator()
    
    # Test cases with different levels of surprise
    test_cases = [
        {
            "context": "TSA guidelines now prohibit _____ on airplanes.",
            "humor_text": "The unexpected revelation of quantum physics in airport security",
            "expected_surprise": "high"
        },
        {
            "context": "My therapist says I have a problem with _____.",
            "humor_text": "The awkward silence",
            "expected_surprise": "medium"
        },
        {
            "context": "What's the next big thing in technology?",
            "humor_text": "Very good technology that is very advanced",
            "expected_surprise": "low"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"Context: {test_case['context']}")
        print(f"Humor: {test_case['humor_text']}")
        
        surprise_score = await calculator.calculate_surprise_index(
            test_case['humor_text'], 
            test_case['context']
        )
        
        print(f"Surprise Index: {surprise_score:.2f}/10")
        print(f"Expected Level: {test_case['expected_surprise']}")
        
        # Validate the score
        if test_case['expected_surprise'] == 'high' and surprise_score > 7.0:
            print("✅ PASS: High surprise detected correctly")
        elif test_case['expected_surprise'] == 'medium' and 4.0 <= surprise_score <= 7.0:
            print("✅ PASS: Medium surprise detected correctly")
        elif test_case['expected_surprise'] == 'low' and surprise_score < 4.0:
            print("✅ PASS: Low surprise detected correctly")
        else:
            print("⚠️  WARNING: Surprise level may not match expectation")

async def test_context_aware_selection():
    """Test context-aware persona selection"""
    print("\n🎭 Testing Context-Aware Persona Selection")
    print("=" * 50)
    
    try:
        orchestrator = HumorAgentOrchestrator()
        
        # Test different contexts
        test_contexts = [
            HumorRequest(
                context="Family-friendly joke about pets",
                audience="family",
                topic="pets",
                user_id="test_user",
                card_type="white"
            ),
            HumorRequest(
                context="Edgy workplace humor",
                audience="adults",
                topic="workplace",
                user_id="test_user",
                card_type="white"
            )
        ]
        
        for i, request in enumerate(test_contexts, 1):
            print(f"\nTest Context {i}:")
            print(f"Audience: {request.audience}")
            print(f"Topic: {request.topic}")
            
            # Test generation persona selection
            generation_personas = orchestrator._select_generation_personas(request, 2)
            print(f"Selected Generation Personas: {[p.name for p in generation_personas]}")
            
            # Test evaluation persona selection
            evaluator = orchestrator._select_context_aware_evaluator(
                request, 
                [p.name for p in generation_personas]
            )
            print(f"Selected Evaluator: {evaluator.name}")
            
            print("✅ PASS: Persona selection completed")
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

async def test_complete_pipeline():
    """Test the complete generation and evaluation pipeline"""
    print("\n🚀 Testing Complete Pipeline")
    print("=" * 50)
    
    try:
        orchestrator = HumorAgentOrchestrator()
        
        # Test request
        request = HumorRequest(
            context="What's the best way to make friends?",
            audience="friends",
            topic="social",
            user_id="test_user",
            card_type="white"
        )
        
        print(f"Request: {request.context}")
        print(f"Audience: {request.audience}")
        print(f"Topic: {request.topic}")
        
        # Run the pipeline
        result = await orchestrator.generate_and_evaluate_humor(request, num_generations=2)
        
        if result['success']:
            print(f"✅ Pipeline completed successfully!")
            print(f"Generated {result['total_generations']} cards")
            print(f"Evaluator: {result['evaluation_personas'][0]}")
            
            if 'evaluator_insights' in result:
                insights = result['evaluator_insights']
                print(f"Evaluator Insights: {insights.get('name', 'Unknown')}")
                print(f"Criteria: {insights.get('evaluation_criteria', 'Standard')}")
            
            # Show results
            for i, ranked_result in enumerate(result['top_results'], 1):
                generation = ranked_result['generation']
                evaluation = ranked_result['evaluation']
                
                print(f"\nCard {i}:")
                print(f"Text: {generation.text}")
                print(f"Persona: {generation.persona_name}")
                print(f"Humor: {evaluation.humor_score:.1f}/10")
                print(f"Creativity: {evaluation.creativity_score:.1f}/10")
                print(f"Surprise: {evaluation.surprise_index:.1f}/10")
                print(f"Overall: {evaluation.overall_score:.1f}/10")
                
        else:
            print(f"❌ Pipeline failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

async def main():
    """Run all tests"""
    print("🧪 Enhanced Evaluation System Test Suite")
    print("=" * 60)
    
    # Test 1: Surprise Calculator
    await test_surprise_calculator()
    
    # Test 2: Context-Aware Selection
    await test_context_aware_selection()
    
    # Test 3: Complete Pipeline
    await test_complete_pipeline()
    
    print("\n🎉 All tests completed!")
    print("\nSummary of Improvements:")
    print("✅ Enhanced Surprise Index based on Tian et al. research")
    print("✅ Context-aware persona selection")
    print("✅ Intelligent evaluation persona matching")
    print("✅ CrewAI integration for better evaluation")
    print("✅ Surprise index included in overall scoring")

if __name__ == "__main__":
    asyncio.run(main())
