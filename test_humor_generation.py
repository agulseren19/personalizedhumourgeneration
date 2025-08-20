#!/usr/bin/env python3
"""
Simple test script for humor generation without database dependencies
"""

import asyncio
import os
from agent_system.agents.improved_humor_agents import ImprovedHumorOrchestrator, HumorRequest

async def test_humor_generation():
    """Test humor generation without database"""
    print("🧪 Testing Humor Generation...")
    
    # Create test request
    request = HumorRequest(
        context="What's the next Happy Meal toy? _____",
        audience="friends",
        topic="general",
        user_id="test_user",
        card_type="white"
    )
    
    # Initialize orchestrator
    orchestrator = ImprovedHumorOrchestrator()
    
    try:
        print("🎭 Generating humor...")
        result = await orchestrator.generate_and_evaluate_humor(request)
        
        print("✅ Result:")
        print(f"   Success: {result.get('success', False)}")
        print(f"   Results: {len(result.get('results', []))}")
        print(f"   Num Results: {result.get('num_results', 0)}")
        
        if result.get('results'):
            for i, res in enumerate(result['results']):
                print(f"   Result {i+1}: {res['generation'].text}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    # Set API key
    os.environ["OPENAI_API_KEY"] = "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"
    
    # Run test
    result = asyncio.run(test_humor_generation())
    
    if result and result.get('success'):
        print("\n🎉 SUCCESS: Humor generation working!")
    else:
        print("\n❌ FAILED: Humor generation not working!")
