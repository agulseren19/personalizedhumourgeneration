#!/usr/bin/env python3
"""
Test script for black card generation
"""

import asyncio
import os
from agent_system.agents.improved_humor_agents import ImprovedHumorOrchestrator, HumorRequest

async def test_black_card_generation():
    """Test black card generation"""
    print("🧪 Testing Black Card Generation...")
    
    # Create test request for BLACK CARD
    request = HumorRequest(
        context="A fun party game setup",
        audience="friends",
        topic="general",
        user_id="test_user",
        card_type="black"  # BLACK CARD TEST
    )
    
    # Initialize orchestrator
    orchestrator = ImprovedHumorOrchestrator()
    
    try:
        print("🎭 Generating black card...")
        result = await orchestrator.generate_and_evaluate_humor(request)
        
        print("✅ Result:")
        print(f"   Success: {result.get('success', False)}")
        print(f"   Results: {len(result.get('results', []))}")
        print(f"   Num Results: {result.get('num_results', 0)}")
        
        if result.get('results'):
            for i, res in enumerate(result['results']):
                print(f"   Black Card {i+1}: {res['generation'].text}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    # Set API key
    os.environ["OPENAI_API_KEY"] = "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"
    
    # Run test
    result = asyncio.run(test_black_card_generation())
    
    if result and result.get('success'):
        print("\n🎉 SUCCESS: Black card generation working!")
    else:
        print("\n❌ FAILED: Black card generation not working!")
