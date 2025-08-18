#!/usr/bin/env python3
"""
Test Script for Improved CrewAI Agents
Compares baseline vs optimized multi-agent approach
"""

import asyncio
import json
import time
import os
import sys
from typing import List, Dict, Any
from dataclasses import dataclass

# Add the agent_system directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

from cah_standalone_analysis import SimpleCAHDataset, SimpleLLMClient, BaselineCAHGenerator, CAHHumorEvaluator

@dataclass
class TestResult:
    black_card: str
    baseline_response: str
    baseline_score: float
    multiagent_response: str
    multiagent_score: float
    improvement: float

class OptimizedCrewAIGenerator:
    """Simplified CrewAI-style generator that mimics our optimized approach"""
    
    def __init__(self, llm_client: SimpleLLMClient):
        self.llm_client = llm_client
    
    async def generate_white_card(self, black_card: str, model: str = "gpt-3.5-turbo") -> str:
        """Generate using optimized CrewAI-style approach"""
        
        # Use our improved, simplified prompt
        prompt = f"""Complete this Cards Against Humanity card with a single, hilarious response:

Black Card: "{black_card}"
Audience: adults
Topic: humor

Respond with just the white card text - be edgy, unexpected, and clever like CAH cards.
Make it appropriate for the "adults" audience.

White Card:"""
        
        # Generate 2 candidates and pick the best (simplified multi-agent)
        candidates = []
        for i in range(2):
            response = await self.llm_client.generate_response(prompt, model, temperature=0.9)
            candidates.append(self._clean_response(response))
        
        # Simple evaluation - pick the longer/more creative one
        return max(candidates, key=lambda x: len(x) + x.count(' '))
    
    def _clean_response(self, response_text: str) -> str:
        """Clean response similar to our optimized agent"""
        text = response_text.strip()
        
        # Remove "White Card:" prefix if present
        if text.lower().startswith('white card:'):
            text = text[11:].strip()
        
        # Remove quotes if present
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
        
        # Take only the first line/sentence
        text = text.split('\n')[0].strip()
        
        return text

async def run_improved_test():
    """Run comparison test with improved agents"""
    print("🧪 Testing Improved CrewAI Agents vs Baseline")
    print("=" * 50)
    
    # Initialize components
    dataset = SimpleCAHDataset()
    llm_client = SimpleLLMClient()
    evaluator = CAHHumorEvaluator(llm_client)
    baseline_generator = BaselineCAHGenerator(llm_client)
    optimized_generator = OptimizedCrewAIGenerator(llm_client)
    
    # Test with selected black cards for consistent comparison
    test_cards = [
        "What did I bring back from Mexico? _____.",
        "What would grandma find disturbing, yet oddly charming? _____.",
        "What's the next Happy Meal toy? _____.",
        "What helps Obama unwind? _____.",
        "What's that smell? _____."
    ]
    
    results = []
    baseline_scores = []
    multiagent_scores = []
    
    for i, black_card in enumerate(test_cards, 1):
        print(f"\n🎯 Test {i}: {black_card}")
        
        # Generate baseline response
        print("   🔹 Generating baseline response...")
        baseline_response = await baseline_generator.generate_white_card(black_card)
        print(f"   📝 Baseline: \"{baseline_response}\"")
        
        # Generate optimized multi-agent response
        print("   🔹 Generating optimized multi-agent response...")
        multiagent_response = await optimized_generator.generate_white_card(black_card)
        print(f"   📝 Multi-Agent: \"{multiagent_response}\"")
        
        # Create combinations for evaluation
        baseline_combination = type('obj', (object,), {
            'black_card': black_card,
            'white_cards': [baseline_response],
            'result': black_card.replace('_____', baseline_response)
        })()
        
        multiagent_combination = type('obj', (object,), {
            'black_card': black_card,
            'white_cards': [multiagent_response],
            'result': black_card.replace('_____', multiagent_response)
        })()
        
        # Evaluate both
        print("   🔹 Evaluating responses...")
        baseline_score = await evaluator.evaluate_humor(baseline_combination)
        multiagent_score = await evaluator.evaluate_humor(multiagent_combination)
        
        improvement = ((multiagent_score - baseline_score) / baseline_score) * 100 if baseline_score > 0 else 0
        
        print(f"   📊 Baseline Score: {baseline_score:.3f}")
        print(f"   📊 Multi-Agent Score: {multiagent_score:.3f}")
        print(f"   📈 Improvement: {improvement:+.1f}%")
        
        # Store results
        results.append(TestResult(
            black_card=black_card,
            baseline_response=baseline_response,
            baseline_score=baseline_score,
            multiagent_response=multiagent_response,
            multiagent_score=multiagent_score,
            improvement=improvement
        ))
        
        baseline_scores.append(baseline_score)
        multiagent_scores.append(multiagent_score)
        
        # Small delay to avoid rate limiting
        await asyncio.sleep(1)
    
    # Calculate overall statistics
    avg_baseline = sum(baseline_scores) / len(baseline_scores)
    avg_multiagent = sum(multiagent_scores) / len(multiagent_scores)
    overall_improvement = ((avg_multiagent - avg_baseline) / avg_baseline) * 100
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 IMPROVED TEST RESULTS SUMMARY")
    print("=" * 50)
    print(f"🎯 Tests Performed: {len(results)}")
    print(f"📈 Average Baseline Score: {avg_baseline:.3f}")
    print(f"📈 Average Multi-Agent Score: {avg_multiagent:.3f}")
    print(f"🚀 Overall Improvement: {overall_improvement:+.1f}%")
    
    improvements = [r.improvement for r in results]
    positive_improvements = [imp for imp in improvements if imp > 0]
    print(f"✅ Tests with Improvement: {len(positive_improvements)}/{len(results)} ({len(positive_improvements)/len(results)*100:.0f}%)")
    
    if positive_improvements:
        print(f"📈 Average Positive Improvement: {sum(positive_improvements)/len(positive_improvements):+.1f}%")
    
    # Save detailed results
    detailed_results = {
        'model_used': 'gpt-3.5-turbo',
        'test_type': 'improved_crewai_vs_baseline',
        'num_tests': len(results),
        'baseline_scores': baseline_scores,
        'multiagent_scores': multiagent_scores,
        'avg_baseline_score': avg_baseline,
        'avg_multiagent_score': avg_multiagent,
        'overall_improvement_percent': overall_improvement,
        'improvements': improvements,
        'positive_improvement_rate': len(positive_improvements) / len(results),
        'timestamp': time.time(),
        'detailed_results': [
            {
                'black_card': r.black_card,
                'baseline_response': r.baseline_response,
                'baseline_score': r.baseline_score,
                'multiagent_response': r.multiagent_response,
                'multiagent_score': r.multiagent_score,
                'improvement_percent': r.improvement
            }
            for r in results
        ]
    }
    
    # Save to file
    os.makedirs('final_results', exist_ok=True)
    with open('final_results/improved_cah_test_results.json', 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: final_results/improved_cah_test_results.json")
    
    # Performance analysis
    print("\n🔍 PERFORMANCE ANALYSIS")
    print("-" * 30)
    if overall_improvement > 10:
        print("✅ EXCELLENT: Multi-agent approach shows significant improvement!")
    elif overall_improvement > 5:
        print("✅ GOOD: Multi-agent approach shows meaningful improvement!")
    elif overall_improvement > 0:
        print("⚠️  MARGINAL: Multi-agent approach shows slight improvement.")
    else:
        print("❌ POOR: Multi-agent approach underperforms baseline.")
    
    print(f"\n🎯 Key Insights:")
    print(f"   • Computational cost reduced (2 candidates vs 5)")
    print(f"   • Evaluation simplified (single score vs complex JSON)")
    print(f"   • Prompt optimization focused on CAH-specific style")
    print(f"   • Success rate: {len(positive_improvements)}/{len(results)} tests improved")
    
    return detailed_results

if __name__ == "__main__":
    asyncio.run(run_improved_test()) 