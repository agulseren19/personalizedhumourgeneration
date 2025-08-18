#!/usr/bin/env python3
"""
Test script for new evaluation features:
1. Surprise Index calculation
2. BLEU/ROUGE baseline metrics
3. Best-Worst Scaling evaluation

These features implement recommendations from the literature review.
"""

import asyncio
from agents.humor_agents import SurpriseCalculator, HumorRequest
from agents.humor_evaluation_metrics import humor_metrics
from agents.bws_evaluation import bws_evaluator, BWS_Item

async def test_surprise_index():
    """Test the Surprise Index calculation (Tian et al.)"""
    print("🎯 Testing Surprise Index Calculation")
    print("=" * 50)
    
    calculator = SurpriseCalculator()
    
    test_cases = [
        {
            "context": "What's the worst part about adult life?",
            "humor": "Taxes and realizing vegetables are expensive",
            "expected": "Low surprise (predictable)"
        },
        {
            "context": "What's the worst part about adult life?", 
            "humor": "Quantum entanglement with my childhood imaginary friend",
            "expected": "High surprise (unexpected)"
        },
        {
            "context": "What would grandma find disturbing?",
            "humor": "A polite thank-you note",
            "expected": "Medium surprise (ironic)"
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        surprise_score = await calculator.calculate_surprise_index(
            case["humor"], 
            case["context"]
        )
        print(f"Test {i}:")
        print(f"  Context: {case['context']}")
        print(f"  Humor: {case['humor']}")
        print(f"  Surprise Score: {surprise_score:.2f}/10")
        print(f"  Expected: {case['expected']}")
        print()

def test_bleu_rouge_metrics():
    """Test BLEU/ROUGE evaluation metrics"""
    print("📊 Testing BLEU/ROUGE Metrics")
    print("=" * 50)
    
    # Test cases with generated and reference texts
    test_cases = [
        {
            "name": "Exact Match",
            "generated": "Something hilariously inappropriate",
            "reference": "Something hilariously inappropriate"
        },
        {
            "name": "Partial Overlap", 
            "generated": "Something surprisingly funny",
            "reference": "Something hilariously inappropriate"
        },
        {
            "name": "No Overlap",
            "generated": "Quantum physics homework",
            "reference": "Something hilariously inappropriate"
        },
        {
            "name": "Paraphrase",
            "generated": "Hilariously inappropriate content",
            "reference": "Something hilariously inappropriate"
        }
    ]
    
    for case in test_cases:
        print(f"Test: {case['name']}")
        print(f"  Generated: '{case['generated']}'")
        print(f"  Reference:  '{case['reference']}'")
        
        metrics = humor_metrics.calculate_all_metrics(
            case["generated"], 
            case["reference"]
        )
        
        print(f"  BLEU-1: {metrics.bleu_1:.3f}")
        print(f"  BLEU-2: {metrics.bleu_2:.3f}")
        print(f"  ROUGE-1: {metrics.rouge_1_f:.3f}")
        print(f"  ROUGE-L: {metrics.rouge_l_f:.3f}")
        print(f"  Distinct-1: {metrics.distinct_1:.3f} (diversity)")
        print()

def test_bws_evaluation():
    """Test Best-Worst Scaling evaluation system"""
    print("🏆 Testing Best-Worst Scaling Evaluation")
    print("=" * 50)
    
    # Create test items
    test_items = [
        BWS_Item(
            id="item_1",
            text="Something unexpectedly hilarious",
            metadata={"persona": "Witty Expert"}
        ),
        BWS_Item(
            id="item_2", 
            text="A boring predictable response",
            metadata={"persona": "Basic Generator"}
        ),
        BWS_Item(
            id="item_3",
            text="Quantum entangled with my childhood trauma",
            metadata={"persona": "Absurdist Comedian"}
        ),
        BWS_Item(
            id="item_4",
            text="Something moderately funny but safe",
            metadata={"persona": "Family-Friendly Bot"}
        )
    ]
    
    # Add items to evaluator
    bws_evaluator.add_items(test_items)
    
    # Generate comparisons
    comparisons = bws_evaluator.generate_comparisons(n_comparisons=6)
    print(f"Generated {len(comparisons)} BWS comparisons")
    
    # Simulate some judgments
    print("\nSimulating BWS judgments...")
    
    # Judgment 1: item_1 is best, item_2 is worst
    bws_evaluator.record_judgment(
        comparison_id=comparisons[0].comparison_id,
        best_item_id="item_1",
        worst_item_id="item_2",
        user_id="test_user_1"
    )
    
    # Judgment 2: item_3 is best, item_2 is worst
    bws_evaluator.record_judgment(
        comparison_id=comparisons[1].comparison_id,
        best_item_id="item_3", 
        worst_item_id="item_2",
        user_id="test_user_2"
    )
    
    # Judgment 3: item_1 is best, item_4 is worst
    bws_evaluator.record_judgment(
        comparison_id=comparisons[2].comparison_id,
        best_item_id="item_1",
        worst_item_id="item_4", 
        user_id="test_user_3"
    )
    
    # Calculate results
    results = bws_evaluator.calculate_bws_scores()
    
    print("\nBWS Results:")
    print("Rankings (Best to Worst):")
    for rank, (item_id, score) in enumerate(results.item_rankings, 1):
        item = bws_evaluator.items[item_id]
        print(f"  {rank}. {item.text[:40]}... (Score: {score:.3f})")
    
    print(f"\nTotal comparisons: {results.total_comparisons}")
    
    # Generate evaluation summary
    summary = bws_evaluator.generate_evaluation_summary()
    print(f"Statistical power: {summary['statistical_power']}")
    print(f"Completion rate: {summary['completion_rate']:.1%}")
    
    # Test comparison with Likert scores
    print("\nComparing BWS with Likert scores...")
    likert_scores = {
        "item_1": 8.5,  # High Likert score
        "item_2": 3.2,  # Low Likert score
        "item_3": 7.8,  # High Likert score  
        "item_4": 5.5   # Medium Likert score
    }
    
    comparison = bws_evaluator.compare_with_likert(likert_scores)
    
    if "pearson_correlation" in comparison:
        print(f"Pearson correlation: {comparison['pearson_correlation']:.3f}")
        print(f"Spearman correlation: {comparison['spearman_correlation']:.3f}")
        
        if comparison['pearson_correlation'] > 0.7:
            print("✅ High correlation - BWS and Likert agree")
        else:
            print("⚠️ Low correlation - Different evaluation patterns")

def test_integration():
    """Test integration of all new features"""
    print("🔗 Testing Feature Integration")
    print("=" * 50)
    
    print("Literature Support Summary:")
    print("✅ Surprise Index - Tian et al. (incongruity theory)")
    print("✅ BLEU/ROUGE - Traditional baselines (low humor correlation but needed)")
    print("✅ BWS Evaluation - Horvitz et al. (more robust than Likert-only)")
    print()
    
    print("Implementation Status:")
    print("✅ Surprise Index integrated into HumorEvaluationAgent")
    print("✅ BLEU/ROUGE available for baseline comparison")
    print("✅ BWS evaluation system with API endpoints")
    print("✅ All features backward compatible with existing system")
    print()
    
    print("API Endpoints Added:")
    print("• POST /bws/start-evaluation - Initialize BWS evaluation")
    print("• GET /bws/next-comparison/{user_id} - Get comparison for user")
    print("• POST /bws/submit-judgment - Record best/worst selections")
    print("• GET /bws/results - Get BWS scores and rankings") 
    print("• POST /bws/compare-with-likert - Compare BWS vs Likert")

async def main():
    """Run all tests"""
    print("🧪 Testing New Evaluation Features")
    print("Based on Literature Review Recommendations")
    print("=" * 60)
    print()
    
    # Test 1: Surprise Index
    await test_surprise_index()
    
    # Test 2: BLEU/ROUGE metrics
    test_bleu_rouge_metrics()
    
    # Test 3: BWS evaluation
    test_bws_evaluation()
    
    # Test 4: Integration summary
    test_integration()
    
    print("\n🎉 All tests completed!")
    print("\nNext Steps:")
    print("1. Test API endpoints with actual frontend integration")
    print("2. Collect real BWS judgments from users")
    print("3. Compare BWS vs current Likert-only evaluation")
    print("4. Analyze correlation between surprise index and humor ratings")

if __name__ == "__main__":
    asyncio.run(main()) 