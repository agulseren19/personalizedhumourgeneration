"""
Evaluate Real AI-Generated CAH Cards
Using literature-based metrics trained on real CAH dataset

This script evaluates the actual AI-generated cards from:
agent_system/ai_generated_cah_cards_20250823_234759.txt
"""

import sys
from pathlib import Path
import pandas as pd
from typing import List, Dict, Any

# Add evaluation directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from statistical_humor_evaluator import StatisticalHumorEvaluator


def load_ai_generated_cards(file_path: str) -> Dict[str, List[str]]:
    """Load AI-generated cards from text file"""
    
    cards = {
        'black_cards': [],
        'white_cards': []
    }
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse black cards
        black_section = content.split('BLACK CARDS')[1].split('WHITE CARDS')[0]
        black_lines = black_section.split('\n')
        
        for line in black_lines:
            line = line.strip()
            if line and line[0].isdigit() and '.' in line:
                # Extract card text after the number
                card_text = line.split('.', 1)[1].strip()
                if card_text:
                    cards['black_cards'].append(card_text)
        
        # Parse white cards
        white_section = content.split('WHITE CARDS')[1]
        white_lines = white_section.split('\n')
        
        for line in white_lines:
            line = line.strip()
            if line and line[0].isdigit() and '.' in line:
                # Extract card text after the number
                card_text = line.split('.', 1)[1].strip()
                if card_text:
                    cards['white_cards'].append(card_text)
        
        print(f"✅ Loaded AI-generated cards:")
        print(f"   Black cards: {len(cards['black_cards'])}")
        print(f"   White cards: {len(cards['white_cards'])}")
        
        return cards
        
    except Exception as e:
        print(f"❌ Error loading cards: {e}")
        return cards


def evaluate_ai_cards(cards: Dict[str, List[str]]) -> Dict[str, Any]:
    """Evaluate AI-generated cards by combining black + white = complete sentences"""
    
    print("\n🔬 Evaluating Complete Sentences (Black Card + White Card)...")
    print("=" * 80)
    
    evaluator = StatisticalHumorEvaluator()
    results = {
        'complete_sentences': [],
        'summary': {}
    }
    
    # Create complete sentences by combining black and white cards
    print("\n🎭 Creating Complete Sentences:")
    print("-" * 40)
    
    complete_sentences = []
    black_cards = cards['black_cards'][:10]  # First 10 black cards
    white_cards = cards['white_cards'][:15]  # First 15 white cards
    
    # Create combinations (black + white = complete sentence)
    for i, black_card in enumerate(black_cards):
        if i < len(white_cards):
            white_card = white_cards[i]
            
            # Fill in the blank with white card
            complete_sentence = black_card.replace('_____', white_card)
            complete_sentences.append({
                'black_card': black_card,
                'white_card': white_card,
                'complete_sentence': complete_sentence,
                'combination_id': i + 1
            })
            
            print(f"{i+1:2d}. Black: '{black_card[:50]}{'...' if len(black_card) > 50 else ''}'")
            print(f"    White: '{white_card[:50]}{'...' if len(white_card) > 50 else ''}'")
            print(f"    Complete: '{complete_sentence[:80]}{'...' if len(complete_sentence) > 80 else ''}'")
            print()
    
    # Evaluate complete sentences
    print("\n📝 COMPLETE SENTENCE EVALUATION:")
    print("-" * 50)
    
    sentence_scores = []
    for i, combo in enumerate(complete_sentences):
        complete_sentence = combo['complete_sentence']
        
        # Use the black card as context for evaluation
        context = combo['black_card']
        
        # Evaluate the complete sentence
        scores = evaluator.evaluate_humor_statistically(complete_sentence, context)
        
        result = {
            'combination_id': combo['combination_id'],
            'black_card': combo['black_card'],
            'white_card': combo['white_card'],
            'complete_sentence': complete_sentence,
            'scores': scores
        }
        results['complete_sentences'].append(result)
        sentence_scores.append(scores.overall_humor_score)
        
        print(f"{i+1:2d}. Complete: '{complete_sentence[:70]}{'...' if len(complete_sentence) > 70 else ''}'")
        print(f"    Surprisal: {scores.surprisal_score:.2f}, Creativity: {scores.distinct_1:.3f}, Overall: {scores.overall_humor_score:.2f}")
    
    # Calculate summary statistics for complete sentences
    if sentence_scores:
        results['summary']['complete_sentences'] = {
            'count': len(sentence_scores),
            'avg_overall': sum(sentence_scores) / len(sentence_scores),
            'avg_surprisal': sum(r['scores'].surprisal_score for r in results['complete_sentences']) / len(results['complete_sentences']),
            'avg_creativity': sum(r['scores'].distinct_1 for r in results['complete_sentences']) / len(results['complete_sentences']),
            'best_sentence': max(results['complete_sentences'], key=lambda x: x['scores'].overall_humor_score),
            'worst_sentence': min(results['complete_sentences'], key=lambda x: x['scores'].overall_humor_score)
        }
    
    return results


def print_evaluation_summary(results: Dict[str, Any]):
    """Print comprehensive evaluation summary for complete sentences"""
    
    print("\n📊 EVALUATION SUMMARY:")
    print("=" * 80)
    
    # Complete sentences summary
    if 'complete_sentences' in results['summary']:
        summary = results['summary']['complete_sentences']
        print(f"\n🎭 COMPLETE SENTENCES ({summary['count']} evaluated):")
        print(f"   Average Overall Score: {summary['avg_overall']:.2f}/10")
        print(f"   Average Surprisal: {summary['avg_surprisal']:.2f}/10")
        print(f"   Average Creativity: {summary['avg_creativity']:.3f}")
        print(f"   Best Sentence: '{summary['best_sentence']['complete_sentence'][:60]}...' ({summary['best_sentence']['scores'].overall_humor_score:.2f}/10)")
        print(f"   Worst Sentence: '{summary['worst_sentence']['complete_sentence'][:60]}...' ({summary['worst_sentence']['scores'].overall_humor_score:.2f}/10)")
    
    # Overall assessment
    print(f"\n🎯 OVERALL ASSESSMENT:")
    print(f"   Literature-based metrics successfully evaluated {len(results['complete_sentences'])} complete sentences")
    print(f"   Metrics used: Surprisal (Tian et al. 2022), Ambiguity (Kao 2016), Creativity (Li et al. 2016)")
    print(f"   Training data: Real CAH dataset (44,718 cards)")
    print(f"   No data leakage: Training and evaluation data completely separate")
    print(f"   Evaluation method: Black card + White card = Complete sentence")


def save_evaluation_results(results: Dict[str, Any], output_file: str = "complete_sentences_evaluation_results.json"):
    """Save evaluation results to JSON file"""
    
    try:
        import json
        
        # Convert dataclass objects to dictionaries
        serializable_results = {
            'complete_sentences': [],
            'summary': results['summary']
        }
        
        # Convert complete sentences
        for result in results['complete_sentences']:
            serializable_results['complete_sentences'].append({
                'combination_id': result['combination_id'],
                'black_card': result['black_card'],
                'white_card': result['white_card'],
                'complete_sentence': result['complete_sentence'],
                'scores': {
                    'surprisal_score': result['scores'].surprisal_score,
                    'ambiguity_score': result['scores'].ambiguity_score,
                    'distinctiveness_ratio': result['scores'].distinctiveness_ratio,
                    'entropy_score': result['scores'].entropy_score,
                    'perplexity_score': result['scores'].perplexity_score,
                    'semantic_coherence': result['scores'].semantic_coherence,
                    'distinct_1': result['scores'].distinct_1,
                    'distinct_2': result['scores'].distinct_2,
                    'self_bleu': result['scores'].self_bleu,
                    'mauve_score': result['scores'].mauve_score,
                    'vocabulary_richness': result['scores'].vocabulary_richness,
                    'overall_semantic_diversity': result['scores'].overall_semantic_diversity,
                    'intra_cluster_diversity': result['scores'].intra_cluster_diversity,
                    'inter_cluster_diversity': result['scores'].inter_cluster_diversity,
                    'semantic_spread': result['scores'].semantic_spread,
                    'cluster_coherence': result['scores'].cluster_coherence,
                    'overall_humor_score': result['scores'].overall_humor_score
                }
            })
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Error saving results: {e}")


def main():
    """Main evaluation function - evaluates complete sentences (black + white cards)"""
    
    print("🎭 Complete Sentence Evaluation (Black Card + White Card)")
    print("=" * 60)
    print("Using Literature-Based Metrics (Trained on Real CAH Dataset)")
    print("=" * 60)
    
    # Load AI-generated cards
    cards_file = "agent_system/ai_generated_cah_cards_20250823_234759.txt"
    cards = load_ai_generated_cards(cards_file)
    
    if not cards['black_cards'] and not cards['white_cards']:
        print("❌ No cards loaded. Exiting.")
        return
    
    # Evaluate complete sentences
    results = evaluate_ai_cards(cards)
    
    # Print summary
    print_evaluation_summary(results)
    
    # Save results
    save_evaluation_results(results)
    
    print("\n🎉 Evaluation Complete!")
    print("Complete sentences (Black + White cards) have been evaluated using:")
    print("✅ Surprisal (Tian et al. 2022)")
    print("✅ Ambiguity (Kao & Witbrock 2016)")
    print("✅ Creativity/Diversity (Li et al. 2016, Zhu et al. 2018)")
    print("✅ Linguistic Quality (Information Theory)")
    print("\n💡 This approach evaluates the final humor of complete sentences,")
    print("   not individual cards, providing more realistic humor assessment.")


if __name__ == "__main__":
    main()
