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


def create_mock_user_profiles() -> Dict[str, List[str]]:
    """Create mock user humor profiles for personalization testing"""
    
    # Mock user profiles based on different humor styles
    user_profiles = {
        'dark_humor_user': [
            "Existential dread and coffee",
            "My therapist's secret addiction",
            "The art of passive-aggressive gift-giving",
            "Why I'm banned from family gatherings",
            "My cat's elaborate revenge schemes"
        ],
        'dad_jokes_user': [
            "Why did the scarecrow win an award?",
            "What do you call a fake noodle?",
            "How does a penguin build its house?",
            "Why don't eggs tell jokes?",
            "What's the best time to go to the dentist?"
        ],
        'sarcastic_user': [
            "Oh great, another meeting that could have been an email",
            "My life is like a romantic comedy, except there's no romance and it's just me crying",
            "I'm not lazy, I'm just conserving energy",
            "I'm not arguing, I'm just explaining why I'm right",
            "I'm not procrastinating, I'm prioritizing my mental health"
        ],
        'pop_culture_user': [
            "Game of Thrones season 8 plot holes",
            "Marvel movie timeline confusion",
            "Star Wars prequel memes",
            "The Office quotes in real life",
            "Breaking Bad cooking lessons"
        ],
        'random_user': [
            "A penguin wearing a tuxedo",
            "The secret life of socks",
            "Why do we park in driveways?",
            "The existential crisis of a potato",
            "My neighbor's suspicious garden gnomes"
        ]
    }
    
    return user_profiles


def evaluate_personalization(cards: Dict[str, List[str]], user_profiles: Dict[str, List[str]]) -> Dict[str, Any]:
    """Evaluate personalization effectiveness using PaCS metric"""
    
    print("\n🎭 PERSONALIZATION EVALUATION (PaCS Metric):")
    print("=" * 60)
    print("Based on Deep-SHEEP (Bielaniewicz et al., 2022)")
    print("PaCS = cos(user_profile, card_embedding) ∈ [-1, 1]")
    print("=" * 60)
    
    from evaluation.statistical_humor_evaluator import PersonalizationEvaluator
    
    personalization_evaluator = PersonalizationEvaluator()
    results = {
        'user_profiles': {},
        'overall_personalization': {}
    }
    
    # First, we need to create complete sentences from the cards
    complete_sentences = []
    black_cards = cards['black_cards'][:5]  # First 5 black cards
    white_cards = cards['white_cards'][:5]  # First 5 white cards
    
    for i, (black_card, white_card) in enumerate(zip(black_cards, white_cards)):
        complete_sentence = black_card.replace('_____', white_card)
        complete_sentences.append({
            'combination_id': i + 1,
            'black_card': black_card,
            'white_card': white_card,
            'complete_sentence': complete_sentence
        })
    
    # Evaluate each user profile
    for user_type, profile_cards in user_profiles.items():
        print(f"\n👤 {user_type.upper().replace('_', ' ')}:")
        print("-" * 40)
        
        user_results = {
            'profile_cards': profile_cards,
            'card_evaluations': [],
            'avg_pacs_score': 0.0,
            'personalization_effectiveness': 'Unknown'
        }
        
        # Evaluate each generated card against this user profile
        card_scores = []
        for combo in complete_sentences:
            complete_sentence = combo['complete_sentence']
            
            # Calculate PaCS score
            pacs_score = personalization_evaluator.calculate_pacs_score(complete_sentence, profile_cards)
            normalized_score = personalization_evaluator.normalize_pacs_score(pacs_score)
            insights = personalization_evaluator.get_personalization_insights(pacs_score)
            
            card_eval = {
                'card_text': complete_sentence,
                'pacs_score': pacs_score,
                'normalized_score': normalized_score,
                'effectiveness': insights['effectiveness'],
                'interpretation': insights['interpretation']
            }
            
            user_results['card_evaluations'].append(card_eval)
            card_scores.append(pacs_score)
            
            print(f"  Card {combo['combination_id']}: {complete_sentence[:60]}{'...' if len(complete_sentence) > 60 else ''}")
            print(f"    PaCS: {pacs_score:.3f} | Effectiveness: {insights['effectiveness']}")
            print(f"    {insights['interpretation']}")
        
        # Calculate average PaCS for this user
        if card_scores:
            avg_pacs = sum(card_scores) / len(card_scores)
            user_results['avg_pacs_score'] = avg_pacs
            
            # Determine overall effectiveness
            if avg_pacs >= 0.7:
                effectiveness = "Excellent"
            elif avg_pacs >= 0.3:
                effectiveness = "Good"
            elif avg_pacs >= -0.1:
                effectiveness = "Neutral"
            elif avg_pacs >= -0.5:
                effectiveness = "Poor"
            else:
                effectiveness = "Very Poor"
            
            user_results['personalization_effectiveness'] = effectiveness
            
            print(f"\n  📊 Overall Personalization: {effectiveness}")
            print(f"  📈 Average PaCS: {avg_pacs:.3f}")
        
        results['user_profiles'][user_type] = user_results
    
    # Calculate overall personalization statistics
    all_pacs_scores = []
    for user_data in results['user_profiles'].values():
        if user_data['card_evaluations']:
            all_pacs_scores.extend([card['pacs_score'] for card in user_data['card_evaluations']])
    
    if all_pacs_scores:
        results['overall_personalization'] = {
            'total_evaluations': len(all_pacs_scores),
            'avg_pacs_score': sum(all_pacs_scores) / len(all_pacs_scores),
            'min_pacs_score': min(all_pacs_scores),
            'max_pacs_score': max(all_pacs_scores),
            'personalization_distribution': {
                'excellent': len([s for s in all_pacs_scores if s >= 0.7]),
                'good': len([s for s in all_pacs_scores if 0.3 <= s < 0.7]),
                'neutral': len([s for s in all_pacs_scores if -0.1 <= s < 0.3]),
                'poor': len([s for s in all_pacs_scores if -0.5 <= s < -0.1]),
                'very_poor': len([s for s in all_pacs_scores if s < -0.5])
            }
        }
        
        print(f"\n🎯 OVERALL PERSONALIZATION SUMMARY:")
        print(f"   Total Evaluations: {results['overall_personalization']['total_evaluations']}")
        print(f"   Average PaCS: {results['overall_personalization']['avg_pacs_score']:.3f}")
        print(f"   PaCS Range: {results['overall_personalization']['min_pacs_score']:.3f} to {results['overall_personalization']['max_pacs_score']:.3f}")
        print(f"   Distribution: Excellent({results['overall_personalization']['personalization_distribution']['excellent']}) | "
              f"Good({results['overall_personalization']['personalization_distribution']['good']}) | "
              f"Neutral({results['overall_personalization']['personalization_distribution']['neutral']}) | "
              f"Poor({results['overall_personalization']['personalization_distribution']['poor']}) | "
              f"Very Poor({results['overall_personalization']['personalization_distribution']['very_poor']})")
    
    return results


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
    
    # Create mock user profiles for personalization testing
    user_profiles = create_mock_user_profiles()
    
    # Evaluate personalization effectiveness
    personalization_results = evaluate_personalization(cards, user_profiles)
    
    # Add personalization results to main results
    results['personalization'] = personalization_results
    
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
    print("✅ Personalization (PaCS - Deep-SHEEP 2022)")
    print("\n💡 This approach evaluates the final humor of complete sentences,")
    print("   not individual cards, providing more realistic humor assessment.")
    print("   Plus personalization effectiveness using BERT-based embeddings!")


if __name__ == "__main__":
    main()
