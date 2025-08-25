#!/usr/bin/env python3
"""
Calculate F1 Score for Complete Sentences Evaluation Results
Using y_true = [1, 0, 1, 1, 1, 1, 1, 1, 1, 1]
"""

import json
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from pathlib import Path

def load_evaluation_results(file_path: str = "complete_sentences_evaluation_results.json"):
    """Load the evaluation results from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        print(f"✅ Loaded evaluation results from {file_path}")
        return results
    except Exception as e:
        print(f"❌ Error loading results: {e}")
        return None

def calculate_f1_metrics(y_true, y_pred, threshold=6.0):
    """Calculate F1 score and related metrics"""
    
    # Convert continuous scores to binary predictions using threshold
    y_pred_binary = [1 if score >= threshold else 0 for score in y_pred]
    
    # Calculate metrics
    f1 = f1_score(y_true, y_pred_binary)
    precision = precision_score(y_true, y_pred_binary)
    recall = recall_score(y_true, y_pred_binary)
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    
    # Additional metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return {
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'specificity': specificity,
        'sensitivity': sensitivity,
        'threshold_used': threshold,
        'confusion_matrix': {
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp)
        },
        'predictions': {
            'y_true': y_true,
            'y_pred_continuous': y_pred,
            'y_pred_binary': y_pred_binary
        }
    }

def analyze_threshold_sensitivity(y_true, y_pred, thresholds=None):
    """Analyze how F1 score changes with different thresholds"""
    if thresholds is None:
        thresholds = np.arange(4.0, 8.1, 0.2)
    
    threshold_analysis = []
    
    for threshold in thresholds:
        y_pred_binary = [1 if score >= threshold else 0 for score in y_pred]
        f1 = f1_score(y_true, y_pred_binary)
        precision = precision_score(y_true, y_pred_binary)
        recall = recall_score(y_true, y_pred_binary)
        
        threshold_analysis.append({
            'threshold': float(threshold),
            'f1_score': float(f1),
            'precision': float(precision),
            'recall': float(recall)
        })
    
    return threshold_analysis

def save_f1_results(results, output_file: str = "f1_score_analysis.json"):
    """Save F1 score analysis results to JSON file"""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"💾 F1 score analysis saved to: {output_file}")
    except Exception as e:
        print(f"❌ Error saving F1 results: {e}")

def print_f1_summary(metrics):
    """Print a summary of the F1 score analysis"""
    print("\n🎯 F1 SCORE ANALYSIS SUMMARY")
    print("=" * 50)
    print(f"Threshold used: {metrics['threshold_used']:.1f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print(f"Sensitivity: {metrics['sensitivity']:.4f}")
    
    print(f"\n📊 Confusion Matrix:")
    cm = metrics['confusion_matrix']
    print(f"   True Negatives: {cm['true_negatives']}")
    print(f"   False Positives: {cm['false_positives']}")
    print(f"   False Negatives: {cm['false_negatives']}")
    print(f"   True Positives: {cm['true_positives']}")
    
    print(f"\n🔍 Predictions:")
    print(f"   y_true: {metrics['predictions']['y_true']}")
    print(f"   y_pred_continuous: {[f'{x:.2f}' for x in metrics['predictions']['y_pred_continuous']]}")
    print(f"   y_pred_binary: {metrics['predictions']['y_pred_binary']}")

def main():
    """Main function to calculate F1 score for evaluation results"""
    
    print("🎭 F1 Score Analysis for Complete Sentences Evaluation")
    print("=" * 60)
    
    # Load evaluation results
    results = load_evaluation_results()
    if not results:
        return
    
    # Extract overall humor scores (first 10 sentences)
    humor_scores = []
    for i, sentence in enumerate(results['complete_sentences'][:10]):
        score = sentence['scores']['overall_humor_score']
        humor_scores.append(score)
        print(f"  Sentence {i+1}: {score:.2f}/10")
    
    # Ground truth labels
    y_true = [1, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    
    print(f"\n📊 Ground Truth Labels: {y_true}")
    print(f"📈 Predicted Scores: {[f'{x:.2f}' for x in humor_scores]}")
    
    # Calculate F1 score with default threshold (6.0)
    print(f"\n🔬 Calculating F1 Score with threshold = 6.0...")
    f1_metrics = calculate_f1_metrics(y_true, humor_scores, threshold=6.0)
    
    # Print summary
    print_f1_summary(f1_metrics)
    
    # Analyze threshold sensitivity
    print(f"\n🔍 Analyzing threshold sensitivity...")
    threshold_analysis = analyze_threshold_sensitivity(y_true, humor_scores)
    
    # Find optimal threshold
    best_threshold = max(threshold_analysis, key=lambda x: x['f1_score'])
    print(f"\n⭐ Optimal Threshold Analysis:")
    print(f"   Best F1 Score: {best_threshold['f1_score']:.4f}")
    print(f"   Optimal Threshold: {best_threshold['threshold']:.1f}")
    print(f"   Precision at optimal: {best_threshold['precision']:.4f}")
    print(f"   Recall at optimal: {best_threshold['recall']:.4f}")
    
    # Prepare results for saving
    final_results = {
        'f1_analysis': f1_metrics,
        'threshold_sensitivity': threshold_analysis,
        'optimal_threshold': best_threshold,
        'summary': {
            'total_sentences': len(humor_scores),
            'positive_labels': sum(y_true),
            'negative_labels': len(y_true) - sum(y_true),
            'avg_humor_score': np.mean(humor_scores),
            'std_humor_score': np.std(humor_scores)
        }
    }
    
    # Save results
    save_f1_results(final_results)
    
    print(f"\n🎉 F1 Score Analysis Complete!")
    print(f"   Results saved to: f1_score_analysis.json")

if __name__ == "__main__":
    main()
