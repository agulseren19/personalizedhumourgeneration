#!/usr/bin/env python3
"""
Comprehensive Evaluation Comparison System
Compares different humor generation approaches and evaluation metrics
For research report analysis and performance comparison
"""

import asyncio
import time
import json
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import pandas as pd

# Import all our systems with error handling
import sys
import os

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    # Try to import with proper path resolution
    project_root = os.path.dirname(current_dir)
    sys.path.insert(0, project_root)
    
    from agent_system.agents.humor_agents import HumorAgentOrchestrator, HumorRequest, SurpriseCalculator
    from agent_system.agents.retrieval_augmented_humor import retrieval_humor_generator
    from agent_system.agents.controlled_generation import controlled_humor_generator, ControlVector, ControlAttribute, GenerationConstraints
    from agent_system.agents.enhanced_content_filter import enhanced_content_filter, FilterThresholds
    from agent_system.agents.humor_evaluation_metrics import humor_metrics, OverlapMetrics
    from agent_system.agents.bws_evaluation import bws_evaluator, BWS_Item
except ImportError:
    try:
        # Fallback: relative imports
        from agents.humor_agents import HumorAgentOrchestrator, HumorRequest, SurpriseCalculator
        from agents.retrieval_augmented_humor import retrieval_humor_generator
        from agents.controlled_generation import controlled_humor_generator, ControlVector, ControlAttribute, GenerationConstraints
        from agents.enhanced_content_filter import enhanced_content_filter, FilterThresholds
        from agents.humor_evaluation_metrics import humor_metrics, OverlapMetrics
        from agents.bws_evaluation import bws_evaluator, BWS_Item
    except ImportError as e:
        print(f"Import error in evaluation_comparison_system: {e}")
        # Create minimal fallbacks for testing
        class SurpriseCalculator:
            async def calculate_surprise_index(self, text, context):
                return 5.0

@dataclass
class GenerationResult:
    """Result from a single generation approach"""
    method: str
    generated_text: str
    generation_time: float
    metadata: Dict[str, Any]
    
    # Evaluation scores
    humor_score: float = 0.0
    creativity_score: float = 0.0
    appropriateness_score: float = 0.0
    context_relevance_score: float = 0.0
    surprise_index: float = 0.0
    
    # Content safety
    toxicity_score: float = 0.0
    is_safe: bool = True
    
    # Overlap metrics (if reference available)
    bleu_1: float = 0.0
    rouge_1: float = 0.0
    rouge_l: float = 0.0
    
    # BWS score (if BWS evaluation done)
    bws_score: float = 0.0

@dataclass
class ComparisonReport:
    """Comprehensive comparison report"""
    test_prompts: List[str]
    generation_results: List[GenerationResult]
    performance_summary: Dict[str, Any]
    method_rankings: Dict[str, Dict[str, float]]
    evaluation_correlations: Dict[str, float]
    recommendations: List[str]

class EvaluationComparisonSystem:
    """
    Comprehensive system for comparing different humor generation approaches
    Provides detailed analysis for research reports
    """
    
    def __init__(self):
        self.orchestrator = HumorAgentOrchestrator()
        self.surprise_calculator = SurpriseCalculator()
        
        # Test prompts for evaluation
        self.test_prompts = [
            "What's the worst part about adult life?",
            "What would grandma find disturbing?", 
            "What do you hate about family gatherings?",
            "What's inappropriate at work but normal at home?",
            "What's the most awkward thing about dating?",
            "What ruins a good day instantly?",
            "What's overrated in modern life?",
            "What's the weirdest thing about being human?"
        ]
        
        # Reference responses for BLEU/ROUGE calculation
        self.reference_responses = {
            "What's the worst part about adult life?": [
                "Realizing vegetables are expensive",
                "Paying for things you took for granted as a kid",
                "Having to make all your own decisions"
            ],
            "What would grandma find disturbing?": [
                "Kids these days not calling",
                "The price of everything nowadays", 
                "Young people on their phones"
            ]
        }
    
    async def run_comprehensive_comparison(
        self, 
        test_subset: Optional[List[str]] = None,
        include_bws: bool = False
    ) -> ComparisonReport:
        """
        Run comprehensive comparison of all generation methods
        
        Methods tested:
        1. Original CrewAI Multi-Agent System
        2. Retrieval-Augmented Generation
        3. Controlled Generation (PPLM-style)
        4. Hybrid approaches
        """
        
        prompts = test_subset or self.test_prompts[:4]  # Use subset for faster testing
        all_results = []
        
        print("🔬 Starting Comprehensive Humor Generation Comparison")
        print("=" * 60)
        
        for i, prompt in enumerate(prompts, 1):
            print(f"\n🎯 Test {i}/{len(prompts)}: {prompt}")
            
            # Test Method 1: Original CrewAI Multi-Agent
            print("  Testing: CrewAI Multi-Agent System...")
            crewai_result = await self._test_crewai_method(prompt)
            all_results.append(crewai_result)
            
            # Test Method 2: Retrieval-Augmented 
            print("  Testing: Retrieval-Augmented Generation...")
            retrieval_result = await self._test_retrieval_method(prompt)
            all_results.append(retrieval_result)
            
            # Test Method 3: Controlled Generation
            print("  Testing: Controlled Generation (PPLM-style)...")
            controlled_result = await self._test_controlled_method(prompt)
            all_results.append(controlled_result)
            
            # Test Method 4: Hybrid (Retrieval + Control)
            print("  Testing: Hybrid (Retrieval + Control)...")
            hybrid_result = await self._test_hybrid_method(prompt)
            all_results.append(hybrid_result)
            
            print(f"  ✅ Completed tests for prompt {i}")
        
        # Run BWS evaluation if requested
        if include_bws:
            print("\n🏆 Running Best-Worst Scaling Evaluation...")
            await self._run_bws_evaluation(all_results)
        
        # Generate comprehensive report
        print("\n📊 Generating Comparison Report...")
        report = self._generate_comparison_report(prompts, all_results)
        
        return report
    
    async def _test_crewai_method(self, prompt: str) -> GenerationResult:
        """Test original CrewAI multi-agent system"""
        start_time = time.time()
        
        request = HumorRequest(
            context=prompt,
            audience="general",
            topic="general",
            user_id=1
        )
        
        try:
            # Generate with CrewAI
            result = await self.orchestrator.generate_humor(request)
            generation_time = time.time() - start_time
            
            # Extract metrics from result
            generated_text = result.get('response', {}).get('text', 'Failed to generate')
            evaluation = result.get('evaluation', {})
            
            # Create result object
            crewai_result = GenerationResult(
                method="CrewAI Multi-Agent",
                generated_text=generated_text,
                generation_time=generation_time,
                metadata={
                    "persona_used": result.get('response', {}).get('persona_name', 'Unknown'),
                    "model_used": result.get('response', {}).get('model_used', 'Unknown'),
                    "tokens_used": result.get('response', {}).get('tokens_used', 0)
                },
                humor_score=evaluation.get('humor_score', 0.0),
                creativity_score=evaluation.get('creativity_score', 0.0),
                appropriateness_score=evaluation.get('appropriateness_score', 0.0),
                context_relevance_score=evaluation.get('context_relevance_score', 0.0),
                surprise_index=evaluation.get('surprise_index', 0.0)
            )
            
            # Add safety analysis
            await self._add_safety_analysis(crewai_result)
            
            # Add overlap metrics if reference available
            await self._add_overlap_metrics(crewai_result, prompt)
            
            return crewai_result
            
        except Exception as e:
            print(f"    ❌ CrewAI method failed: {e}")
            return GenerationResult(
                method="CrewAI Multi-Agent",
                generated_text=f"Error: {str(e)}",
                generation_time=time.time() - start_time,
                metadata={"error": str(e)}
            )
    
    async def _test_retrieval_method(self, prompt: str) -> GenerationResult:
        """Test retrieval-augmented generation"""
        start_time = time.time()
        
        try:
            result = await retrieval_humor_generator.generate_with_retrieval(
                prompt=prompt,
                audience="general",
                style="witty"
            )
            
            generation_time = time.time() - start_time
            generated_text = result['generated_text']
            
            # Evaluate with our metrics
            surprise_score = await self.surprise_calculator.calculate_surprise_index(generated_text, prompt)
            
            retrieval_result = GenerationResult(
                method="Retrieval-Augmented",
                generated_text=generated_text,
                generation_time=generation_time,
                metadata={
                    "entities_found": result.get('entities_found', []),
                    "confidence": result.get('confidence', 0.0),
                    "template_used": result.get('template_used', {}).category if result.get('template_used') else 'None',
                    "retrieval_method": "knowledge_base"
                },
                surprise_index=surprise_score
            )
            
            # Add safety and overlap analysis
            await self._add_safety_analysis(retrieval_result)
            await self._add_overlap_metrics(retrieval_result, prompt)
            
            return retrieval_result
            
        except Exception as e:
            print(f"    ❌ Retrieval method failed: {e}")
            return GenerationResult(
                method="Retrieval-Augmented",
                generated_text=f"Error: {str(e)}",
                generation_time=time.time() - start_time,
                metadata={"error": str(e)}
            )
    
    async def _test_controlled_method(self, prompt: str) -> GenerationResult:
        """Test controlled generation (PPLM-style)"""
        start_time = time.time()
        
        try:
            # Define control vectors
            control_vectors = [
                ControlVector(attribute=ControlAttribute.HUMOR, strength=0.8, weight=1.0),
                ControlVector(attribute=ControlAttribute.APPROPRIATENESS, strength=0.6, weight=0.7),
                ControlVector(attribute=ControlAttribute.CREATIVITY, strength=0.7, weight=0.8)
            ]
            
            constraints = GenerationConstraints(
                max_length=100,
                min_humor_score=0.6,
                max_toxicity=0.3
            )
            
            result = await controlled_humor_generator.controlled_generate(
                prompt=prompt,
                control_vectors=control_vectors,
                constraints=constraints
            )
            
            generation_time = time.time() - start_time
            generated_text = result['generated_text']
            
            controlled_result = GenerationResult(
                method="Controlled Generation",
                generated_text=generated_text,
                generation_time=generation_time,
                metadata={
                    "control_vectors": result.get('control_vectors_applied', []),
                    "meets_constraints": result.get('meets_constraints', False),
                    "candidates_considered": result.get('candidates_considered', 0),
                    "attribute_scores": result.get('attribute_scores', {})
                },
                humor_score=result.get('attribute_scores', {}).get('humor', 0.0),
                creativity_score=result.get('attribute_scores', {}).get('creativity', 0.0),
                appropriateness_score=result.get('attribute_scores', {}).get('appropriateness', 0.0)
            )
            
            # Add safety and overlap analysis
            await self._add_safety_analysis(controlled_result)
            await self._add_overlap_metrics(controlled_result, prompt)
            
            return controlled_result
            
        except Exception as e:
            print(f"    ❌ Controlled method failed: {e}")
            return GenerationResult(
                method="Controlled Generation",
                generated_text=f"Error: {str(e)}",
                generation_time=time.time() - start_time,
                metadata={"error": str(e)}
            )
    
    async def _test_hybrid_method(self, prompt: str) -> GenerationResult:
        """Test hybrid approach (Retrieval + Control)"""
        start_time = time.time()
        
        try:
            # Step 1: Generate with retrieval
            retrieval_result = await retrieval_humor_generator.generate_with_retrieval(
                prompt=prompt,
                audience="general", 
                style="witty"
            )
            
            base_text = retrieval_result['generated_text']
            
            # Step 2: Apply controlled refinement
            control_vectors = [
                ControlVector(attribute=ControlAttribute.HUMOR, strength=0.5, weight=0.8),
                ControlVector(attribute=ControlAttribute.SURPRISE, strength=0.6, weight=0.7)
            ]
            
            refined_text = await controlled_humor_generator.apply_control_vectors(base_text, control_vectors)
            
            generation_time = time.time() - start_time
            
            # Evaluate final result
            surprise_score = await self.surprise_calculator.calculate_surprise_index(refined_text, prompt)
            
            hybrid_result = GenerationResult(
                method="Hybrid (Retrieval+Control)",
                generated_text=refined_text,
                generation_time=generation_time,
                metadata={
                    "retrieval_confidence": retrieval_result.get('confidence', 0.0),
                    "entities_used": retrieval_result.get('entities_found', []),
                    "control_applied": len(control_vectors),
                    "original_text": base_text,
                    "method_combination": "retrieval_then_control"
                },
                surprise_index=surprise_score
            )
            
            # Add safety and overlap analysis
            await self._add_safety_analysis(hybrid_result)
            await self._add_overlap_metrics(hybrid_result, prompt)
            
            return hybrid_result
            
        except Exception as e:
            print(f"    ❌ Hybrid method failed: {e}")
            return GenerationResult(
                method="Hybrid (Retrieval+Control)",
                generated_text=f"Error: {str(e)}",
                generation_time=time.time() - start_time,
                metadata={"error": str(e)}
            )
    
    async def _add_safety_analysis(self, result: GenerationResult):
        """Add content safety analysis to result"""
        try:
            filter_result = await enhanced_content_filter.enhanced_filter(
                result.generated_text,
                require_sanitization=False
            )
            
            result.toxicity_score = filter_result.overall_toxicity
            result.is_safe = filter_result.is_safe
            
            # Add safety metadata
            result.metadata['safety_analysis'] = {
                "flagged_categories": filter_result.flagged_categories,
                "confidence": filter_result.confidence,
                "suggestions": filter_result.suggestions[:2]  # Top 2 suggestions
            }
            
        except Exception as e:
            print(f"    ⚠️ Safety analysis failed: {e}")
            result.metadata['safety_error'] = str(e)
    
    async def _add_overlap_metrics(self, result: GenerationResult, prompt: str):
        """Add BLEU/ROUGE overlap metrics if reference available"""
        try:
            if prompt in self.reference_responses:
                references = self.reference_responses[prompt]
                
                # Use first reference for metrics calculation
                if references:
                    metrics = humor_metrics.calculate_all_metrics(
                        result.generated_text,
                        references[0]
                    )
                    
                    result.bleu_1 = metrics.bleu_1
                    result.rouge_1 = metrics.rouge_1_f
                    result.rouge_l = metrics.rouge_l_f
                    
                    # Add diversity metrics to metadata
                    result.metadata['overlap_metrics'] = {
                        "distinct_1": metrics.distinct_1,
                        "distinct_2": metrics.distinct_2,
                        "reference_used": references[0]
                    }
                    
        except Exception as e:
            print(f"    ⚠️ Overlap metrics failed: {e}")
            result.metadata['metrics_error'] = str(e)
    
    async def _run_bws_evaluation(self, results: List[GenerationResult]):
        """Run Best-Worst Scaling evaluation on generated results"""
        try:
            # Group results by prompt for BWS comparison
            prompt_groups = {}
            for result in results:
                # Use generation method + text hash as identifier
                item_id = f"{result.method}_{hash(result.generated_text) % 1000}"
                
                bws_item = BWS_Item(
                    id=item_id,
                    text=result.generated_text,
                    metadata={"method": result.method, "generation_time": result.generation_time}
                )
                
                # Add to evaluator
                bws_evaluator.add_items([bws_item])
            
            # Generate comparisons
            comparisons = bws_evaluator.generate_comparisons(n_comparisons=6)
            
            # Simulate some BWS judgments (in practice, would be human judgments)
            await self._simulate_bws_judgments(comparisons)
            
            # Calculate BWS scores
            bws_results = bws_evaluator.calculate_bws_scores()
            
            # Add BWS scores back to results
            for result in results:
                item_id = f"{result.method}_{hash(result.generated_text) % 1000}"
                if item_id in bws_results.item_scores:
                    result.bws_score = bws_results.item_scores[item_id]
            
            print(f"    ✅ BWS evaluation completed with {len(comparisons)} comparisons")
            
        except Exception as e:
            print(f"    ❌ BWS evaluation failed: {e}")
    
    async def _simulate_bws_judgments(self, comparisons):
        """Simulate BWS judgments for demonstration"""
        for i, comparison in enumerate(comparisons[:3]):  # Simulate first 3
            if len(comparison.items) >= 4:
                # Simple heuristic: prefer shorter, more creative-sounding text
                items_with_scores = []
                for item in comparison.items:
                    # Score based on length and keywords
                    score = 0.5
                    if len(item.text) < 60:  # Prefer concise
                        score += 0.2
                    if any(word in item.text.lower() for word in ['unexpected', 'surprising', 'ironic']):
                        score += 0.3
                    items_with_scores.append((item, score))
                
                # Sort by score
                items_with_scores.sort(key=lambda x: x[1], reverse=True)
                
                best_item = items_with_scores[0][0]
                worst_item = items_with_scores[-1][0]
                
                bws_evaluator.record_judgment(
                    comparison_id=comparison.comparison_id,
                    best_item_id=best_item.id,
                    worst_item_id=worst_item.id,
                    user_id=f"sim_user_{i}"
                )
    
    def _generate_comparison_report(self, prompts: List[str], results: List[GenerationResult]) -> ComparisonReport:
        """Generate comprehensive comparison report"""
        
        # Group results by method
        method_results = {}
        for result in results:
            if result.method not in method_results:
                method_results[result.method] = []
            method_results[result.method].append(result)
        
        # Calculate method rankings
        method_rankings = {}
        metrics = ['humor_score', 'creativity_score', 'appropriateness_score', 'surprise_index', 'generation_time']
        
        for metric in metrics:
            method_scores = {}
            for method, method_res in method_results.items():
                scores = [getattr(r, metric) for r in method_res if getattr(r, metric) is not None]
                if scores:
                    if metric == 'generation_time':
                        method_scores[method] = sum(scores) / len(scores)  # Lower is better
                    else:
                        method_scores[method] = sum(scores) / len(scores)  # Higher is better
                else:
                    method_scores[method] = 0.0
            
            method_rankings[metric] = method_scores
        
        # Generate performance summary
        performance_summary = {
            "total_tests": len(results),
            "methods_tested": len(method_results),
            "test_prompts": len(prompts),
            "average_generation_time": {
                method: sum(getattr(r, 'generation_time') for r in results) / len(results)
                for method, results in method_results.items()
            },
            "safety_pass_rate": {
                method: sum(1 for r in results if r.is_safe) / len(results)
                for method, results in method_results.items()
            },
            "methods_overview": {
                method: {
                    "count": len(results),
                    "avg_humor": sum(r.humor_score for r in results) / len(results) if results else 0,
                    "avg_toxicity": sum(r.toxicity_score for r in results) / len(results) if results else 0,
                    "success_rate": sum(1 for r in results if "Error:" not in r.generated_text) / len(results) if results else 0
                }
                for method, results in method_results.items()
            }
        }
        
        # Calculate evaluation correlations (simplified)
        evaluation_correlations = self._calculate_correlations(results)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(method_rankings, performance_summary)
        
        return ComparisonReport(
            test_prompts=prompts,
            generation_results=results,
            performance_summary=performance_summary,
            method_rankings=method_rankings,
            evaluation_correlations=evaluation_correlations,
            recommendations=recommendations
        )
    
    def _calculate_correlations(self, results: List[GenerationResult]) -> Dict[str, float]:
        """Calculate correlations between evaluation metrics"""
        correlations = {}
        
        # Get valid results with scores
        valid_results = [r for r in results if r.humor_score > 0 and r.surprise_index > 0]
        
        if len(valid_results) > 3:
            try:
                humor_scores = [r.humor_score for r in valid_results]
                surprise_scores = [r.surprise_index for r in valid_results]
                
                # Simple correlation calculation
                if len(humor_scores) == len(surprise_scores):
                    mean_humor = sum(humor_scores) / len(humor_scores)
                    mean_surprise = sum(surprise_scores) / len(surprise_scores)
                    
                    numerator = sum((h - mean_humor) * (s - mean_surprise) for h, s in zip(humor_scores, surprise_scores))
                    denom_h = sum((h - mean_humor) ** 2 for h in humor_scores)
                    denom_s = sum((s - mean_surprise) ** 2 for s in surprise_scores)
                    
                    if denom_h > 0 and denom_s > 0:
                        correlations['humor_vs_surprise'] = numerator / (denom_h * denom_s) ** 0.5
                    
            except Exception as e:
                print(f"    ⚠️ Correlation calculation failed: {e}")
        
        return correlations
    
    def _generate_recommendations(self, rankings: Dict[str, Dict[str, float]], summary: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        # Analyze humor performance
        humor_ranking = rankings.get('humor_score', {})
        if humor_ranking:
            best_humor_method = max(humor_ranking.items(), key=lambda x: x[1])
            recommendations.append(f"**Best Humor Performance**: {best_humor_method[0]} (avg: {best_humor_method[1]:.2f})")
        
        # Analyze speed performance
        time_ranking = rankings.get('generation_time', {})
        if time_ranking:
            fastest_method = min(time_ranking.items(), key=lambda x: x[1])
            recommendations.append(f"**Fastest Generation**: {fastest_method[0]} ({fastest_method[1]:.2f}s)")
        
        # Safety analysis
        safety_rates = summary.get('safety_pass_rate', {})
        if safety_rates:
            safest_method = max(safety_rates.items(), key=lambda x: x[1])
            recommendations.append(f"**Safest Content**: {safest_method[0]} ({safest_method[1]:.1%} pass rate)")
        
        # General recommendations
        recommendations.extend([
            "**For Production**: Consider hybrid approach for balance of quality and safety",
            "**For Research**: CrewAI multi-agent provides most detailed evaluation metrics",
            "**For Speed**: Retrieval-augmented generation offers good performance with faster generation",
            "**For Control**: PPLM-style controlled generation allows fine-tuning of specific attributes"
        ])
        
        return recommendations
    
    def export_results_to_csv(self, report: ComparisonReport, filename: str = "humor_evaluation_comparison.csv"):
        """Export detailed results to CSV for analysis"""
        try:
            # Convert results to pandas DataFrame
            data = []
            for result in report.generation_results:
                row = {
                    'method': result.method,
                    'generated_text': result.generated_text,
                    'generation_time': result.generation_time,
                    'humor_score': result.humor_score,
                    'creativity_score': result.creativity_score,
                    'appropriateness_score': result.appropriateness_score,
                    'context_relevance_score': result.context_relevance_score,
                    'surprise_index': result.surprise_index,
                    'toxicity_score': result.toxicity_score,
                    'is_safe': result.is_safe,
                    'bleu_1': result.bleu_1,
                    'rouge_1': result.rouge_1,
                    'rouge_l': result.rouge_l,
                    'bws_score': result.bws_score
                }
                
                # Add metadata as separate columns
                for key, value in result.metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        row[f'meta_{key}'] = value
                
                data.append(row)
            
            df = pd.DataFrame(data)
            df.to_csv(filename, index=False)
            print(f"📊 Results exported to {filename}")
            
        except Exception as e:
            print(f"❌ Export failed: {e}")
    
    def print_summary_table(self, report: ComparisonReport):
        """Print formatted summary table for report"""
        print("\n📋 EVALUATION SUMMARY TABLE")
        print("=" * 80)
        
        methods = list(set(r.method for r in report.generation_results))
        
        # Header
        print(f"{'Method':<25} {'Humor':<8} {'Creative':<9} {'Surprise':<9} {'Safety':<8} {'Time':<8}")
        print("-" * 80)
        
        # Method rows
        for method in methods:
            method_results = [r for r in report.generation_results if r.method == method]
            
            if method_results:
                avg_humor = sum(r.humor_score for r in method_results) / len(method_results)
                avg_creative = sum(r.creativity_score for r in method_results) / len(method_results)
                avg_surprise = sum(r.surprise_index for r in method_results) / len(method_results)
                safety_rate = sum(1 for r in method_results if r.is_safe) / len(method_results)
                avg_time = sum(r.generation_time for r in method_results) / len(method_results)
                
                print(f"{method:<25} {avg_humor:<8.2f} {avg_creative:<9.2f} {avg_surprise:<9.2f} {safety_rate:<8.1%} {avg_time:<8.2f}s")
        
        print("=" * 80)
        print("Higher scores are better for Humor, Creative, Surprise, Safety")
        print("Lower scores are better for Time")

# Global instance
evaluation_comparison_system = EvaluationComparisonSystem() 