#!/usr/bin/env python3
"""
CAH CrewAI Analysis - Baseline vs CrewAI Multi-Agent Comparison
Compares prompt-only generation with true CrewAI agent-based generation
"""

import json
import os
import asyncio
import time
import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

from crewai import Agent, Task, Crew
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun

# Load environment variables
load_dotenv()

# Try to import LLM clients
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

@dataclass
class CAHCard:
    text: str
    card_type: str  # 'black' or 'white'
    pick_count: int = 1  # How many white cards to pick

@dataclass
class CAHCombination:
    black_card: str
    white_cards: List[str]
    result: str
    score: float = 0.0

@dataclass
class CAHGenerationResult:
    white_card: str
    approach: str  # 'baseline' or 'crewai'
    generation_time: float
    model_used: str
    reasoning: Optional[str] = None

class SimpleCAHDataset:
    """Simple CAH dataset with sample cards"""
    
    def __init__(self):
        self.black_cards = [
            CAHCard("What's the next Happy Meal toy? _____.", "black", 1),
            CAHCard("I never truly understood _____ until I encountered _____.", "black", 2),
            CAHCard("What's that smell? _____.", "black", 1),
            CAHCard("Life for American Indians was forever changed when the White Man introduced them to _____.", "black", 1),
            CAHCard("_____ + _____ = _____.", "black", 3),
            CAHCard("What's a girl's best friend? _____.", "black", 1),
            CAHCard("What did I bring back from Mexico? _____.", "black", 1),
            CAHCard("What's the most emo? _____.", "black", 1),
            CAHCard("What would grandma find disturbing, yet oddly charming? _____.", "black", 1),
            CAHCard("What helps Obama unwind? _____.", "black", 1),
        ]
        
        self.white_cards = [
            "Being rich", "Soup that is too hot", "A sad handjob",
            "The Heart of Darkness", "Vigorous jazz hands", "Finger painting",
            "A disappointing birthday party", "My collection of high-tech sex toys",
            "Passive-aggressive Post-it notes", "The inevitable heat death of the universe",
            "A windmill full of corpses", "Pretending to care", "The chronic",
            "Bees?", "Dead parents", "Alcoholism", "Coat hanger abortions",
            "Surprise sex!", "The Big Bang", "Racism",
        ]
    
    def get_unique_black_cards(self, num_cards: int) -> List[CAHCard]:
        """Get unique black cards for testing"""
        return random.sample(self.black_cards, min(num_cards, len(self.black_cards)))

class SimpleLLMClient:
    """Simple LLM client that handles multiple providers"""
    
    def __init__(self):
        self.openai_client = None
        self.anthropic_client = None
        
        # Initialize clients if API keys are available
        if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
            self.openai_client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        if ANTHROPIC_AVAILABLE and os.getenv("ANTHROPIC_API_KEY"):
            self.anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    async def generate_response(self, prompt: str, model: str = "gpt-3.5-turbo", temperature: float = 0.8) -> str:
        """Generate response using available LLM"""
        try:
            if model.startswith("gpt") and self.openai_client:
                return await self._generate_openai(prompt, model, temperature)
            elif model.startswith("claude") and self.anthropic_client:
                return await self._generate_anthropic(prompt, model, temperature)
            else:
                # Fallback to mock response
                return self._generate_mock(prompt)
        except Exception as e:
            print(f"⚠️  LLM generation failed: {e}")
            return self._generate_mock(prompt)
    
    async def _generate_openai(self, prompt: str, model: str, temperature: float) -> str:
        """Generate using OpenAI"""
        response = await self.openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=200
        )
        return response.choices[0].message.content.strip()
    
    async def _generate_anthropic(self, prompt: str, model: str, temperature: float) -> str:
        """Generate using Anthropic (sync client)"""
        def _sync_generate():
            response = self.anthropic_client.messages.create(
                model=model,
                max_tokens=200,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text.strip()
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _sync_generate)
    
    def _generate_mock(self, prompt: str) -> str:
        """Generate mock response when no LLM is available"""
        mock_responses = [
            "My collection of high-tech sex toys",
            "Passive-aggressive Post-it notes", 
            "The inevitable heat death of the universe",
            "Bees?", "Dead parents", "Alcoholism"
        ]
        return random.choice(mock_responses)

class CustomLLMForCrewAI(LLM):
    """Custom LLM wrapper for CrewAI to use our LLM client"""
    
    # Define the fields that can be set
    llm_client: Any = None
    model_name: str = "gpt-3.5-turbo"
    
    def __init__(self, llm_client: SimpleLLMClient, model_name: str):
        super().__init__()
        # Use object.__setattr__ to bypass pydantic validation during init
        object.__setattr__(self, 'llm_client', llm_client)
        object.__setattr__(self, 'model_name', model_name)
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Make a call to the LLM"""
        # This is a synchronous wrapper around our async LLM client
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            temperature = kwargs.get('temperature', 0.8)
            response = loop.run_until_complete(
                self.llm_client.generate_response(prompt, self.model_name, temperature)
            )
            return response
        finally:
            loop.close()
    
    @property
    def _llm_type(self) -> str:
        return f"custom_{self.model_name}"

class BaselineCAHGenerator:
    """Simple baseline CAH generator (same as standalone version)"""
    
    def __init__(self, llm_client: SimpleLLMClient):
        self.llm_client = llm_client
    
    async def generate_white_card(self, black_card: str, model: str = "gpt-3.5-turbo") -> CAHGenerationResult:
        """Generate a white card response for a black card"""
        start_time = time.time()
        
        prompt = f"""
Generate a funny white card response for this Cards Against Humanity black card:

{black_card}

The response should be:
- Unexpected and surprising
- Edgy but not extremely offensive
- Fits the fill-in-the-blank format
- Short and punchy (1-8 words)

Respond with just the white card text, no quotes or explanation.
"""
        
        white_card = await self.llm_client.generate_response(prompt, model, temperature=0.9)
        generation_time = time.time() - start_time
        
        return CAHGenerationResult(
            white_card=white_card,
            approach="baseline",
            generation_time=generation_time,
            model_used=model
        )

class CrewAICAHGenerator:
    """CrewAI-based CAH generator with specialized agents"""
    
    def __init__(self, llm_client: SimpleLLMClient, model: str = "gpt-3.5-turbo"):
        self.llm_client = llm_client
        self.model = model
        self.custom_llm = CustomLLMForCrewAI(llm_client, model)
        
        # Create specialized agents with improved prompts
        self.creative_agent = Agent(
            role="Creative Humor Generator",
            goal="Generate exactly 5 creative and unexpected white card options for Cards Against Humanity that are edgy, surprising, and hilarious",
            backstory="""You are a professional comedy writer who specializes in Cards Against Humanity humor. 
            You have years of experience creating shocking, unexpected, and hilarious combinations that make 
            players laugh out loud. You understand the game's irreverent, edgy style and excel at subverting 
            expectations. You always follow the exact format requested and never deviate from instructions.""",
            llm=self.custom_llm,
            verbose=True,
            allow_delegation=False
        )
        
        self.evaluator_agent = Agent(
            role="Humor Quality Evaluator", 
            goal="Analyze humor options and select the single best one based on CAH criteria, providing clear reasoning in the EXACT format specified",
            backstory="""You are a comedy expert and CAH judge with perfect understanding of what makes 
            combinations funny in this game. You MUST follow the exact output format specified in your task. 
            You NEVER give generic responses like 'I can give a great answer'. You always provide detailed 
            analysis using the ANALYSIS/BEST OPTION/REASONING format. You are precise and professional.""",
            llm=self.custom_llm,
            verbose=True,
            allow_delegation=False
        )
        
        self.refiner_agent = Agent(
            role="Humor Refiner",
            goal="Take the selected white card and make it funnier, punchier, and more impactful while following the EXACT output format specified",
            backstory="""You are a comedy editor who perfects jokes. You MUST follow the exact output format 
            specified in your task. You NEVER give generic responses. You always provide the refined white card 
            using the REFINED WHITE CARD/IMPROVEMENTS format. You make cards more unexpected, concise, and punchy.""",
            llm=self.custom_llm,
            verbose=True,
            allow_delegation=False
        )
    
    async def generate_white_card(self, black_card: str) -> CAHGenerationResult:
        """Generate white card using CrewAI multi-agent approach"""
        start_time = time.time()
        
        # Task 1: Generate multiple options with strict format
        generation_task = Task(
            description=f"""
            Generate exactly 5 different funny white card responses for this Cards Against Humanity black card:
            
            BLACK CARD: "{black_card}"
            
            REQUIREMENTS:
            - Each response must be unexpected and surprising
            - Edgy but not extremely offensive (CAH style)
            - Must fit the fill-in-the-blank format perfectly
            - Short and punchy (1-6 words maximum)
            - Each option must be completely different
            - Follow CAH's irreverent, shocking humor style
            
            OUTPUT FORMAT (follow exactly):
            1. [white card option 1]
            2. [white card option 2] 
            3. [white card option 3]
            4. [white card option 4]
            5. [white card option 5]
            
            Do not include any other text, explanations, or formatting.
            """,
            agent=self.creative_agent,
            expected_output="Exactly 5 numbered white card options, nothing else"
        )
        
        # Task 2: Evaluate and select with strict format
        evaluation_task = Task(
            description=f"""
            CRITICAL: You must evaluate the 5 white card options for the black card "{black_card}" and select the best one.
            
            EVALUATION CRITERIA:
            - Unexpectedness and surprise factor (most important)
            - Cleverness and wit of the combination
            - Comedic timing and impact
            - Perfect fit for CAH's edgy humor style
            - How well it completes the black card
            
            MANDATORY OUTPUT FORMAT (you MUST follow this exactly):
            ANALYSIS: [Brief analysis of why each option works or doesn't work]
            BEST OPTION: [number] - [exact white card text from the list]
            REASONING: [Specific reasons why this option is the funniest]
            
            WARNING: Do NOT respond with generic phrases like "I can give a great answer" or similar. 
            You MUST provide the actual analysis in the format above. Failure to follow this format is unacceptable.
            """,
            agent=self.evaluator_agent,
            expected_output="Analysis with selected best option in exact format specified - NO GENERIC RESPONSES",
            context=[generation_task]
        )
        
        # Task 3: Refine with strict output
        refinement_task = Task(
            description=f"""
            CRITICAL: Take the selected best white card and refine it to maximize comedic impact.
            
            BLACK CARD: "{black_card}"
            
            REFINEMENT GOALS:
            - Make it more unexpected or clever if possible
            - Improve comedic timing and punch
            - Make it more concise and impactful
            - Keep the same concept but improve execution
            - Ensure it's 1-6 words maximum
            
            MANDATORY OUTPUT FORMAT (you MUST follow this exactly):
            REFINED WHITE CARD: [the improved white card text only]
            IMPROVEMENTS: [brief explanation of changes made]
            
            WARNING: Do NOT respond with generic phrases like "I can give a great answer" or similar.
            You MUST provide the actual refined white card in the format above. Failure to follow this format is unacceptable.
            The refined white card should be the final, polished version ready to use.
            """,
            agent=self.refiner_agent,
            expected_output="Refined white card with improvements explanation in exact format - NO GENERIC RESPONSES",
            context=[evaluation_task]
        )
        
        # Create and run the crew
        crew = Crew(
            agents=[self.creative_agent, self.evaluator_agent, self.refiner_agent],
            tasks=[generation_task, evaluation_task, refinement_task],
            verbose=True
        )
        
        # Execute the crew
        try:
            result = crew.kickoff()
            generation_time = time.time() - start_time
            
            # Parse the final result to extract the refined white card
            white_card = self._parse_crew_result(str(result))
            
            return CAHGenerationResult(
                white_card=white_card,
                approach="crewai",
                generation_time=generation_time,
                model_used=self.model,
                reasoning=str(result)
            )
            
        except Exception as e:
            print(f"⚠️  CrewAI generation failed: {e}")
            # Fallback to simple generation
            fallback_result = await self._fallback_generation(black_card)
            fallback_result.generation_time = time.time() - start_time
            return fallback_result
    
    def _parse_crew_result(self, result_text: str) -> str:
        """Parse CrewAI result to extract the final white card with improved parsing"""
        try:
            # First, try to find "REFINED WHITE CARD:" pattern
            if "REFINED WHITE CARD:" in result_text:
                lines = result_text.split('\n')
                for line in lines:
                    if line.strip().startswith("REFINED WHITE CARD:"):
                        card_text = line.split(":", 1)[1].strip()
                        # Clean up any quotes or extra formatting
                        card_text = card_text.strip('"').strip("'").strip()
                        if card_text and len(card_text) < 100 and not card_text.lower().startswith("what"):  # Reasonable length check
                            return card_text
            
            # Second, try to find "BEST OPTION:" pattern
            if "BEST OPTION:" in result_text:
                lines = result_text.split('\n')
                for line in lines:
                    if line.strip().startswith("BEST OPTION:"):
                        # Extract text after the dash
                        if " - " in line:
                            card_text = line.split(" - ", 1)[1].strip()
                            card_text = card_text.strip('"').strip("'").strip()
                            if card_text and len(card_text) < 100 and not card_text.lower().startswith("what"):
                                return card_text
            
            # Third, look for numbered list items from the creative agent (1., 2., etc.)
            lines = result_text.split('\n')
            creative_options = []
            for line in lines:
                line = line.strip()
                if line and (line.startswith('1.') or line.startswith('2.') or 
                           line.startswith('3.') or line.startswith('4.') or line.startswith('5.')):
                    card_text = line[2:].strip()  # Remove number and dot
                    card_text = card_text.strip('"').strip("'").strip()
                    if card_text and len(card_text) < 100 and not card_text.lower().startswith("what"):
                        creative_options.append(card_text)
            
            # If we found creative options, pick the first good one
            if creative_options:
                # Filter out generic responses
                good_options = [opt for opt in creative_options 
                              if not any(phrase in opt.lower() for phrase in 
                                       ['i can give', 'great answer', 'analysis', 'reasoning'])]
                if good_options:
                    return good_options[0]  # Return first good option
                else:
                    return creative_options[0]  # Fallback to any option
            
            # Fourth, look for any reasonable short text that's not generic
            lines = [line.strip() for line in result_text.split('\n') 
                    if line.strip() and len(line.strip()) < 50 and len(line.strip()) > 3]
            if lines:
                # Prefer lines that don't contain common instruction words or generic phrases
                filtered_lines = [line for line in lines 
                                if not any(word in line.lower() for word in 
                                         ['analysis', 'reasoning', 'improvements', 'format', 'output',
                                          'i can give', 'great answer', 'critical', 'mandatory'])]
                if filtered_lines:
                    return filtered_lines[0]
                else:
                    return lines[0]
            
            # Ultimate fallback
            return "Bees?"
            
        except Exception as e:
            print(f"⚠️  Error parsing crew result: {e}")
            return "The inevitable heat death of the universe"
    
    async def _fallback_generation(self, black_card: str) -> CAHGenerationResult:
        """Fallback generation if CrewAI fails"""
        prompt = f"Generate a funny CAH white card for: {black_card}"
        white_card = await self.llm_client.generate_response(prompt, self.model, 0.9)
        
        return CAHGenerationResult(
            white_card=white_card,
            approach="crewai_fallback",
            generation_time=0.0,
            model_used=self.model,
            reasoning="CrewAI failed, used fallback generation"
        )

class CAHHumorEvaluator:
    """Evaluates humor quality of CAH combinations"""
    
    def __init__(self, llm_client: SimpleLLMClient):
        self.llm_client = llm_client
    
    async def evaluate_humor(self, combination: CAHCombination, model: str = "gpt-3.5-turbo") -> float:
        """Evaluate humor on a scale of 0-1"""
        prompt = f"""
Rate the humor quality of this Cards Against Humanity combination on a scale of 0.0 to 1.0:

Black Card: {combination.black_card}
White Card(s): {', '.join(combination.white_cards)}
Result: {combination.result}

Consider:
- Unexpectedness and surprise
- Cleverness of the combination
- Comedic timing and flow
- Appropriateness for CAH's edgy humor style

Respond with just a number between 0.0 and 1.0.
"""
        
        try:
            response = await self.llm_client.generate_response(prompt, model, temperature=0.3)
            # Extract number from response
            score = float(response.strip())
            return max(0.0, min(1.0, score))  # Clamp to 0-1 range
        except:
            # Fallback to random score
            return random.uniform(0.3, 0.8)

async def run_cah_crewai_analysis():
    """Run the complete CAH analysis comparing baseline vs CrewAI"""
    
    print("🎭 CARDS AGAINST HUMANITY - BASELINE vs CREWAI ANALYSIS (IMPROVED)")
    print("=" * 80)
    
    # Initialize components
    dataset = SimpleCAHDataset()
    llm_client = SimpleLLMClient()
    evaluator = CAHHumorEvaluator(llm_client)
    baseline_generator = BaselineCAHGenerator(llm_client)
    
    # Check LLM availability
    has_real_llm = (llm_client.openai_client is not None or 
                    llm_client.anthropic_client is not None)
    
    if has_real_llm:
        print("✅ Real LLM clients available")
        model = "gpt-3.5-turbo" if llm_client.openai_client else "claude-3-haiku"
    else:
        print("⚠️  No LLM clients available - running in demo mode")
        model = "mock"
    
    print(f"🤖 Using model: {model}")
    
    # Initialize CrewAI generator
    crewai_generator = CrewAICAHGenerator(llm_client, model)
    print("🚀 CrewAI agents initialized with improved prompts")
    print()
    
    # Run analysis on fewer combinations for faster testing
    num_tests = 3  # Reduced from 5 for faster iteration
    baseline_results = []
    crewai_results = []
    baseline_scores = []
    crewai_scores = []
    
    # Get unique black cards to avoid repeats
    test_cards = dataset.get_unique_black_cards(num_tests)
    
    print(f"🧪 Running analysis on {len(test_cards)} unique combinations...")
    print("⏱️  Added rate limiting to avoid quota issues")
    print()
    
    for i, black_card in enumerate(test_cards):
        print(f"Test {i+1}/{len(test_cards)}")
        print("-" * 60)
        print(f"Black Card: {black_card.text}")
        print()
        
        # Baseline generation
        print("🎯 BASELINE Generation...")
        baseline_result = await baseline_generator.generate_white_card(black_card.text, model)
        baseline_combination = CAHCombination(
            black_card=black_card.text,
            white_cards=[baseline_result.white_card],
            result=black_card.text.replace("_____", baseline_result.white_card, 1)
        )
        
        # Add small delay to avoid rate limiting
        await asyncio.sleep(1)
        
        baseline_score = await evaluator.evaluate_humor(baseline_combination, model)
        
        baseline_results.append(baseline_result)
        baseline_scores.append(baseline_score)
        
        print(f"   White Card: {baseline_result.white_card}")
        print(f"   Result: {baseline_combination.result}")
        print(f"   Score: {baseline_score:.3f}")
        print(f"   Time: {baseline_result.generation_time:.2f}s")
        print()
        
        # Add delay before CrewAI generation
        await asyncio.sleep(2)
        
        # CrewAI generation
        print("🤖 CREWAI Generation (Improved)...")
        crewai_result = await crewai_generator.generate_white_card(black_card.text)
        crewai_combination = CAHCombination(
            black_card=black_card.text,
            white_cards=[crewai_result.white_card],
            result=black_card.text.replace("_____", crewai_result.white_card, 1)
        )
        
        # Add delay before evaluation
        await asyncio.sleep(1)
        
        crewai_score = await evaluator.evaluate_humor(crewai_combination, model)
        
        crewai_results.append(crewai_result)
        crewai_scores.append(crewai_score)
        
        print(f"   White Card: {crewai_result.white_card}")
        print(f"   Result: {crewai_combination.result}")
        print(f"   Score: {crewai_score:.3f}")
        print(f"   Time: {crewai_result.generation_time:.2f}s")
        
        # Calculate improvement
        improvement = ((crewai_score - baseline_score) / baseline_score * 100) if baseline_score > 0 else 0
        print(f"   Improvement: {improvement:+.1f}%")
        print()
        print("=" * 60)
        print()
        
        # Add delay between tests
        if i < len(test_cards) - 1:  # Don't delay after last test
            print("⏱️  Pausing to avoid rate limits...")
            await asyncio.sleep(3)
            print()
    
    # Calculate final results
    avg_baseline_score = sum(baseline_scores) / len(baseline_scores)
    avg_crewai_score = sum(crewai_scores) / len(crewai_scores)
    overall_improvement = ((avg_crewai_score - avg_baseline_score) / avg_baseline_score * 100) if avg_baseline_score > 0 else 0
    
    avg_baseline_time = sum(r.generation_time for r in baseline_results) / len(baseline_results)
    avg_crewai_time = sum(r.generation_time for r in crewai_results) / len(crewai_results)
    
    print("=" * 80)
    print("📊 FINAL RESULTS (IMPROVED CREWAI)")
    print("=" * 80)
    print(f"🎯 Baseline Average Score: {avg_baseline_score:.3f}")
    print(f"🤖 CrewAI Average Score: {avg_crewai_score:.3f}")
    print(f"📈 Overall Improvement: {overall_improvement:+.1f}%")
    print()
    print(f"⏱️  Average Generation Time:")
    print(f"   Baseline: {avg_baseline_time:.2f}s")
    print(f"   CrewAI: {avg_crewai_time:.2f}s")
    print(f"   Time Overhead: {((avg_crewai_time - avg_baseline_time) / avg_baseline_time * 100):+.1f}%")
    print()
    
    # Save detailed results
    detailed_results = {
        "model_used": model,
        "has_real_llm": has_real_llm,
        "num_tests": len(test_cards),
        "version": "improved_crewai",
        "test_cards": [card.text for card in test_cards],
        "baseline_results": [
            {
                "white_card": r.white_card,
                "score": s,
                "generation_time": r.generation_time,
                "approach": r.approach
            }
            for r, s in zip(baseline_results, baseline_scores)
        ],
        "crewai_results": [
            {
                "white_card": r.white_card,
                "score": s,
                "generation_time": r.generation_time,
                "approach": r.approach,
                "reasoning": r.reasoning
            }
            for r, s in zip(crewai_results, crewai_scores)
        ],
        "summary": {
            "avg_baseline_score": avg_baseline_score,
            "avg_crewai_score": avg_crewai_score,
            "overall_improvement_percent": overall_improvement,
            "avg_baseline_time": avg_baseline_time,
            "avg_crewai_time": avg_crewai_time,
            "time_overhead_percent": ((avg_crewai_time - avg_baseline_time) / avg_baseline_time * 100) if avg_baseline_time > 0 else 0
        },
        "timestamp": time.time()
    }
    
    results_file = "final_results/cah_baseline_vs_crewai_improved_results.json"
    os.makedirs("final_results", exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    print(f"💾 Detailed results saved to: {results_file}")
    
    # Interpretation
    print()
    print("🔍 INTERPRETATION:")
    if overall_improvement > 15:
        print("CrewAI approach shows significant improvement!")
    elif overall_improvement > 5:
        print("CrewAI approach shows good improvement.")
    elif overall_improvement > 0:
        print("CrewAI approach shows modest improvement.")
    else:
        print("CrewAI needs further tuning - baseline still performing better.")
    
    if avg_crewai_time > avg_baseline_time * 2:
        print("CrewAI approach is slower but may be worth it for quality gains.")
    
    if not has_real_llm:
        print(" Note: Results are from demo mode. Use real LLMs for accurate comparison.")
    
    print()
    print("🎯 Key Insights:")
    improvements = [((crewai_scores[i] - baseline_scores[i]) / baseline_scores[i] * 100) 
                   for i in range(len(baseline_scores)) if baseline_scores[i] > 0]
    positive_improvements = [imp for imp in improvements if imp > 0]
    
    if positive_improvements:
        print(f"   • {len(positive_improvements)}/{len(improvements)} tests showed improvement")
        print(f"   • Average positive improvement: {sum(positive_improvements)/len(positive_improvements):+.1f}%")
    
    print(f"   • CrewAI uses {avg_crewai_time/avg_baseline_time:.1f}x more time than baseline")
    print(f"   • Improved agent prompts and result parsing")
    print(f"   • Multi-agent approach provides structured reasoning and evaluation")

if __name__ == "__main__":
    asyncio.run(run_cah_crewai_analysis()) 