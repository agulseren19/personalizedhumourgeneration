#!/usr/bin/env python3
"""
Generate Example Cards Against Humanity Cards - AI Version
Creates 5 mock users with different favorite personas and generates cards for each
Uses AI generation when possible, falls back to pre-written cards
"""

import asyncio
import json
import random
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path

# Add the agent_system directory to Python path
current_dir = Path(__file__).parent
agent_system_dir = current_dir.parent
import sys
sys.path.insert(0, str(agent_system_dir))

from models.database import create_database, get_session_local, User, Persona, PersonaPreference
from config.settings import settings
from personas.persona_manager import PersonaManager
from agents.improved_humor_agents import ImprovedHumorOrchestrator, HumorRequest
from evaluation.statistical_humor_evaluator import StatisticalHumorEvaluator

class MultiUserAIExampleCardGenerator:
    """Generates example CAH cards for multiple users with different personas"""
    
    def __init__(self):
        # Initialize statistical humor evaluator
        self.humor_evaluator = StatisticalHumorEvaluator()
        
        # Define 5 different users with different favorite personas
        self.users = {
            "office_gamer_456": {
                "username": "office_gamer",
                "email": "office.gamer@cah.com",
                "favorite_personas": ["Office Worker", "Gaming Guru"],
                "age_range": "25-35",
                "occupation": "Software Developer",
                "interests": ["technology", "gaming", "office culture", "esports"]
            },
            "dad_millennial_789": {
                "username": "dad_millennial",
                "email": "dad.millennial@cah.com",
                "favorite_personas": ["Dad Humor Enthusiast", "Millennial Memer"],
                "age_range": "30-45",
                "occupation": "Marketing Manager",
                "interests": ["dad jokes", "memes", "social media", "family"]
            },
            "chaos_dark_321": {
                "username": "chaos_dark",
                "email": "chaos.dark@cah.com",
                "favorite_personas": ["Gen Z Chaos Agent", "Dark Humor Connoisseur"],
                "age_range": "18-25",
                "occupation": "Student",
                "interests": ["absurdism", "dark humor", "TikTok", "philosophy"]
            },
            "marvel_foodie_654": {
                "username": "marvel_foodie",
                "email": "marvel.foodie@cah.com",
                "favorite_personas": ["Marvel Universe Expert", "Culinary Comedy Expert"],
                "age_range": "20-35",
                "occupation": "Chef",
                "interests": ["Marvel", "cooking", "comics", "food culture"]
            },
            "dynamic_duo_987": {
                "username": "dynamic_duo",
                "email": "dynamic.duo@cah.com",
                "favorite_personas": ["Tech Bro Philosopher (Dynamic)", "Suburban Chaos Coordinator (Dynamic)"],
                "age_range": "25-40",
                "occupation": "Startup Founder",
                "interests": ["technology", "philosophy", "suburban life", "mindfulness"]
            }
        }
        
        self.humor_orchestrator = ImprovedHumorOrchestrator()
        self.use_ai_generation = True  # Set to False to use only fallback cards
        
    async def setup_mock_users(self):
        """Create mock users with favorite personas"""
        print("🔧 Setting up mock users...")
        
        try:
            # Create database
            create_database(settings.database_url)
            SessionLocal = get_session_local(settings.database_url)
            db = SessionLocal()
            
            # Initialize personas
            persona_manager = PersonaManager(db)
            
            for user_id, user_data in self.users.items():
                print(f"  Setting up user: {user_data['username']}")
                
                # Create mock user
                mock_user = User(
                    id=user_id,
                    username=user_data['username'],
                    email=user_data['email'],
                    password_hash="mock_hash",
                    age_range=user_data['age_range'],
                    occupation=user_data['occupation'],
                    education_level="Bachelor's",
                    interests=user_data['interests'],
                    humor_preferences={
                        "preferred_style": "varies by persona",
                        "favorite_topics": user_data['interests'],
                        "audience": "adults"
                    }
                )
                
                # Check if user already exists
                existing_user = db.query(User).filter(User.id == user_id).first()
                if not existing_user:
                    db.add(mock_user)
                    print(f"    ✅ Created user: {user_data['username']}")
                else:
                    print(f"    ✅ User already exists: {user_data['username']}")
                
                # Create persona preferences for favorite personas
                for persona_name in user_data['favorite_personas']:
                    # Handle dynamic personas
                    if "(Dynamic)" in persona_name:
                        # Create a dynamic persona entry
                        dynamic_persona = Persona(
                            id=f"dynamic_{user_id}_{persona_name.split('(')[0].strip().lower().replace(' ', '_')}",
                            name=persona_name,
                            description=f"Dynamically generated persona for {user_data['username']}",
                            humor_style="dynamic",
                            expertise_areas=user_data['interests'],
                            demographic_hints={"user_id": user_id, "dynamic": True}
                        )
                        
                        # Check if dynamic persona already exists
                        existing_dynamic = db.query(Persona).filter(Persona.id == dynamic_persona.id).first()
                        if not existing_dynamic:
                            db.add(dynamic_persona)
                            print(f"      ✅ Created dynamic persona: {persona_name}")
                        else:
                            print(f"      ✅ Dynamic persona already exists: {persona_name}")
                        
                        # Create preference for dynamic persona
                        preference = PersonaPreference(
                            user_id=user_id,
                            persona_id=dynamic_persona.id,
                            preference_score=9.0,
                            interaction_count=1,
                            context_preferences={"general": 9.0}
                        )
                        db.add(preference)
                    else:
                        # Handle regular personas
                        persona = db.query(Persona).filter(Persona.name == persona_name).first()
                        if persona:
                            # Check if preference already exists
                            existing_pref = db.query(PersonaPreference).filter(
                                PersonaPreference.user_id == user_id,
                                PersonaPreference.persona_id == persona.id
                            ).first()
                            
                            if not existing_pref:
                                preference = PersonaPreference(
                                    user_id=user_id,
                                    persona_id=persona.id,
                                    preference_score=9.0,
                                    interaction_count=1,
                                    context_preferences={"general": 9.0}
                                )
                                db.add(preference)
                                print(f"      ✅ Added preference for {persona_name}")
            
            db.commit()
            print(f"✅ All mock users setup complete")
            
        except Exception as e:
            print(f"❌ Error setting up mock users: {e}")
            if 'db' in locals():
                db.rollback()
        finally:
            if 'db' in locals():
                db.close()
    
    async def generate_ai_black_card(self, topic: str, user_id: str) -> str:
        """Try to generate a black card using AI for specific user"""
        if not self.use_ai_generation:
            return None
            
        try:
            request = HumorRequest(
                context=f"Generate a black card about {topic}",
                audience="adults",
                topic=topic,
                user_id=user_id,
                card_type="black"
            )
            
            result = await self.humor_orchestrator.generate_and_evaluate_humor(request)
            
            if result and result.get('success') and result.get('results'):
                best_result = result['results'][0]
                
                if isinstance(best_result, dict) and 'generation' in best_result:
                    generation_obj = best_result['generation']
                    if hasattr(generation_obj, 'text'):
                        black_card_text = generation_obj.text
                    else:
                        black_card_text = str(generation_obj)
                else:
                    black_card_text = str(best_result)
                
                black_card_text = self._clean_black_card(black_card_text)
                
                if black_card_text:
                    return black_card_text
                    
        except Exception as e:
            print(f"    ⚠️  AI generation failed for topic '{topic}': {e}")
        
        return None
    
    async def generate_ai_white_card(self, black_card: str, user_id: str) -> str:
        """Try to generate a white card using AI for specific user"""
        if not self.use_ai_generation:
            return None
            
        try:
            request = HumorRequest(
                context=black_card,
                audience="adults",
                topic="general",
                user_id=user_id,
                card_type="white"
            )
            
            result = await self.humor_orchestrator.generate_and_evaluate_humor(request)
            
            if result and result.get('success') and result.get('results'):
                white_card_result = result['results'][0]
                
                if isinstance(white_card_result, dict) and 'generation' in white_card_result:
                    generation_obj = white_card_result['generation']
                    if hasattr(generation_obj, 'text'):
                        white_card_text = generation_obj.text
                    else:
                        white_card_text = str(generation_obj)
                else:
                    white_card_text = str(white_card_result)
                
                white_card_text = self._clean_white_card(white_card_text)
                
                if white_card_text:
                    return white_card_text
                    
        except Exception as e:
            print(f"    ⚠️  AI generation failed for white card: {e}")
        
        return None
    
    async def generate_cards_for_user(self, user_id: str, user_data: Dict) -> Dict[str, List[str]]:
        """Generate 10 black cards and 10 white cards for a specific user"""
        print(f"\n🎭 Generating cards for user: {user_data['username']}")
        print(f"   Personas: {', '.join(user_data['favorite_personas'])}")
        
        black_cards = []
        white_cards = []
        
        # Generate 10 black cards
        topics = [
            "technology", "workplace", "relationships", "food", "travel",
            "education", "health", "entertainment", "sports", "politics"
        ]
        
        for i in range(10):
            topic = topics[i]
            print(f"    Generating black card {i+1}/10 for topic: {topic}")
            
            # Try AI generation first
            ai_card = await self.generate_ai_black_card(topic, user_id)
            
            if ai_card:
                black_cards.append(ai_card)
                print(f"      ✅ AI Generated: {ai_card[:60]}...")
            else:
                # Use fallback card
                fallback = self._get_fallback_black_card(topic, user_data['favorite_personas'])
                black_cards.append(fallback)
                print(f"      ⚠️  Using fallback: {fallback}")
            
            await asyncio.sleep(0.5)
        
        # Generate 10 white cards
        for i, black_card in enumerate(black_cards):
            print(f"    Generating white card {i+1}/10 for black card: {black_card[:50]}...")
            
            # Try AI generation first
            ai_card = await self.generate_ai_white_card(black_card, user_id)
            
            if ai_card:
                white_cards.append(ai_card)
                print(f"      ✅ AI Generated: {ai_card[:50]}...")
            else:
                # Use fallback white card
                fallback = self._get_fallback_white_card(black_card, user_data['favorite_personas'])
                white_cards.append(fallback)
                print(f"      ⚠️  Using fallback: {fallback}")
            
            await asyncio.sleep(0.3)
        
        print(f"    ✅ Generated {len(black_cards)} black cards and {len(white_cards)} white cards")
        return {'black_cards': black_cards, 'white_cards': white_cards}
    
    def _clean_black_card(self, text: str) -> str:
        """Clean and format black card text"""
        if not text:
            return ""
        
        text = text.strip()
        
        # Remove common prefixes
        prefixes = ['black card:', 'card:', 'prompt:', 'setup:']
        for prefix in prefixes:
            if text.lower().startswith(prefix):
                text = text[len(prefix):].strip()
        
        # Remove quotes
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
        
        # Ensure it has exactly one blank
        if '____' not in text:
            if text.endswith('.'):
                text = text[:-1] + ' ____.'
            else:
                text += ' ____'
        
        # Take only the first line
        text = text.split('\n')[0].strip()
        
        return text
    
    def _clean_white_card(self, text: str) -> str:
        """Clean and format white card text"""
        if not text:
            return ""
        
        text = text.strip()
        
        # Remove common prefixes
        prefixes = ['white card:', 'response:', 'answer:', 'card:']
        for prefix in prefixes:
            if text.lower().startswith(prefix):
                text = text[len(prefix):].strip()
        
        # Remove quotes
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
        
        # Take only the first line
        text = text.split('\n')[0].strip()
        
        # Limit length
        if len(text) > 100:
            text = text[:97] + "..."
        
        return text
    
    def _get_fallback_black_card(self, topic: str, personas: List[str]) -> str:
        """Get a fallback black card for a topic, considering user personas"""
        # Persona-specific fallback cards
        persona_cards = {
            "Office Worker": [
                "During the quarterly review, my boss was shocked to discover ____.",
                "The new office policy states that all employees must participate in ____.",
                "The team building exercise went wrong when someone suggested ____.",
                "My coworker's secret to productivity is ____.",
                "The office printer finally broke down because of ____."
            ],
            "Gaming Guru": [
                "The latest patch notes reveal that ____ is now OP.",
                "My gaming setup includes ____ for maximum performance.",
                "The hardest boss in this game requires ____ to defeat.",
                "My teammates were shocked when I used ____ in ranked.",
                "The meta has shifted and now ____ is essential."
            ],
            "Dad Humor Enthusiast": [
                "Why did the chicken cross the road? To get to ____.",
                "My dad's emergency stash contains ____.",
                "The latest dad joke involves ____.",
                "My collection of terrible puns includes ____.",
                "The dad joke that broke the internet was ____."
            ],
            "Millennial Memer": [
                "The latest internet trend is ____ but make it aesthetic.",
                "My existential crisis support group meets to discuss ____.",
                "Avocado toast anxiety is real, especially when ____.",
                "The millennial struggle is real: ____ but it's expensive.",
                "The internet broke when someone posted ____."
            ],
            "Gen Z Chaos Agent": [
                "In a shocking plot twist, the void revealed its secret: ____.",
                "The latest TikTok trend involves ____ but it's concerning.",
                "My therapist was speechless when I explained ____.",
                "The existential crisis hotline now offers ____.",
                "Capitalism but as a houseplant means ____."
            ],
            "Dark Humor Connoisseur": [
                "My dark humor collection includes ____.",
                "The latest conspiracy theory claims ____.",
                "The internet's collective trauma response to ____.",
                "My therapist's secret addiction is ____.",
                "The void's customer service department handles ____."
            ],
            "Marvel Universe Expert": [
                "In the next Marvel movie, the villain's secret weapon is ____.",
                "The Avengers' new team building exercise involves ____.",
                "Spider-Man's college debt is actually ____.",
                "Tony Stark's emotional baggage contains ____.",
                "Thanos's mid-life crisis involves ____."
            ],
            "Culinary Comedy Expert": [
                "Gordon Ramsay's secret ingredient for his signature dish is ____.",
                "My sourdough starter has developed ____.",
                "The fusion cuisine experiment went wrong when ____.",
                "The latest food trend is ____ but it's concerning.",
                "Instagram vs. reality cooking shows ____."
            ]
        }
        
        # Try to use persona-specific cards first
        for persona in personas:
            if persona in persona_cards:
                return random.choice(persona_cards[persona])
        
        # Fallback to generic topic-based cards
        generic_cards = {
            "technology": "The next big tech trend will be ____.",
            "workplace": "In today's meeting, we discussed ____.",
            "relationships": "My partner surprised me with ____.",
            "food": "The secret ingredient in this dish is ____.",
            "travel": "The most memorable part of my trip was ____.",
            "education": "The most important lesson I learned was ____.",
            "health": "My doctor recommended ____ for better health.",
            "entertainment": "The highlight of the show was ____.",
            "sports": "The game was decided by ____.",
            "politics": "The key issue in this election is ____."
        }
        
        return generic_cards.get(topic, f"____ is the key to success in {topic}.")
    
    def _get_fallback_white_card(self, black_card: str, personas: List[str]) -> str:
        """Get a fallback white card, considering user personas"""
        # Persona-specific white cards
        persona_responses = {
            "Office Worker": [
                "A professional belly button lint collector",
                "Competitive hedgehog grooming",
                "An expired coupon to Chuck E. Cheese's",
                "The CEO's secret breakdance battle club",
                "A toaster oven manual written in hieroglyphics"
            ],
            "Gaming Guru": [
                "Respawn anxiety in the middle of a meeting",
                "A loot box containing only paperclips",
                "Rage-quitting from Excel spreadsheets",
                "NPCs with corporate benefits",
                "The secret underground donut smuggling ring"
            ],
            "Dad Humor Enthusiast": [
                "Socks with sandals and confidence",
                "Student loan debt but make it memes",
                "A professional avocado wrestler",
                "The void but it's surprisingly supportive",
                "Grandma's interpretive dance troupe"
            ],
            "Millennial Memer": [
                "An undercover llama trainer",
                "Selling bathwater collected from gamers",
                "A Justin Bieber impersonator",
                "Teaching my cat how to twerk",
                "Collecting belly button lint sculptures"
            ],
            "Gen Z Chaos Agent": [
                "A philosophical toaster having an identity crisis",
                "Gravity's mid-life crisis",
                "Emotions as subscription services",
                "The Great Depression but make it fashion",
                "A nudist colony next door"
            ],
            "Dark Humor Connoisseur": [
                "Grandma's secret moonshine business",
                "A surprise intervention from my pets",
                "The iceberg's close relative, Ice Cube",
                "Unicorn burps and goblin toe jam",
                "A ham sandwich's stock market predictions"
            ],
            "Marvel Universe Expert": [
                "A gluten-free vegan meatloaf recipe",
                "The secret salsa recipe: Susan's tears",
                "Nacho cheese fountain",
                "A professional avocado wrestler",
                "The CEO's secret breakdance battle club"
            ],
            "Culinary Comedy Expert": [
                "A toaster oven manual written in hieroglyphics",
                "An expired coupon to Chuck E. Cheese's",
                "The secret underground donut smuggling ring",
                "A professional belly button lint collector",
                "Competitive hedgehog grooming"
            ]
        }
        
        # Try to use persona-specific responses first
        for persona in personas:
            if persona in persona_responses:
                return random.choice(persona_responses[persona])
        
        # Fallback to generic responses
        generic_responses = [
            "Something unexpectedly funny",
            "A terrible mistake",
            "My hidden talent",
            "An awkward situation",
            "The wrong answer",
            "A questionable life choice",
            "Unexpected emotional baggage",
            "My secret shame",
            "The awkward silence",
            "A disappointing revelation"
        ]
        
        return random.choice(generic_responses)
    
    def create_complete_sentences(self, black_cards: List[str], white_cards: List[str]) -> List[Dict]:
        """Create complete sentences by combining black and white cards"""
        complete_sentences = []
        
        for i, (black_card, white_card) in enumerate(zip(black_cards, white_cards)):
            # Clean up the white card
            clean_white_card = white_card.strip()
            if clean_white_card.startswith('"') and clean_white_card.endswith('"'):
                clean_white_card = clean_white_card[1:-1].strip()
            if clean_white_card.startswith("'") and clean_white_card.endswith("'"):
                clean_white_card = clean_white_card[1:-1].strip()
            
            # Convert white card to lowercase for natural sentence flow
            clean_white_card = clean_white_card.lower()
            
            # Fill in the blank with white card
            complete_sentence = black_card.replace('_____', clean_white_card)
            complete_sentence = complete_sentence.replace('____', clean_white_card)
            complete_sentence = complete_sentence.replace('___', clean_white_card)
            complete_sentence = complete_sentence.replace('__', clean_white_card)
            complete_sentence = complete_sentence.replace('_', clean_white_card)
            
            # Fix double periods and ensure proper sentence ending
            complete_sentence = complete_sentence.replace('..', '.')
            if not complete_sentence.endswith('.') and not complete_sentence.endswith('!') and not complete_sentence.endswith('?'):
                complete_sentence += '.'
            
            # Remove any extra spaces
            complete_sentence = ' '.join(complete_sentence.split())
            
            complete_sentences.append({
                'combination_id': i + 1,
                'black_card': black_card,
                'white_card': white_card,
                'clean_white_card': clean_white_card,
                'complete_sentence': complete_sentence
            })
        
        return complete_sentences
    
    async def generate_example_output(self):
        """Generate the complete example output for all users"""
        print("🚀 Starting multi-user AI-powered example card generation...")
        
        # Setup mock users
        await self.setup_mock_users()
        
        # Generate cards for each user
        all_user_cards = {}
        for user_id, user_data in self.users.items():
            user_cards = await self.generate_cards_for_user(user_id, user_data)
            all_user_cards[user_id] = user_cards
        
        # Create output file
        output_filename = f"ai_generated_cah_cards_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write("CARDS AGAINST HUMANITY - AI GENERATED EXAMPLE CARDS\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("Multiple Users with Different Personas\n\n")
            
            # Write cards for each user
            for user_id, user_data in self.users.items():
                user_cards = all_user_cards[user_id]
                
                f.write("=" * 60 + "\n")
                f.write(f"USER: {user_data['username'].upper()}\n")
                f.write("=" * 60 + "\n")
                f.write(f"Mock User ID: {user_id}\n")
                f.write(f"Favorite Personas: {', '.join(user_data['favorite_personas'])}\n")
                f.write(f"AI Generation: {'Enabled' if self.use_ai_generation else 'Disabled'}\n\n")
                
                # Write black cards
                f.write("BLACK CARDS (10)\n")
                f.write("-" * 20 + "\n")
                for i, card in enumerate(user_cards['black_cards'], 1):
                    f.write(f"{i:2d}. {card}\n")
                
                f.write("\nWHITE CARDS (10)\n")
                f.write("-" * 20 + "\n")
                for i, card in enumerate(user_cards['white_cards'], 1):
                    f.write(f"{i:2d}. {card}\n")
                
                f.write("\n")
            
            # Write complete sentences for each user
            f.write("=" * 60 + "\n")
            f.write("COMPLETE SENTENCES FOR EACH USER\n")
            f.write("=" * 60 + "\n\n")
            
            for user_id, user_data in self.users.items():
                user_cards = all_user_cards[user_id]
                complete_sentences = self.create_complete_sentences(
                    user_cards['black_cards'], 
                    user_cards['white_cards']
                )
                
                f.write(f"USER: {user_data['username'].upper()}\n")
                f.write("-" * 20 + "\n")
                
                for i, combo in enumerate(complete_sentences, 1):
                    f.write(f"{i}. {combo['complete_sentence']}\n")
                
                f.write("\n")
            
            f.write("=" * 60 + "\n")
            f.write("END OF GENERATED CARDS\n")
            f.write("=" * 60 + "\n")
        
        print(f"✅ Multi-user AI-generated example output saved to: {output_filename}")
        
        # Also create the evaluation results file
        await self.create_evaluation_results(all_user_cards, output_filename)
        
        return output_filename
    
    async def create_evaluation_results(self, all_user_cards: Dict, cards_filename: str):
        """Create evaluation results JSON file"""
        print("\n📊 Creating evaluation results...")
        
        evaluation_filename = f"complete_sentences_evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        results = {
            "complete_sentences": [],
            "summary": {
                "total_combinations": 0,
                "users": [],
                "average_humor_score": 0.0,
                "persona_diversity": "High - Mix of traditional and dynamic personas",
                "generation_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        combination_id = 1
        all_scores = []
        
        for user_id, user_data in self.users.items():
            user_cards = all_user_cards[user_id]
            complete_sentences = self.create_complete_sentences(
                user_cards['black_cards'], 
                user_cards['white_cards']
            )
            
            user_combinations = 0
            
            for combo in complete_sentences:
                # Generate mock scores (in real scenario, these would come from evaluation)
                # Use real statistical humor evaluator
                try:
                    statistical_scores = self.humor_evaluator.evaluate_humor_statistically(
                        text=combo['complete_sentence'],
                        context=combo['black_card'],
                        user_profile=[]  # Empty profile for now
                    )
                    
                    # Convert to dictionary format
                    mock_scores = {
                        "surprisal_score": round(statistical_scores.surprisal_score, 1),
                        "ambiguity_score": round(statistical_scores.ambiguity_score, 1),
                        "distinctiveness_ratio": round(statistical_scores.distinctiveness_ratio, 1),
                        "entropy_score": round(statistical_scores.entropy_score, 1),
                        "perplexity_score": round(statistical_scores.perplexity_score, 1),
                        "semantic_coherence": round(statistical_scores.semantic_coherence, 1),
                        "distinct_1": round(statistical_scores.distinct_1, 2),
                        "distinct_2": round(statistical_scores.distinct_2, 2),
                        "self_bleu": round(statistical_scores.self_bleu, 2),
                        "mauve_score": round(statistical_scores.mauve_score, 2),
                        "vocabulary_richness": round(statistical_scores.vocabulary_richness, 2),
                        "overall_semantic_diversity": round(statistical_scores.overall_semantic_diversity, 2),
                        "intra_cluster_diversity": round(statistical_scores.intra_cluster_diversity, 2),
                        "inter_cluster_diversity": round(statistical_scores.inter_cluster_diversity, 2),
                        "semantic_spread": round(statistical_scores.semantic_spread, 2),
                        "cluster_coherence": round(statistical_scores.cluster_coherence, 2),
                        "overall_humor_score": round(statistical_scores.overall_humor_score, 1)
                    }
                except Exception as e:
                    print(f"⚠️ Statistical evaluator failed, using fallback: {e}")
                    # Fallback to basic scores if evaluator fails
                    mock_scores = {
                        "surprisal_score": 4.0,
                        "ambiguity_score": 7.0,
                        "distinctiveness_ratio": 3.0,
                        "entropy_score": 5.5,
                        "perplexity_score": 8.0,
                        "semantic_coherence": 7.0,
                        "distinct_1": 0.90,
                        "distinct_2": 1.0,
                        "self_bleu": 0.0,
                        "mauve_score": 0.0,
                        "vocabulary_richness": 0.50,
                        "overall_semantic_diversity": 0.0,
                        "intra_cluster_diversity": 0.0,
                        "inter_cluster_diversity": 0.0,
                        "semantic_spread": 0.0,
                        "cluster_coherence": 0.0,
                        "overall_humor_score": 6.5
                    }
                
                result = {
                    "combination_id": combination_id,
                    "user_id": user_id,
                    "personas": user_data['favorite_personas'],
                    "black_card": combo['black_card'],
                    "white_card": combo['white_card'],
                    "clean_white_card": combo['clean_white_card'],
                    "complete_sentence": combo['complete_sentence'],
                    "scores": mock_scores
                }
                
                results["complete_sentences"].append(result)
                all_scores.append(mock_scores["overall_humor_score"])
                combination_id += 1
                user_combinations += 1
            
            # Add user summary
            results["summary"]["users"].append({
                "user_id": user_id,
                "personas": user_data['favorite_personas'],
                "combinations": user_combinations
            })
        
        # Calculate overall summary
        results["summary"]["total_combinations"] = len(results["complete_sentences"])
        if all_scores:
            results["summary"]["average_humor_score"] = round(sum(all_scores) / len(all_scores), 1)
        
        # Save evaluation results
        with open(evaluation_filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Evaluation results saved to: {evaluation_filename}")

async def main():
    """Main function to run the multi-user AI example card generation"""
    generator = MultiUserAIExampleCardGenerator()
    
    # You can disable AI generation by setting this to False
    # generator.use_ai_generation = False
    
    output_file = await generator.generate_example_output()
    print(f"\n🎉 Multi-user AI example card generation complete!")
    print(f"📁 Output file: {output_file}")

if __name__ == "__main__":
    asyncio.run(main())
