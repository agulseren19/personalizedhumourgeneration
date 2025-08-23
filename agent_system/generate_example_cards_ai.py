#!/usr/bin/env python3
"""
Generate Example Cards Against Humanity Cards - AI Version
Creates a mock user with favorite personas and generates 30 black cards and 90 white cards
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

class AIExampleCardGenerator:
    """Generates example CAH cards using AI when possible"""
    
    def __init__(self):
        self.mock_user_id = "mock_user_123"
        self.favorite_personas = [
            "Dad Humor Enthusiast",
            "Millennial Memer", 
            "Gen Z Chaos Agent",
            "Marvel Universe Expert",
            "Corporate Humor Specialist"
        ]
        self.humor_orchestrator = ImprovedHumorOrchestrator()
        self.use_ai_generation = True  # Set to False to use only fallback cards
        
    async def setup_mock_user(self):
        """Create a mock user with favorite personas"""
        print("🔧 Setting up mock user...")
        
        try:
            # Create database
            create_database(settings.database_url)
            SessionLocal = get_session_local(settings.database_url)
            db = SessionLocal()
            
            # Initialize personas
            persona_manager = PersonaManager(db)
            
            # Create mock user
            mock_user = User(
                id=self.mock_user_id,
                username="example_user",
                email="example@cah.com",
                password_hash="mock_hash",
                age_range="25-35",
                occupation="Software Developer",
                education_level="Bachelor's",
                interests=["technology", "gaming", "comedy", "movies"],
                humor_preferences={
                    "preferred_style": "edgy but clever",
                    "favorite_topics": ["tech", "pop culture", "workplace"],
                    "audience": "adults"
                }
            )
            
            # Check if user already exists
            existing_user = db.query(User).filter(User.id == self.mock_user_id).first()
            if not existing_user:
                db.add(mock_user)
                db.commit()
                print(f"✅ Created mock user: {mock_user.username}")
            else:
                print(f"✅ Mock user already exists: {existing_user.username}")
            
            # Create persona preferences for favorite personas
            for persona_name in self.favorite_personas:
                persona = db.query(Persona).filter(Persona.name == persona_name).first()
                if persona:
                    # Check if preference already exists
                    existing_pref = db.query(PersonaPreference).filter(
                        PersonaPreference.user_id == self.mock_user_id,
                        PersonaPreference.persona_id == persona.id
                    ).first()
                    
                    if not existing_pref:
                        # Create high preference score for favorite personas
                        preference = PersonaPreference(
                            user_id=self.mock_user_id,
                            persona_id=persona.id,
                            preference_score=9.0,  # High score for favorites
                            interaction_count=1,
                            context_preferences={"general": 9.0}
                        )
                        db.add(preference)
                        print(f"✅ Added preference for {persona_name}")
            
            db.commit()
            print(f"✅ Mock user setup complete with {len(self.favorite_personas)} favorite personas")
            
        except Exception as e:
            print(f"❌ Error setting up mock user: {e}")
            if 'db' in locals():
                db.rollback()
        finally:
            if 'db' in locals():
                db.close()
    
    async def generate_ai_black_card(self, topic: str) -> str:
        """Try to generate a black card using AI"""
        if not self.use_ai_generation:
            return None
            
        try:
            request = HumorRequest(
                context=f"Generate a black card about {topic}",
                audience="adults",
                topic=topic,
                user_id=self.mock_user_id,
                card_type="black"
            )
            
            result = await self.humor_orchestrator.generate_and_evaluate_humor(request)
            
            if result and result.get('success') and result.get('results'):
                # Get the best black card
                best_result = result['results'][0]
                
                # The structure is: result['results'][0]['generation'].text
                if isinstance(best_result, dict) and 'generation' in best_result:
                    generation_obj = best_result['generation']
                    if hasattr(generation_obj, 'text'):
                        black_card_text = generation_obj.text
                    else:
                        black_card_text = str(generation_obj)
                else:
                    black_card_text = str(best_result)
                
                # Clean and format the black card
                black_card_text = self._clean_black_card(black_card_text)
                
                if black_card_text:
                    return black_card_text
                    
        except Exception as e:
            print(f"    ⚠️  AI generation failed for topic '{topic}': {e}")
        
        return None
    
    async def generate_ai_white_card(self, black_card: str) -> str:
        """Try to generate a white card using AI"""
        if not self.use_ai_generation:
            return None
            
        try:
            request = HumorRequest(
                context=black_card,
                audience="adults",
                topic="general",
                user_id=self.mock_user_id,
                card_type="white"
            )
            
            result = await self.humor_orchestrator.generate_and_evaluate_humor(request)
            
            if result and result.get('success') and result.get('results'):
                # Get a white card result
                white_card_result = result['results'][0]
                
                # The structure is: result['results'][0]['generation'].text
                if isinstance(white_card_result, dict) and 'generation' in white_card_result:
                    generation_obj = white_card_result['generation']
                    if hasattr(generation_obj, 'text'):
                        white_card_text = generation_obj.text
                    else:
                        white_card_text = str(generation_obj)
                else:
                    white_card_text = str(white_card_result)
                
                # Clean and format the white card
                white_card_text = self._clean_white_card(white_card_text)
                
                if white_card_text:
                    return white_card_text
                    
        except Exception as e:
            print(f"    ⚠️  AI generation failed for white card: {e}")
        
        return None
    
    async def generate_black_cards(self) -> List[str]:
        """Generate 30 black cards using AI when possible"""
        print("🎭 Generating 30 black cards...")
        
        black_cards = []
        topics = [
            "technology", "workplace", "relationships", "food", "travel",
            "education", "health", "entertainment", "sports", "politics",
            "social media", "dating", "family", "money", "fashion",
            "music", "movies", "books", "gaming", "fitness",
            "cooking", "shopping", "driving", "parties", "holidays",
            "weather", "animals", "science", "history", "art"
        ]
        
        for i in range(30):
            topic = topics[i % len(topics)]
            print(f"  Generating black card {i+1}/30 for topic: {topic}")
            
            # Try AI generation first
            ai_card = await self.generate_ai_black_card(topic)
            
            if ai_card:
                black_cards.append(ai_card)
                print(f"    ✅ AI Generated: {ai_card[:60]}...")
            else:
                # Use fallback card
                fallback = self._get_fallback_black_card(topic)
                black_cards.append(fallback)
                print(f"    ⚠️  Using fallback: {fallback}")
            
            # Small delay to avoid rate limiting
            await asyncio.sleep(0.5)
        
        print(f"✅ Generated {len(black_cards)} black cards")
        return black_cards
    
    async def generate_white_cards(self, black_cards: List[str]) -> List[str]:
        """Generate 90 white cards using AI when possible"""
        print("🎭 Generating 90 white cards...")
        
        white_cards = []
        
        for i, black_card in enumerate(black_cards):
            print(f"  Generating white cards for black card {i+1}/30: {black_card[:50]}...")
            
            for j in range(3):  # 3 white cards per black card
                # Try AI generation first
                ai_card = await self.generate_ai_white_card(black_card)
                
                if ai_card:
                    white_cards.append(ai_card)
                    print(f"    ✅ AI Generated white card {j+1}/3: {ai_card[:50]}...")
                else:
                    # Use fallback white card
                    fallback = self._get_fallback_white_card(black_card)
                    white_cards.append(fallback)
                    print(f"    ⚠️  Using fallback white card {j+1}/3: {fallback}")
                
                # Small delay to avoid rate limiting
                await asyncio.sleep(0.3)
        
        print(f"✅ Generated {len(white_cards)} white cards")
        return white_cards
    
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
            # Add blank if missing
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
    
    def _get_fallback_black_card(self, topic: str) -> str:
        """Get a fallback black card for a topic"""
        fallback_cards = {
            "technology": "The next big tech trend will be ____.",
            "workplace": "In today's meeting, we discussed ____.",
            "relationships": "My partner surprised me with ____.",
            "food": "The secret ingredient in this dish is ____.",
            "travel": "The most memorable part of my trip was ____.",
            "education": "The most important lesson I learned was ____.",
            "health": "My doctor recommended ____ for better health.",
            "entertainment": "The highlight of the show was ____.",
            "sports": "The game was decided by ____.",
            "politics": "The key issue in this election is ____.",
            "social media": "My latest post went viral because of ____.",
            "dating": "The most awkward moment on my date was ____.",
            "family": "Family gatherings always involve ____.",
            "money": "The best investment I ever made was ____.",
            "fashion": "This season's must-have accessory is ____.",
            "music": "The song that changed my life was ____.",
            "movies": "The plot twist I never saw coming was ____.",
            "books": "The character I most relate to is ____.",
            "gaming": "The hardest level in this game requires ____.",
            "fitness": "My workout routine includes ____.",
            "cooking": "The recipe calls for ____.",
            "shopping": "I impulse-bought ____.",
            "driving": "The worst driving habit is ____.",
            "parties": "Party games always end with ____.",
            "holidays": "The best holiday tradition is ____.",
            "weather": "The weather forecast predicts ____.",
            "animals": "My pet's favorite activity is ____.",
            "science": "The breakthrough discovery was ____.",
            "history": "The most important historical event was ____.",
            "art": "The masterpiece was inspired by ____."
        }
        
        return fallback_cards.get(topic, f"____ is the key to success in {topic}.")
    
    def _get_fallback_white_card(self, black_card: str) -> str:
        """Get a fallback white card for a black card"""
        fallback_responses = [
            "Something unexpectedly funny",
            "A terrible mistake",
            "My hidden talent",
            "An awkward situation",
            "The wrong answer",
            "A questionable life choice",
            "Unexpected emotional baggage",
            "My secret shame",
            "The awkward silence",
            "A disappointing revelation",
            "My questionable decisions",
            "Something I shouldn't have said",
            "My greatest weakness",
            "An embarrassing moment",
            "The thing I regret most",
            "My biggest fear",
            "A terrible idea",
            "My worst habit",
            "Something I can't explain",
            "My biggest mistake",
            "An unfortunate accident",
            "My secret obsession",
            "The thing I'm most ashamed of",
            "My biggest flaw",
            "Something I shouldn't have done",
            "My worst nightmare",
            "An embarrassing story",
            "My biggest regret",
            "Something I can't unsee",
            "My greatest failure"
        ]
        
        return random.choice(fallback_responses)
    
    async def generate_example_output(self):
        """Generate the complete example output"""
        print("🚀 Starting AI-powered example card generation...")
        
        # Setup mock user
        await self.setup_mock_user()
        
        # Generate black cards
        black_cards = await self.generate_black_cards()
        
        # Generate white cards
        white_cards = await self.generate_white_cards(black_cards)
        
        # Create output file
        output_filename = f"ai_generated_cah_cards_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write("CARDS AGAINST HUMANITY - AI GENERATED EXAMPLE CARDS\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Mock User ID: {self.mock_user_id}\n")
            f.write(f"Favorite Personas: {', '.join(self.favorite_personas)}\n")
            f.write(f"AI Generation: {'Enabled' if self.use_ai_generation else 'Disabled'}\n\n")
            
            f.write("BLACK CARDS (30)\n")
            f.write("-" * 20 + "\n")
            for i, card in enumerate(black_cards, 1):
                f.write(f"{i:2d}. {card}\n")
            
            f.write("\n" + "=" * 60 + "\n\n")
            
            f.write("WHITE CARDS (90)\n")
            f.write("-" * 20 + "\n")
            for i, card in enumerate(white_cards, 1):
                f.write(f"{i:2d}. {card}\n")
            
            f.write("\n" + "=" * 60 + "\n\n")
            
            f.write("CARD COMBINATIONS\n")
            f.write("-" * 20 + "\n")
            f.write("Each black card should be paired with 3 white cards:\n\n")
            
            for i, black_card in enumerate(black_cards):
                f.write(f"Black Card {i+1}: {black_card}\n")
                start_idx = i * 3
                for j in range(3):
                    white_idx = start_idx + j
                    if white_idx < len(white_cards):
                        f.write(f"  White Card {white_idx+1}: {white_cards[white_idx]}\n")
                f.write("\n")
        
        print(f"✅ AI-generated example output saved to: {output_filename}")
        print(f"📊 Generated {len(black_cards)} black cards and {len(white_cards)} white cards")
        
        return output_filename

async def main():
    """Main function to run the AI example card generation"""
    generator = AIExampleCardGenerator()
    
    # You can disable AI generation by setting this to False
    # generator.use_ai_generation = False
    
    output_file = await generator.generate_example_output()
    print(f"\n🎉 AI example card generation complete!")
    print(f"📁 Output file: {output_file}")

if __name__ == "__main__":
    asyncio.run(main())
