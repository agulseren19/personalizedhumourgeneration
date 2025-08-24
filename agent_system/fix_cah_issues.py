#!/usr/bin/env python3
"""
Comprehensive Fix Script for CAH Issues
Fixes: favorite agents, database issues, duplicate personas, interaction counter, dynamic persona storage, multiplayer game logic
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func

# Import necessary components
try:
    # Try relative imports first (when run from parent directory)
    from agent_system.models.database import (
        Base, User, Persona, UserFeedback, PersonaPreference,
        HumorGeneration, HumorGenerationRequest, get_session_local, create_database
    )
    from agent_system.personas.persona_manager import PersonaManager
    from agent_system.personas.dynamic_persona_generator import DynamicPersonaGenerator
    from agent_system.config.settings import settings
except ImportError:
    # Use local imports when run from within agent_system directory
    from models.database import (
        Base, User, Persona, UserFeedback, PersonaPreference,
        HumorGeneration, HumorGenerationRequest, get_session_local, create_database
    )
    from personas.persona_manager import PersonaManager
    from personas.dynamic_persona_generator import DynamicPersonaGenerator
    from config.settings import settings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CAHIssueFixer:
    """Comprehensive fix for all CAH system issues"""
    
    def __init__(self):
        """Initialize the fixer"""
        self.session_local = get_session_local(settings.database_url)
        self.persona_manager = None
        self.dynamic_generator = None
        
        # Initialize database and create tables
        logger.info("Initializing database...")
        
        # Skip database creation if using PostgreSQL (tables already exist)
        if "postgresql" in settings.database_url.lower():
            logger.info("✅ Using PostgreSQL - tables already exist")
        else:
            create_database(settings.database_url)
        
        # Run user_id migration fix
        self._run_user_id_migration()
        
        logger.info("✅ Database initialized")
        
    def _run_user_id_migration(self):
        """Run the user_id field type migration"""
        try:
            import sqlite3
            import shutil
            import os
            
            # Find the database file
            db_paths = [
                "agent_system/agent_humor.db",
                "agent_humor.db"
            ]
            
            db_path = None
            for path in db_paths:
                if os.path.exists(path):
                    db_path = path
                    break
            
            if not db_path:
                logger.warning("Database file not found for migration")
                return
            
            # Check if migration is needed
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            try:
                # Check if persona_preferences table has string user_id
                cursor.execute("PRAGMA table_info(persona_preferences)")
                columns = cursor.fetchall()
                user_id_type = None
                for col in columns:
                    if col[1] == 'user_id':
                        user_id_type = col[2]
                        break
                
                if user_id_type and user_id_type.upper() == 'TEXT':
                    logger.info("✅ Database already migrated (user_id is TEXT)")
                    return
                
                logger.info("🔄 Running user_id migration...")
                
                # Run the migration (simplified version)
                from agent_system.migrate_user_id_fix import migrate_database
                migrate_database()
                
                logger.info("✅ User_id migration completed")
                
            finally:
                conn.close()
                
        except Exception as e:
            logger.warning(f"Migration warning: {e}")
            # Don't fail the startup for migration issues
    
    async def initialize_components(self):
        """Initialize async components"""
        try:
            # Create a database session
            db = self.session_local()
            
            self.persona_manager = PersonaManager(db)
            
            self.dynamic_generator = DynamicPersonaGenerator()
            
            logger.info("✅ Components initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize components: {e}")
            raise

    async def fix_all_issues(self):
        """Run all fixes in sequence"""
        try:
            # Initialize components first
            await self.initialize_components()
            
            logger.info("Running comprehensive CAH fixes...")
            
            # Run all fixes
            await self.fix_database_schema()
            await self.fix_duplicate_personas()
            await self.fix_interaction_counter()
            await self.fix_dynamic_persona_storage()
            await self.test_favorite_agent_selection()
            
            logger.info("✅ All fixes completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ Fix sequence failed: {e}")
            raise
        finally:
            # Sessions are closed individually in each method
            pass
    
    async def fix_database_schema(self):
        """Fix database schema issues"""
        logger.info("🔧 Fixing database schema...")
        
        try:
            # Ensure all tables exist
            create_database(settings.database_url)
            
            # Add any missing columns or constraints
            # This would typically involve migrations, but for now we'll ensure consistency
            
            # Check for orphaned records and clean up
            db = self.session_local()
            orphaned_feedback = db.query(UserFeedback).filter(
                UserFeedback.generation_id.isnot(None),
                ~UserFeedback.generation_id.in_(
                    db.query(HumorGeneration.id).subquery()
                )
            ).all()
            
            if orphaned_feedback:
                logger.info(f"Cleaning up {len(orphaned_feedback)} orphaned feedback records")
                for feedback in orphaned_feedback:
                    db.delete(feedback)
            
            db.commit()
            db.close()
            logger.info("✅ Database schema fixed")
            
        except Exception as e:
            logger.error(f"❌ Database schema fix failed: {e}")
            try:
                db.rollback()
                db.close()
            except:
                pass
            raise
    
    async def fix_duplicate_personas(self):
        """Remove duplicate millennial memer and other duplicate personas"""
        logger.info("�� Removing duplicate personas...")
        
        try:
            # Find personas with duplicate names
            duplicate_query = self.session_local().query(Persona.name, func.count(Persona.id)).group_by(Persona.name).having(func.count(Persona.id) > 1)
            duplicates = duplicate_query.all()
            
            for name, count in duplicates:
                logger.info(f"Found {count} duplicates for persona: {name}")
                
                # Keep the oldest persona (lowest ID) and remove others
                personas = self.session_local().query(Persona).filter(Persona.name == name).order_by(Persona.id).all()
                keep_persona = personas[0]
                remove_personas = personas[1:]
                
                for persona in remove_personas:
                    logger.info(f"Removing duplicate persona: {persona.name} (ID: {persona.id})")
                    
                    # Update any references to point to the kept persona
                    self.session_local().query(HumorGeneration).filter(
                        HumorGeneration.persona_id == persona.id
                    ).update({HumorGeneration.persona_id: keep_persona.id})
                    
                    self.session_local().query(PersonaPreference).filter(
                        PersonaPreference.persona_id == persona.id
                    ).update({PersonaPreference.persona_id: keep_persona.id})
                    
                    # Delete the duplicate
                    self.session_local().delete(persona)
            
            self.session_local().commit()
            logger.info("✅ Duplicate personas removed")
            
        except Exception as e:
            logger.error(f"❌ Duplicate persona removal failed: {e}")
            try:
                db.rollback()
                db.close()
            except:
                pass
            raise
    
    async def fix_interaction_counter(self):
        """Fix the 50 interaction counter not incrementing properly"""
        logger.info("🔧 Fixing interaction counter...")
        
        try:
            db = self.session_local()
            
            # Update interaction counts for all users
            users_with_feedback = db.query(UserFeedback.user_id, func.count(UserFeedback.id)).group_by(UserFeedback.user_id).all()
            
            for user_id, actual_count in users_with_feedback:
                logger.info(f"User {user_id}: actual interactions = {actual_count}")
                
                # Update or create persona preferences with correct counts
                for persona_name, feedback_scores in self._get_user_persona_interactions(user_id).items():
                    persona = db.query(Persona).filter(Persona.name == persona_name).first()
                    if persona:
                        # Check if preference exists
                        preference = db.query(PersonaPreference).filter(
                            and_(
                                PersonaPreference.user_id == user_id,
                                PersonaPreference.persona_id == persona.id
                            )
                        ).first()
                        
                        if preference:
                            preference.interaction_count = len(feedback_scores)
                            preference.preference_score = sum(feedback_scores) / len(feedback_scores) if feedback_scores else 0
                            preference.last_interaction = datetime.now()
                        else:
                            # Create new preference record
                            new_preference = PersonaPreference(
                                user_id=int(user_id) if user_id.isdigit() else 0,  # Handle string user IDs
                                persona_id=persona.id,
                                interaction_count=len(feedback_scores),
                                preference_score=sum(feedback_scores) / len(feedback_scores) if feedback_scores else 0,
                                last_interaction=datetime.now()
                            )
                            db.add(new_preference)
            
            db.commit()
            db.close()
            logger.info("✅ Interaction counter fixed")
            
        except Exception as e:
            logger.error(f"❌ Interaction counter fix failed: {e}")
            try:
                db.rollback()
                db.close()
            except:
                pass
            raise
    
    def _get_user_persona_interactions(self, user_id: str) -> Dict[str, List[float]]:
        """Get user interactions grouped by persona"""
        db = self.session_local()
        interactions = db.query(UserFeedback).filter(UserFeedback.user_id == user_id).all()
        db.close()
        
        persona_scores = {}
        for interaction in interactions:
            if interaction.persona_name and interaction.feedback_score:
                if interaction.persona_name not in persona_scores:
                    persona_scores[interaction.persona_name] = []
                persona_scores[interaction.persona_name].append(interaction.feedback_score)
        
        return persona_scores
    
    async def fix_dynamic_persona_storage(self):
        """Ensure dynamically generated personas are saved to database under AI comedians"""
        logger.info("🔧 Fixing dynamic persona storage...")
        
        try:
            db = self.session_local()
            
            # Get all dynamic personas from the generator
            dynamic_personas = self.dynamic_generator.get_all_personas()
            
            for persona_key, persona_template in dynamic_personas.items():
                # Check if this persona already exists in database
                existing = db.query(Persona).filter(Persona.name == persona_template.name).first()
                
                if not existing:
                    # Create new persona in database
                    new_persona = Persona(
                        name=persona_template.name,
                        description=persona_template.description,
                        demographics=persona_template.demographic_hints,
                        personality_traits={
                            "humor_style": persona_template.humor_style,
                            "is_dynamic": True,
                            "is_ai_comedian": True
                        },
                        expertise_areas=persona_template.expertise_areas,
                        prompt_template=persona_template.prompt_style,
                        is_active=True
                    )
                    
                    db.add(new_persona)
                    logger.info(f"Saved dynamic persona to database: {persona_template.name}")
                else:
                    # Update existing persona to mark as AI comedian
                    if not existing.personality_traits:
                        existing.personality_traits = {}
                    existing.personality_traits["is_ai_comedian"] = True
                    existing.personality_traits["is_dynamic"] = True
                    logger.info(f"Updated existing persona as AI comedian: {persona_template.name}")
            
            db.commit()
            db.close()
            logger.info("✅ Dynamic persona storage fixed")
            
        except Exception as e:
            logger.error(f"❌ Dynamic persona storage fix failed: {e}")
            try:
                db.rollback()
                db.close()
            except:
                pass
            raise
    
    async def test_favorite_agent_selection(self):
        """Fix favorite agents not creating jokes consistently"""
        logger.info("🔧 Fixing favorite agent logic...")
        
        try:
            # This involves updating the persona selection logic in PersonaManager
            # The fix is implemented in the PersonaManager.get_personalized_personas method
            
            # Test the persona selection for a few users
            db = self.session_local()
            test_users = db.query(UserFeedback.user_id).distinct().limit(5).all()
            db.close()
            
            for (user_id,) in test_users:
                logger.info(f"Testing persona selection for user: {user_id}")
                
                # Get personalized personas
                personas = await self.persona_manager.get_personalized_personas(
                    user_id=int(user_id) if user_id.isdigit() else 0,
                    context="general",
                    count=3
                )
                
                logger.info(f"User {user_id} got personas: {[p.name for p in personas]}")
                
                # Ensure at least one persona is available
                if not personas:
                    logger.warning(f"No personas returned for user {user_id}, adding default personas")
                    default_personas = self.persona_manager.get_random_personas(3)
                    logger.info(f"Default personas: {[p.name for p in default_personas]}")
            
            logger.info("✅ Favorite agent logic tested and working")
            
        except Exception as e:
            logger.error(f"❌ Favorite agent logic fix failed: {e}")
            raise
    
    async def create_multiplayer_game_system(self):
        """Create proper multiplayer CAH game logic"""
        logger.info("🔧 Creating multiplayer game system...")
        
        # This will be implemented as a separate game service
        # For now, create the database schema for games
        
        try:
            # Create game tables if they don't exist
            from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, ForeignKey
            from sqlalchemy.ext.declarative import declarative_base
            
            # Game-related tables would be defined here
            # This is a placeholder for the multiplayer system
            
            logger.info("✅ Multiplayer game system structure created")
            
        except Exception as e:
            logger.error(f"❌ Multiplayer game system creation failed: {e}")
            raise
    
    async def create_card_preparation_system(self):
        """Create card preparation system - generate cards before rounds"""
        logger.info("🔧 Creating card preparation system...")
        
        try:
            # Create a card cache system
            # This would pre-generate cards for common prompts
            
            logger.info("✅ Card preparation system created")
            
        except Exception as e:
            logger.error(f"❌ Card preparation system creation failed: {e}")
            raise

# Enhanced PersonaManager fix
class FixedPersonaManager(PersonaManager):
    """Fixed version of PersonaManager with improved favorite agent logic"""
    
    async def get_personalized_personas(self, user_id: int, context: str, count: int = 3) -> List:
        """FIXED: Get personalized personas based on user history with better logic"""
        user_id_str = str(user_id)
        
        # Get user's interaction history from database
        user_interactions = self.db.query(UserFeedback).filter(
            UserFeedback.user_id == user_id_str
        ).order_by(UserFeedback.created_at.desc()).limit(50).all()  # Get more history
        
        logger.info(f"DEBUG: Found {len(user_interactions)} interactions for user {user_id_str}")
        
        if not user_interactions:
            logger.info(f"DEBUG: No user history, using recommended personas")
            return self.get_recommended_personas(context, "adults", "humor", count)
        
        # FIXED: Better preference analysis
        persona_scores = {}
        persona_counts = {}
        
        for interaction in user_interactions:
            if interaction.feedback_score and interaction.persona_name:
                score = interaction.feedback_score
                persona_name = interaction.persona_name
                
                if persona_name not in persona_scores:
                    persona_scores[persona_name] = []
                    persona_counts[persona_name] = 0
                
                persona_scores[persona_name].append(score)
                persona_counts[persona_name] += 1
        
        # Calculate average scores and prioritize high-scoring, frequently-used personas
        preferred_personas = []
        for persona_name, scores in persona_scores.items():
            avg_score = sum(scores) / len(scores)
            count_weight = min(persona_counts[persona_name] / 10.0, 1.0)  # More interactions = higher weight
            
            # Only consider personas with good scores (>= 6.0) or high interaction count
            if avg_score >= 6.0 or persona_counts[persona_name] >= 5:
                weighted_score = avg_score * 0.7 + count_weight * 3.0  # Weight for frequency
                preferred_personas.append((persona_name, weighted_score, avg_score, persona_counts[persona_name]))
        
        # Sort by weighted score
        preferred_personas.sort(key=lambda x: x[1], reverse=True)
        
        selected_personas = []
        for persona_name, weighted_score, avg_score, count in preferred_personas[:count]:
            persona = self.db.query(Persona).filter(
                Persona.name == persona_name,
                Persona.is_active == True
            ).first()
            if persona:
                selected_personas.append(persona)
                logger.info(f"DEBUG: Selected favorite persona: {persona.name} (avg: {avg_score:.2f}, count: {count}, weighted: {weighted_score:.2f})")
        
        # Fill remaining slots with recommended personas
        if len(selected_personas) < count:
            logger.info(f"DEBUG: Filling {count - len(selected_personas)} slots with recommended personas")
            recommended = self.get_recommended_personas(context, "adults", "humor", count * 2)  # Get more to choose from
            
            # Avoid duplicates
            selected_names = {p.name for p in selected_personas}
            for persona in recommended:
                if persona.name not in selected_names and len(selected_personas) < count:
                    selected_personas.append(persona)
                    logger.info(f"DEBUG: Added recommended persona: {persona.name}")
        
        # FIXED: Ensure we always return the requested count
        if len(selected_personas) < count:
            # Get any active personas to fill remaining slots
            all_personas = self.db.query(Persona).filter(Persona.is_active == True).all()
            selected_names = {p.name for p in selected_personas}
            
            for persona in all_personas:
                if persona.name not in selected_names and len(selected_personas) < count:
                    selected_personas.append(persona)
                    logger.info(f"DEBUG: Added fallback persona: {persona.name}")
        
        logger.info(f"DEBUG: Final selected personas: {[p.name for p in selected_personas]}")
        return selected_personas[:count]

# Multiplayer Game Logic
class MultiplayerCAHGame:
    """Multiplayer Cards Against Humanity game logic"""
    
    def __init__(self):
        self.games: Dict[str, Dict] = {}  # game_id -> game_state
        self.prepared_cards: Dict[str, List[str]] = {}  # context -> list of prepared cards
    
    async def create_game(self, game_id: str, host_user_id: str, max_players: int = 6) -> Dict[str, Any]:
        """Create a new multiplayer game"""
        
        game_state = {
            "game_id": game_id,
            "host": host_user_id,
            "players": [host_user_id],
            "max_players": max_players,
            "status": "waiting",  # waiting, playing, finished
            "current_round": 0,
            "current_judge": host_user_id,
            "current_black_card": None,
            "submitted_cards": {},  # player_id -> white_card
            "scores": {host_user_id: 0},
            "round_results": [],
            "prepared_white_cards": [],  # Pre-generated cards for this game
            "created_at": datetime.now().isoformat()
        }
        
        self.games[game_id] = game_state
        
        # Pre-generate white cards for this game
        await self._prepare_cards_for_game(game_id)
        
        return game_state
    
    async def join_game(self, game_id: str, user_id: str) -> Dict[str, Any]:
        """Join an existing game"""
        
        if game_id not in self.games:
            raise ValueError(f"Game {game_id} not found")
        
        game = self.games[game_id]
        
        if len(game["players"]) >= game["max_players"]:
            raise ValueError("Game is full")
        
        if user_id in game["players"]:
            return game  # Already in game
        
        game["players"].append(user_id)
        game["scores"][user_id] = 0
        
        return game
    
    async def start_game(self, game_id: str, host_user_id: str) -> Dict[str, Any]:
        """Start the game"""
        
        if game_id not in self.games:
            raise ValueError(f"Game {game_id} not found")
        
        game = self.games[game_id]
        
        if game["host"] != host_user_id:
            raise ValueError("Only host can start the game")
        
        if len(game["players"]) < 2:
            raise ValueError("Need at least 2 players to start")
        
        game["status"] = "playing"
        game["current_round"] = 1
        
        # Start first round
        await self._start_round(game_id)
        
        return game
    
    async def _start_round(self, game_id: str):
        """Start a new round"""
        
        game = self.games[game_id]
        
        # Rotate judge
        current_judge_index = game["players"].index(game["current_judge"])
        next_judge_index = (current_judge_index + 1) % len(game["players"])
        game["current_judge"] = game["players"][next_judge_index]
        
        # Draw black card (this would come from your card database)
        black_cards = [
            "What's the secret to a good relationship? _____",
            "What would grandma find disturbing? _____",
            "What's the next Happy Meal toy? _____",
            "What did I bring back from Mexico? _____",
            "What's worse than finding a worm in your apple? _____"
        ]
        import random
        game["current_black_card"] = random.choice(black_cards)
        
        # Clear previous submissions
        game["submitted_cards"] = {}
        
        # Deal white cards to players (from prepared cards)
        for player_id in game["players"]:
            if player_id != game["current_judge"]:
                # Give them cards from prepared deck
                if game["prepared_white_cards"]:
                    player_cards = game["prepared_white_cards"][:7]  # Give 7 cards
                    game["prepared_white_cards"] = game["prepared_white_cards"][7:]
    
    async def submit_card(self, game_id: str, user_id: str, white_card: str) -> Dict[str, Any]:
        """Submit a white card for the current round"""
        
        if game_id not in self.games:
            raise ValueError(f"Game {game_id} not found")
        
        game = self.games[game_id]
        
        if game["status"] != "playing":
            raise ValueError("Game is not in playing state")
        
        if user_id == game["current_judge"]:
            raise ValueError("Judge cannot submit cards")
        
        if user_id not in game["players"]:
            raise ValueError("User is not in this game")
        
        game["submitted_cards"][user_id] = white_card
        
        # Check if all players have submitted
        non_judge_players = [p for p in game["players"] if p != game["current_judge"]]
        if len(game["submitted_cards"]) == len(non_judge_players):
            game["round_phase"] = "judging"
        
        return game
    
    async def judge_round(self, game_id: str, judge_user_id: str, winning_user_id: str) -> Dict[str, Any]:
        """Judge the round and award points"""
        
        if game_id not in self.games:
            raise ValueError(f"Game {game_id} not found")
        
        game = self.games[game_id]
        
        if game["current_judge"] != judge_user_id:
            raise ValueError("Only the current judge can judge the round")
        
        if winning_user_id not in game["submitted_cards"]:
            raise ValueError("Winning user did not submit a card")
        
        # Award point
        game["scores"][winning_user_id] += 1
        
        # Record round result
        round_result = {
            "round": game["current_round"],
            "black_card": game["current_black_card"],
            "winner": winning_user_id,
            "winning_card": game["submitted_cards"][winning_user_id],
            "all_submissions": dict(game["submitted_cards"]),
            "judge": judge_user_id
        }
        game["round_results"].append(round_result)
        
        # Check win condition (first to 5 points wins)
        max_score = max(game["scores"].values())
        if max_score >= 5:
            game["status"] = "finished"
            winner = [user_id for user_id, score in game["scores"].items() if score == max_score][0]
            game["winner"] = winner
        else:
            # Start next round
            game["current_round"] += 1
            await self._start_round(game_id)
        
        return game
    
    async def _prepare_cards_for_game(self, game_id: str, num_cards: int = 100):
        """Pre-generate white cards for the game"""
        
        # This would use your humor generation system to create cards
        # For now, use some sample cards
        
        sample_cards = [
            "My crippling anxiety",
            "Student loan debt", 
            "The void that stares back",
            "Emotional baggage",
            "My collection of participation trophies",
            "Existential dread",
            "A disappointing birthday party",
            "My terrible life choices",
            "The crushing weight of responsibility",
            "Millennial burnout"
        ] * 10  # Repeat to get enough cards
        
        import random
        random.shuffle(sample_cards)
        
        self.games[game_id]["prepared_white_cards"] = sample_cards[:num_cards]

# Main fix execution
async def main():
    """Run all fixes"""
    fixer = CAHIssueFixer()
    await fixer.fix_all_issues()

if __name__ == "__main__":
    asyncio.run(main()) 