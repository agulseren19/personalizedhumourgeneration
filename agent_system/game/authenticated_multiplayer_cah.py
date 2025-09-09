#!/usr/bin/env python3
"""
Authenticated Multiplayer Cards Against Humanity Game System
Updated to use authenticated users and store learning values per user
"""

import asyncio
import json
import random
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
from sqlalchemy.orm import Session
import inspect

# Import database models and authentication
try:
    from ..models.database import User, UserFeedback, PersonaPreference, Persona
    from ..agents.improved_humor_agents import ImprovedHumorOrchestrator, HumorRequest
    from ..personas.persona_manager import PersonaManager
except ImportError:
    # Fallback to absolute imports when running directly
    from models.database import User, UserFeedback, PersonaPreference, Persona
    from agents.improved_humor_agents import ImprovedHumorOrchestrator, HumorRequest
    from personas.persona_manager import PersonaManager

logger = logging.getLogger(__name__)

class GameStatus(Enum):
    WAITING = "waiting"
    STARTING = "starting" 
    IN_PROGRESS = "in_progress"
    JUDGING = "judging"
    ROUND_COMPLETE = "round_complete"
    FINISHED = "finished"

class RoundPhase(Enum):
    WAITING = "waiting"
    CARD_SUBMISSION = "card_submission"
    JUDGING = "judging"
    RESULTS = "results"

@dataclass
class AuthenticatedPlayer:
    user_id: str  # Now using actual user ID from database (UUID)
    email: str
    username: str
    score: int = 0
    is_host: bool = False
    is_judge: bool = False
    hand: List[str] = None
    submitted_card: Optional[str] = None
    connected: bool = True
    humor_preferences: Dict[str, Any] = None  # User's learned humor preferences
    
    def __post_init__(self):
        if self.hand is None:
            self.hand = []
        if self.humor_preferences is None:
            self.humor_preferences = {}

@dataclass
class GameRound:
    round_number: int
    black_card: str
    judge_id: str  # Now using user ID (UUID)
    submissions: Dict[str, str]  # user_id -> white_card
    winner_id: Optional[str] = None
    winning_card: Optional[str] = None
    phase: RoundPhase = RoundPhase.CARD_SUBMISSION
    start_time: datetime = None
    
    def __post_init__(self):
        if self.start_time is None:
            self.start_time = datetime.now()

@dataclass
class PersonalizedCard:
    """A personalized card with persona information"""
    text: str
    persona_name: str
    persona_type: str  # "favorite", "exploration", etc.
    generated_for_round: int
    is_safe: bool = True  # Safety flag from detoxify

@dataclass
class GameState:
    """Game state for multiplayer CAH"""
    game_id: str
    players: Dict[str, AuthenticatedPlayer]  # user_id -> player (UUID)
    status: GameStatus
    current_round: Optional[GameRound]
    round_history: List[GameRound]
    settings: Dict[str, Any]
    created_at: datetime
    max_score: int = 5  # Game ends after 5 rounds
    max_players: int = 6
    prepared_white_cards: List[str] = None
    prepared_black_cards: List[str] = None
    # Round-specific card storage
    current_round_cards: Dict[str, List[PersonalizedCard]] = None  # player_id -> [PersonalizedCard]
    
    def __post_init__(self):
        if self.prepared_white_cards is None:
            self.prepared_white_cards = []
        if self.prepared_black_cards is None:
            self.prepared_black_cards = []
        if self.current_round_cards is None:
            self.current_round_cards = {}

class AuthenticatedMultiplayerCAHGame:
    """Multiplayer Cards Against Humanity game manager with authentication"""
    
    def __init__(self, humor_orchestrator: ImprovedHumorOrchestrator, persona_manager: PersonaManager):
        # Don't store db_session - we'll get it fresh for each operation
        self.games: Dict[str, GameState] = {}
        self.humor_orchestrator = humor_orchestrator
        self.persona_manager = persona_manager
        self.background_generation_running: Dict[str, bool] = {}
        
        # WebSocket connection management - store connections per game
        self.active_connections: Dict[str, Dict[str, Any]] = {}  # game_id -> {user_id: websocket}
        
        # Default black cards for quick start
        self.default_black_cards = [
            "What's the secret to a good relationship? _____",
            "What would grandma find disturbing? _____", 
            "What's the next Happy Meal toy? _____",
            "What did I bring back from Mexico? _____",
            "What's worse than finding a worm in your apple? _____",
            "What's the best way to get rich? _____",
            "What's the secret to success? _____",
            "What's the worst advice you can give? _____"
        ]
        
        # Default white cards for quick start
        self.default_white_cards = [
            "My crippling anxiety", "Student loan debt", "The void that stares back",
            "Emotional baggage", "My terrible life choices", "Existential dread",
            "Millennial burnout", "A disappointing birthday party", 
            "The crushing weight of responsibility", "My collection of participation trophies",
            "The awkward silence", "Unexpected emotional baggage", "A disappointing revelation",
            "My secret shame", "Questionable life choices", "The printer's existential crisis",
            "Accidentally sending 'love you' to the entire company", "The coffee machine's passive-aggressive messages",
            "My child's elaborate excuses for bedtime", "The toy that only works when I'm not looking",
            "My kid's negotiation skills are better than mine", "Accidentally speedrunning my morning routine",
            "Lag in real life", "Achievement unlocked: Adulting", "My browser history",
            "What I found in my browser history", "My secret guilty pleasure", "The worst part about adult life",
            "My most embarrassing moment", "The real reason I can't sleep at night", "My questionable decisions",
            "The thing I regret most", "My biggest fear", "What keeps me up at night", "My greatest weakness",
            "What I brought to the potluck", "My secret talent", "The thing I'm most ashamed of",
            "My biggest mistake", "What I learned the hard way", "My most embarrassing purchase",
            "The thing I can't live without", "My biggest pet peeve", "What I'm most afraid of",
            "My greatest achievement", "The thing I'm most proud of", "My biggest regret",
            "What I wish I knew earlier", "My most embarrassing moment", "The thing I can't stop thinking about",
            "What's my secret power?", "What will I bring back in time to convince people that I am a powerful wizard?",
            "What's the most emo?", "What gives me uncontrollable gas?", "What would complete my breakfast?",
            "What's the new fad diet?", "What's that sound?", "What helps Obama unwind?",
            "What never fails to liven up the party?"
        ]
    
    # WebSocket connection management methods
    def add_websocket_connection(self, game_id: str, user_id: str, websocket: Any):
        """Add a WebSocket connection for a user in a game"""
        if game_id not in self.active_connections:
            self.active_connections[game_id] = {}
        self.active_connections[game_id][user_id] = websocket
        logger.info(f"Added WebSocket connection for user {user_id} in game {game_id}")
    
    def remove_websocket_connection(self, game_id: str, user_id: str):
        """Remove a WebSocket connection for a user in a game"""
        if game_id in self.active_connections and user_id in self.active_connections[game_id]:
            del self.active_connections[game_id][user_id]
            logger.info(f"Removed WebSocket connection for user {user_id} in game {game_id}")
            
            # Clean up empty game connections
            if not self.active_connections[game_id]:
                del self.active_connections[game_id]
                logger.info(f"Cleaned up empty game connections for {game_id}")
    
    def get_websocket_connections(self, game_id: str) -> Dict[str, Any]:
        """Get all WebSocket connections for a game"""
        return self.active_connections.get(game_id, {})
    
    async def broadcast_to_game(self, game_id: str, message: dict, exclude_user: str = None):
        """Broadcast a message to all connected players in a game"""
        if game_id not in self.active_connections:
            logger.warning(f"No active WebSocket connections for game {game_id}")
            print(f"🎯 BROADCAST DEBUG: No active connections found for game {game_id}")
            return
        
        connections = self.active_connections[game_id]
        logger.info(f"Broadcasting to {len(connections)} players in game {game_id}")
        print(f"🎯 BROADCAST DEBUG: Found {len(connections)} connections: {list(connections.keys())}")
        
        # Also try to broadcast via main.py WebSocket connections if available
        try:
            import sys
            if 'main' in sys.modules:
                from ..api.main import active_websocket_connections
                if game_id in active_websocket_connections:
                    main_connections = active_websocket_connections[game_id]
                    logger.info(f"🎯 BROADCAST DEBUG: Found {len(main_connections)} connections in main.py for game {game_id}")
                    
                    # Broadcast to main.py connections as well
                    for user_id_str, websocket in main_connections.items():
                        try:
                            if exclude_user and user_id_str == exclude_user:
                                continue
                            await websocket.send_text(json.dumps(message))
                            logger.info(f"✅ Broadcasted via main.py to user {user_id_str}")
                        except Exception as e:
                            logger.warning(f"⚠️ Failed to broadcast via main.py to user {user_id_str}: {e}")
        except Exception as e:
            logger.debug(f"Could not broadcast via main.py: {e}")
        
        # Broadcast to game manager connections
        for user_id, websocket in connections.items():
            if exclude_user and user_id == exclude_user:
                continue
            try:
                await websocket.send_text(json.dumps(message))
                print(f"🎯 BROADCAST DEBUG: Successfully sent to user {user_id}")
            except Exception as e:
                logger.error(f"Failed to send message to user {user_id}: {e}")
                # Remove dead connection
                self.remove_websocket_connection(game_id, user_id)
    
    async def create_game(self, db: Session, game_id: str, host_user_id: str, settings: Dict[str, Any] = None) -> GameState:
        """Create a new game with authenticated host"""
        try:
            # Get host user from database using the provided session
            host_user = db.query(User).filter(User.id == host_user_id).first()
            if not host_user:
                raise ValueError(f"User {host_user_id} not found")
            
            # Create host player
            host_player = AuthenticatedPlayer(
                user_id=host_user.id,
                email=host_user.email,
                username=host_user.username or host_user.email.split('@')[0],
                is_host=True,
                humor_preferences=host_user.humor_preferences or {}
            )
            
            # Create game state
            game_state = GameState(
                game_id=game_id,
                players={host_user.id: host_player},
                status=GameStatus.WAITING,
                current_round=None,
                round_history=[],
                settings=settings or {},
                created_at=datetime.now()
            )
            
            # Save to database
            try:
                from ..models.database import Game, GamePlayer
                
                # Create game record
                db_game = Game(
                    id=game_id,
                    status=game_state.status.value,
                    settings=settings or {}
                )
                db.add(db_game)
                
                # Create host player record
                db_player = GamePlayer(
                    game_id=game_id,
                    user_id=host_user.id,
                    username=host_player.username,
                    is_host=True,
                    is_judge=False,
                    score=0
                )
                db.add(db_player)
                
                db.commit()
                logger.info(f"Game {game_id} saved to database")
                
            except Exception as db_error:
                logger.error(f"Failed to save game to database: {db_error}")
                db.rollback()
                # Continue with in-memory game even if database save fails
            
            self.games[game_id] = game_state
            logger.info(f"Game {game_id} created by user {host_user.email}")
            return game_state
            
        except Exception as e:
            logger.error(f"Failed to create game: {e}")
            raise
    
    async def join_game(self, db: Session, game_id: str, user_id: str) -> bool:
        """Join a game with authenticated user"""
        try:
            if game_id not in self.games:
                return False
            
            game = self.games[game_id]
            if game.status != GameStatus.WAITING:
                return False
            
            if user_id in game.players:
                return True  # Already in game
            
            if len(game.players) >= getattr(game, 'max_players', 6):
                return False
            
            # Get user from database using the provided session
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                return False
            
            # Create player
            player = AuthenticatedPlayer(
                user_id=user.id,
                email=user.email,
                username=user.username or user.email.split('@')[0],
                humor_preferences=user.humor_preferences or {}
            )
            
            # Save player to database
            try:
                from ..models.database import GamePlayer
                
                db_player = GamePlayer(
                    game_id=game_id,
                    user_id=user.id,
                    username=player.username,
                    is_host=False,
                    is_judge=False,
                    score=0
                )
                db.add(db_player)
                db.commit()
                logger.info(f"Player {user.email} saved to database for game {game_id}")
                
            except Exception as db_error:
                logger.error(f"Failed to save player to database: {db_error}")
                db.rollback()
                # Continue with in-memory player even if database save fails
            
            game.players[user_id] = player
            logger.info(f"User {user.email} joined game {game_id}")
            
            # Note: Broadcasting is handled by multiplayer_routes.py after this method returns
            # This prevents duplicate broadcasting and ensures proper synchronization
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to join game: {e}")
            return False
    
    async def start_game(self, db: Session, game_id: str) -> bool:
        """Start the game and begin first round"""
        try:
            logger.info(f"🎯 start_game CALLED for game {game_id}")
            
            if game_id not in self.games:
                logger.error(f"Game {game_id} not found in start_game")
                return False
            
            game = self.games[game_id]
            if game.status != GameStatus.WAITING or len(game.players) < 2:
                logger.error(f"Game {game_id} cannot start: status={game.status.value}, players={len(game.players)}")
                return False
            
            logger.info(f"🎯 Starting game {game_id} with {len(game.players)} players")
            game.status = GameStatus.STARTING
            
            # Prepare cards for all players based on their preferences
            logger.info(f"🎯 About to call _prepare_cards_for_game for game {game_id}")
            await self._prepare_cards_for_game(db, game)
            logger.info(f"🎯 _prepare_cards_for_game completed for game {game_id}")
            
            # Deal initial hands to all players
            logger.info(f"🎯 About to call _deal_initial_hands for game {game_id}")
            await self._deal_initial_hands(game)
            logger.info(f"🎯 _deal_initial_hands completed for game {game_id}")
            
            # Start first round
            logger.info(f"🎯 About to call _start_new_round for game {game_id}")
            await self._start_new_round(game)
            logger.info(f"🎯 _start_new_round completed for game {game_id}")
            
            game.status = GameStatus.IN_PROGRESS
            logger.info(f"Game {game_id} started with {len(game.players)} players")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start game: {e}")
            return False
    
    async def _deal_initial_hands(self, game: GameState):
        """Deal initial white cards to all players - use personalized cards if available, otherwise fallback"""
        try:
            cards_per_player = 3  # Reduced from 7 to 3 to match the new system
            logger.info(f"🎯 Dealing {cards_per_player} cards per player. Total prepared cards: {len(game.prepared_white_cards)}")
            
            # Check if we have personalized cards generated
            has_personalized_cards = hasattr(game, 'player_personalized_cards') and game.player_personalized_cards
            
            if has_personalized_cards:
                logger.info(f"🎯 Using personalized cards for initial hands")
                # Use personalized cards for initial hands
                for player_id, player in game.players.items():
                    if player_id in game.player_personalized_cards:
                        personalized_cards = game.player_personalized_cards[player_id]
                        cards_to_deal = min(cards_per_player, len(personalized_cards))
                        
                        player.hand = []
                        for i in range(cards_to_deal):
                            if personalized_cards:
                                card = personalized_cards.pop(0)  # Take from the top
                                player.hand.append(card)
                                logger.info(f"✅ Dealt personalized card {i+1} to {player.username}: {card}")
                        
                        logger.info(f"✅ Dealt {len(player.hand)} personalized cards to {player.username}")
                    else:
                        logger.warning(f"⚠️ No personalized cards found for {player.username}, using fallback")
                        self._give_fallback_cards(player, cards_per_player)
            else:
                logger.info(f"🎯 No personalized cards available, using fallback cards for initial hands")
                # Use fallback cards if no personalized cards available
                for player in game.players.values():
                    self._give_fallback_cards(player, cards_per_player)
                
        except Exception as e:
            logger.error(f"❌ Failed to deal initial hands: {e}")
            # Critical fallback: give each player emergency cards
            for player in game.players.values():
                if len(player.hand) == 0:
                    logger.warning(f"🚨 Player {player.user_id} has no cards, giving emergency fallback")
                    self._give_emergency_cards(player, 3)
    
    def _give_fallback_cards(self, player: AuthenticatedPlayer, num_cards: int):
        """Give fallback cards to a player"""
        fallback_cards = [
            "My crippling anxiety", "Student loan debt", "The void that stares back",
            "Emotional baggage", "My terrible life choices", "Existential dread",
            "Millennial burnout", "A disappointing birthday party", "The crushing weight of responsibility",
            "My collection of participation trophies", "The awkward silence", "Unexpected emotional baggage",
            "A disappointing revelation", "My secret shame", "Questionable life choices",
            "The printer's existential crisis", "Accidentally sending 'love you' to the entire company",
            "The coffee machine's passive-aggressive messages", "My child's elaborate excuses for bedtime",
            "The toy that only works when I'm not looking", "My kid's negotiation skills are better than mine",
            "Accidentally speedrunning my morning routine", "Lag in real life", "Achievement unlocked: Adulting",
            "My browser history", "What I found in my browser history", "My secret guilty pleasure",
            "The worst part about adult life", "My most embarrassing moment", "The real reason I can't sleep at night",
            "My questionable decisions", "The thing I regret most", "My biggest fear", "What keeps me up at night",
            "My greatest weakness", "What I brought to the potluck", "My secret talent", "The thing I'm most ashamed of",
            "My biggest mistake", "What I learned the hard way", "My most embarrassing purchase",
            "The thing I can't live without", "My biggest pet peeve", "What I'm most afraid of",
            "My greatest achievement", "The thing I'm most proud of", "My biggest regret",
            "What I wish I knew earlier", "My most embarrassing moment", "The thing I can't stop thinking about"
        ]
        
        # Shuffle fallback cards
        random.shuffle(fallback_cards)
        
        player.hand = []
        for i in range(num_cards):
            if i < len(fallback_cards):
                card = fallback_cards[i]
                cleaned_card = self._validate_and_clean_card(card)
                if cleaned_card:
                    player.hand.append(cleaned_card)
                    logger.debug(f"✅ Dealt fallback card {i+1}: {cleaned_card}")
                else:
                    # Use emergency fallback if validation fails
                    emergency_card = "Something unexpectedly funny"
                    player.hand.append(emergency_card)
                    logger.debug(f"🔄 Used emergency fallback card {i+1}: {emergency_card}")
            else:
                # If we run out of fallback cards, use emergency cards
                emergency_card = "Something unexpectedly funny"
                player.hand.append(emergency_card)
                logger.debug(f"🔄 Used emergency fallback card {i+1}: {emergency_card}")
        
        logger.info(f"✅ Dealt {len(player.hand)} fallback cards to player {player.user_id} ({player.username})")
        
    def _give_emergency_cards(self, player: AuthenticatedPlayer, num_cards: int):
        """Give emergency fallback cards to a player"""
        emergency_cards = [
            "Something unexpectedly funny", "A terrible mistake", "My hidden talent",
            "An awkward situation", "The wrong answer", "My questionable decisions",
            "The awkward silence", "Unexpected emotional baggage", "A disappointing revelation"
        ]
        player.hand = emergency_cards[:num_cards]
        logger.info(f"🚨 Emergency fallback: gave {len(player.hand)} emergency cards to player {player.user_id}")
    
    async def _deal_card_to_player(self, game: GameState, player: AuthenticatedPlayer):
        """Deal 3 personalized white cards to a player from pre-generated cards"""
        try:
            logger.info(f"🎯 _deal_card_to_player CALLED for player {player.username} in game {game.game_id}")
            logger.info(f"🎯 _deal_card_to_player CALLED from: {self._get_calling_function()}")
            
            # Find the player_id key for this player
            player_id = None
            for pid, p in game.players.items():
                if p.user_id == player.user_id:
                    player_id = pid
                    break
            
            logger.info(f"🎯 Found player_id: {player_id} for player {player.username}")
            
            # Check if we have pre-generated personalized cards for this player
            if player_id and hasattr(game, 'player_personalized_cards') and player_id in game.player_personalized_cards:
                personalized_cards = game.player_personalized_cards[player_id]
                logger.info(f"🎯 Found {len(personalized_cards)} pre-generated cards for player {player.username}")
                
                # Deal up to 3 cards from the personalized deck (don't force exactly 3)
                cards_dealt = 0
                cards_to_deal = min(3, len(personalized_cards))
                
                for i in range(cards_to_deal):
                    if personalized_cards and len(personalized_cards) > 0:
                        card = personalized_cards.pop(0)  # Take from the top
                        if card not in player.hand:
                            player.hand.append(card)
                            cards_dealt += 1
                            logger.info(f"✅ Dealt personalized card {cards_dealt} for {player.username}: {card}")
                
                logger.info(f"✅ Dealt {cards_dealt} personalized cards to {player.username} from {cards_to_deal} available")
                
                # Only add fallback cards if we have NO personalized cards at all AND player has less than 3 cards
                if cards_dealt == 0 and len(player.hand) < 3:
                    logger.warning(f"⚠️ No personalized cards available for {player.username}, using minimal fallback")
                    self._give_fallback_cards(player, 3 - len(player.hand))
                
            else:
                # Fallback to default cards if no personalized cards available
                logger.warning(f"No personalized cards found for {player.username}, using fallback cards")
                logger.warning(f"player_id: {player_id}, hasattr: {hasattr(game, 'player_personalized_cards')}")
                if hasattr(game, 'player_personalized_cards'):
                    logger.warning(f"Available player IDs: {list(game.player_personalized_cards.keys())}")
                
                # Only add fallback cards if player has less than 3 cards
                if len(player.hand) < 3:
                    self._give_fallback_cards(player, 3 - len(player.hand))
                logger.info(f"Added fallback cards to {player.username}")
                
        except Exception as e:
            logger.error(f"Error dealing cards for {player.username}: {e}")
            # Ultimate fallback - only add cards if player has less than 3
            if len(player.hand) < 3:
                self._give_emergency_cards(player, 3 - len(player.hand))
            logger.info(f"Added emergency fallback cards to {player.username}")
    
    def _get_calling_function(self):
        """Get the name of the function that called this method"""
        try:
            frame = inspect.currentframe()
            caller_frame = frame.f_back
            if caller_frame:
                return caller_frame.f_code.co_name
            return "unknown"
        except:
            return "unknown"
    
    async def _prepare_cards_for_game(self, db: Session, game: GameState):
        """Prepare initial game setup - cards will be generated round-by-round"""
        try:
            logger.info(f"🎯 _prepare_cards_for_game CALLED for game {game.game_id} with {len(game.players)} players")
            
            # Prepare black cards
            game.prepared_black_cards = self.default_black_cards.copy()
            random.shuffle(game.prepared_black_cards)
            
            # Also prepare some general fallback cards
            game.prepared_white_cards = self._get_default_white_cards().copy()
            random.shuffle(game.prepared_white_cards)

            logger.info(f"🎯 Game setup complete! Cards will be generated round-by-round for context.")
            logger.info(f"🎯 Total black cards prepared: {len(game.prepared_black_cards)}")
            
        except Exception as e:
            logger.error(f"❌ Failed to prepare initial cards: {e}")
            # Critical fallback - use default cards for everyone
            game.prepared_white_cards = self._get_default_white_cards().copy()
            game.prepared_black_cards = self.default_black_cards.copy()
            logger.info(f"🔄 Critical fallback: using default cards for all players")
    
    async def _generate_remaining_cards_in_background(self, db: Session, game: GameState):
        """Generate remaining 3 rounds worth of cards in background while game is running"""
        try:
            logger.info(f"🔄 Background task: generating remaining 3 rounds of cards...")
            
            for player_id, player in game.players.items():
                try:
                    # Generate 9 more cards (3 rounds × 3 cards = 3 generation calls)
                    additional_cards = await self._generate_personalized_cards_for_player(player, 3)  # 3 calls = 9 cards
                    
                    # Add to existing cards
                    if hasattr(game, 'player_personalized_cards') and player_id in game.player_personalized_cards:
                        game.player_personalized_cards[player_id].extend(additional_cards)
                        logger.info(f"✅ Added {len(additional_cards)} background cards for {player.username}")
                    else:
                        logger.warning(f"⚠️ No existing cards found for {player.username} in background task")
                        
                except Exception as e:
                    logger.error(f"❌ Background generation failed for {player.username}: {e}")
            
            logger.info(f"🎯 Background card generation completed! All 5 rounds ready.")
            
        except Exception as e:
            logger.error(f"❌ Background card generation failed: {e}")
    
    async def _generate_personalized_cards_for_player(self, player: AuthenticatedPlayer, num_calls: int, black_card: str = None, round_number: int = None) -> List[PersonalizedCard]:
        """Generate personalized cards for a specific player - now returns PersonalizedCard objects with safety info"""
        try:
            context = f"Generate a funny white card response to this black card: '{black_card}'" if black_card else "Generate a short, funny phrase for Cards Against Humanity white card"
            logger.info(f"🎯 Generating {num_calls * 3} cards for {player.username} for round {round_number} with context: {black_card}")
            personalized_cards = []
            
            for i in range(num_calls):
                try:
                    logger.info(f"🎯 Making generation call {i+1}/{num_calls} for {player.username}")
                    
                    # Get player's top 2 favorite personas
                    favorite_personas = await self._get_player_favorites(player.user_id) or []
                    
                    # Add 1 random/exploration persona to ensure 2 favorites + 1 exploration
                    final_personas = favorite_personas[:2]  # Take top 2 favorites
                    
                    # Add 1 exploration persona (different from favorites)
                    exploration_personas = ["Dark Humor Connoisseur", "Techie Tomfoolery", "Corporate Humor Specialist"]
                    for exploration in exploration_personas:
                        if exploration not in final_personas:
                            final_personas.append(exploration)
                            logger.info(f"🎭 Added exploration persona: {exploration}")
                            break
                    
                    logger.info(f"🎭 Final personas for {player.username}: {final_personas} (2 favorites + 1 exploration)")
                    
                    # Create humor request with black card context
                    request = HumorRequest(
                        context=f"{context}. Make it edgy and humorous, suitable for adult audiences.",
                        audience="adults",
                        topic="general",
                        user_id=str(player.user_id),
                        humor_type="edgy",
                        card_type="white",
                        favorite_personas=final_personas  # Pass 2 favorites + 1 exploration
                    )
                    
                    # Generate personalized cards using CrewAI - this returns 3 cards (one per persona)
                    generated_cards = await self.humor_orchestrator.generate_and_evaluate_humor(request)
                    
                    logger.info(f"🎯 AI Generation returned: {generated_cards}")
                    
                    # Handle both list and dictionary response formats
                    if generated_cards and isinstance(generated_cards, list) and len(generated_cards) > 0:
                        logger.info(f"🎯 Processing {len(generated_cards)} AI-generated cards (list format)")
                        
                        # Add ALL generated cards from this call (typically 3 cards)
                        for j, card_result in enumerate(generated_cards):
                            logger.info(f"🎯 Processing card result {j+1}: {card_result}")
                            
                            card_text = card_result.text if hasattr(card_result, 'text') else str(card_result)
                            logger.info(f"🎯 Extracted card text: '{card_text}'")
                            
                            # Get persona name and safety info
                            persona_name = "Unknown Persona"
                            is_safe = True
                            
                            if hasattr(card_result, 'persona_name'):
                                persona_name = card_result.persona_name
                            if hasattr(card_result, 'is_safe'):
                                is_safe = card_result.is_safe
                            if hasattr(card_result, 'toxicity_score'):
                                # Consider card safe if toxicity score is low
                                is_safe = getattr(card_result, 'toxicity_score', 0.1) < 0.5
                            
                            cleaned_card = self._validate_and_clean_card(card_text)
                            logger.info(f"🎯 After cleaning: '{cleaned_card}' (safe: {is_safe})")
                            
                            if cleaned_card and cleaned_card not in [card.text for card in personalized_cards]:
                                # Create PersonalizedCard with safety info
                                personalized_card = PersonalizedCard(
                                    text=cleaned_card,
                                    persona_name=persona_name,
                                    persona_type="favorite" if j < 2 else "exploration",
                                    generated_for_round=round_number or 1,
                                    is_safe=is_safe
                                )
                                personalized_cards.append(personalized_card)
                                logger.info(f"✅ Generated card {len(personalized_cards)} for {player.username}: {cleaned_card} (by {persona_name}, safe: {is_safe})")
                            else:
                                if not cleaned_card:
                                    logger.warning(f"⚠️ Card was cleaned to empty/None: '{card_text}' -> '{cleaned_card}'")
                                else:
                                    logger.warning(f"⚠️ Card already exists (duplicate): '{cleaned_card}'")
                        
                        logger.info(f"✅ Generation call {i+1} returned {len(generated_cards)} cards for {player.username}")
                        logger.info(f"✅ Total cards so far: {len(personalized_cards)}")
                        
                    elif generated_cards and isinstance(generated_cards, dict) and generated_cards.get('success') and generated_cards.get('results'):
                        logger.info(f"🎯 Processing {len(generated_cards['results'])} AI-generated cards (dict format)")
                        
                        # Handle dictionary response format
                        for j, result in enumerate(generated_cards['results']):
                            logger.info(f"🎯 Processing result {j+1}: {result}")
                            
                            if 'generation' in result and hasattr(result['generation'], 'text'):
                                card_text = result['generation'].text
                                logger.info(f"🎯 Extracted card text: '{card_text}'")
                                
                                cleaned_card = self._validate_and_clean_card(card_text)
                                logger.info(f"🎯 After cleaning: '{cleaned_card}'")
                                
                                if cleaned_card and cleaned_card not in [card.text for card in personalized_cards]:
                                    # Create PersonalizedCard with safety info
                                    personalized_card = PersonalizedCard(
                                        text=cleaned_card,
                                        persona_name="Generated Persona",
                                        persona_type="favorite" if j < 2 else "exploration",
                                        generated_for_round=round_number or 1,
                                        is_safe=True  # Assume safe for dict format
                                    )
                                    personalized_cards.append(personalized_card)
                                    logger.info(f"✅ Generated card {len(personalized_cards)} for {player.username}: {cleaned_card}")
                                else:
                                    if not cleaned_card:
                                        logger.warning(f"⚠️ Card was cleaned to empty/None: '{card_text}' -> '{cleaned_card}'")
                                    else:
                                        logger.warning(f"⚠️ Card already exists (duplicate): '{cleaned_card}'")
                            else:
                                logger.warning(f"⚠️ Result {j+1} missing 'generation' or 'text' attribute: {result}")
                        
                        logger.info(f"✅ Generation call {i+1} returned {len(generated_cards['results'])} cards for {player.username}")
                        logger.info(f"✅ Total cards so far: {len(personalized_cards)}")
                        
                    else:
                        logger.warning(f"⚠️ AI generation failed or returned empty for call {i+1}")
                        logger.warning(f"⚠️ Generated cards: {generated_cards}")
                        # Fallback - add 3 simple personalized cards for this call
                        for j in range(3):
                            fallback_card = PersonalizedCard(
                                text=f"{player.username}'s secret talent #{len(personalized_cards)+1}",
                                persona_name="Fallback Persona",
                                persona_type="fallback",
                                generated_for_round=round_number or 1,
                                is_safe=True
                            )
                            if fallback_card.text not in [card.text for card in personalized_cards]:
                                personalized_cards.append(fallback_card)
                                logger.info(f"🔄 Used fallback card {len(personalized_cards)} for {player.username}: {fallback_card.text}")
                            
                except Exception as e:
                    logger.error(f"Error in generation call {i+1} for {player.username}: {e}")
                    # Simple fallback - add 3 cards for this failed call
                    for j in range(3):
                        fallback_card = PersonalizedCard(
                            text=f"{player.username}'s hidden power #{len(personalized_cards)+1}",
                            persona_name="Emergency Persona",
                            persona_type="fallback",
                            generated_for_round=round_number or 1,
                            is_safe=True
                        )
                        if fallback_card.text not in [card.text for card in personalized_cards]:
                            personalized_cards.append(fallback_card)
                            logger.info(f"🔄 Used error fallback card {len(personalized_cards)} for {player.username}: {fallback_card.text}")
            
            logger.info(f"🎯 Generated {len(personalized_cards)} total cards for {player.username} from {num_calls} generation calls")
            logger.info(f"🎯 Final cards: {[card.text for card in personalized_cards]}")
            return personalized_cards
            
        except Exception as e:
            logger.error(f"Error generating personalized cards for {player.username}: {e}")
            # Ultimate fallback - return fallback PersonalizedCards
            fallback_cards = []
            for i in range(3):
                fallback_card = PersonalizedCard(
                    text=f"{player.username}'s backup plan #{i+1}",
                    persona_name="Emergency Persona",
                    persona_type="fallback",
                    generated_for_round=round_number or 1,
                    is_safe=True
                )
                fallback_cards.append(fallback_card)
            return fallback_cards
    
    async def _get_player_favorites(self, user_id: str) -> Optional[List[str]]:
        """Get player's top 2 favorite personas"""
        try:
            from agent_system.knowledge.improved_aws_knowledge_base import improved_aws_knowledge_base
            
            user_preferences = await improved_aws_knowledge_base.get_user_preference(user_id)
            if user_preferences and user_preferences.liked_personas:
                # Return top 2 favorite personas
                return user_preferences.liked_personas[:2]
            return None
        except Exception as e:
            logger.error(f"Error getting favorites for {user_id}: {e}")
            return None

    def _get_best_persona_for_player(self, db: Session, player: AuthenticatedPlayer) -> str:
        """Get the best persona for a player based on their preferences"""
        try:
            # Query persona preferences for this user - convert user_id to string for database compatibility
            preferences = db.query(PersonaPreference).filter(
                PersonaPreference.user_id == str(player.user_id)
            ).order_by(PersonaPreference.preference_score.desc()).first()
            
            if preferences:
                # Get persona name
                persona = db.query(Persona).filter(Persona.id == preferences.persona_id).first()
                if persona:
                    return persona.name
            
            # Fallback to default persona
            return "Dark Humor Connoisseur"
            
        except Exception as e:
            logger.error(f"Failed to get best persona: {e}")
            return "Dark Humor Connoisseur"
    
    def _get_default_white_cards(self) -> List[str]:
        """Get default white cards for fallback"""
        return [
            "My collection of terrible puns",
            "Existential crisis support group",
            "The void but it's surprisingly supportive",
            "Capitalism but as a houseplant",
            "Anxiety served with ranch dressing",
            "My sanity (sold separately)",
            "Goldfish crackers for dinner again",
            "Kid logic applied to adult problems"
        ]
    
    async def _start_new_round(self, game: GameState):
        """Start a new round in the game with round-specific card generation"""
        try:
            # Select judge (rotate through players)
            round_number = len(game.round_history) + 1
            player_ids = list(game.players.keys())
            judge_id = player_ids[(round_number - 1) % len(player_ids)]
            
            # Select black card
            if game.prepared_black_cards:
                black_card = random.choice(game.prepared_black_cards)
            else:
                # Use default black cards if none prepared
                default_black_cards = [
                    "What's the secret to a good relationship? _____",
                    "What would grandma find disturbing? _____", 
                    "What's the next Happy Meal toy? _____",
                    "What did I bring back from Mexico? _____",
                    "What's worse than finding a worm in your apple? _____"
                ]
                black_card = random.choice(default_black_cards)
            
            # Create new round
            new_round = GameRound(
                round_number=round_number,
                black_card=black_card,
                judge_id=judge_id,
                submissions={}
            )
            
            game.current_round = new_round
            logger.info(f"🎯 New round {round_number} created with judge {judge_id}")
            logger.info(f"🎯 Round phase: {new_round.phase.value}")
            logger.info(f"🎯 Total players: {len(game.players)}")
            logger.info(f"🎯 Non-judge players: {[p.user_id for p in game.players.values() if p.user_id != judge_id]}")
            
            # Set judge status for all players
            for player in game.players.values():
                player.is_judge = (player.user_id == judge_id)
                player.submitted_card = None  # Reset submission status
                logger.info(f"🎯 Player {player.user_id} ({player.username}): is_judge={player.is_judge}, judge_id={judge_id}")
            
            # Generate round-specific cards for each player (excluding judge)
            logger.info(f"🎯 Generating round-specific cards for round {round_number} with black card: '{black_card}'")
            
            for player in game.players.values():
                if player.user_id != judge_id:
                    try:
                        # Generate 3 personalized cards for this specific round and black card
                        personalized_cards = await self._generate_personalized_cards_for_player(
                            player, 1, black_card, round_number  # 1 call = 3 cards, with black card context
                        )
                        
                        # Store round-specific cards
                        game.current_round_cards[player.user_id] = personalized_cards
                        logger.info(f"🎯 Generated {len(personalized_cards)} round-specific cards for {player.username}")
                        
                        # Deal cards to player's hand
                        player.hand = [card.text for card in personalized_cards if card.is_safe]  # Only safe cards
                        logger.info(f"🎯 Player {player.user_id} ({player.username}) got {len(player.hand)} safe cards for round {round_number}")
                        
                    except Exception as e:
                        logger.error(f"❌ Failed to generate round-specific cards for {player.username}: {e}")
                        # Fallback: give 3 simple cards
                        player.hand = [
                            f"{player.username}'s creative response #1",
                            f"{player.username}'s creative response #2", 
                            f"{player.username}'s creative response #3"
                        ]
                        logger.info(f"🔄 Used fallback cards for {player.username}")
            
            logger.info(f"Round {round_number} started in game {game.game_id}")
            
        except Exception as e:
            logger.error(f"Failed to start new round: {e}")
    
    async def submit_card(self, game_id: str, user_id: str, white_card: str) -> bool:
        """Submit a white card for the current round"""
        try:
            if game_id not in self.games:
                logger.error(f"Game {game_id} not found in submit_card")
                return False
            
            game = self.games[game_id]
            if not game:
                logger.error(f"Game {game_id} is None in submit_card")
                return False
                
            if not game.players:
                logger.error(f"Game {game_id} has no players in submit_card")
                return False
                
            if user_id not in game.players:
                logger.error(f"User {user_id} not found in game {game_id} players: {list(game.players.keys())}")
                return False
            
            if not game.current_round or game.current_round.phase != RoundPhase.CARD_SUBMISSION:
                logger.error(f"Game {game_id} not in card submission phase")
                return False
            
            if user_id == game.current_round.judge_id:
                logger.info(f"Judge {user_id} cannot submit cards")
                return False  # Judge can't submit cards
            
            if user_id in game.current_round.submissions:
                logger.info(f"User {user_id} already submitted a card")
                return False  # Already submitted
            
            # Verify the card is in player's hand
            player = game.players[user_id]
            if not player.hand:
                logger.error(f"Player {user_id} has no hand")
                return False
                
            if white_card not in player.hand:
                logger.warning(f"Card '{white_card}' not found in player {user_id}'s hand. Available: {player.hand}")
                return False
            
            # Remove the card from player's hand
            player.hand.remove(white_card)
            
            # Record submission
            game.current_round.submissions[user_id] = white_card
            logger.info(f"🎯 SUBMISSION RECORDED: user_id={user_id}, card='{white_card}'")
            logger.info(f"🎯 TOTAL SUBMISSIONS NOW: {len(game.current_round.submissions)}")
            logger.info(f"🎯 SUBMISSIONS CONTENT: {game.current_round.submissions}")
            
            # CRITICAL: Verify the submission was actually stored
            if user_id in game.current_round.submissions:
                logger.info(f"✅ VERIFICATION: Submission confirmed stored for user {user_id}")
                logger.info(f"✅ VERIFICATION: Stored card: '{game.current_round.submissions[user_id]}'")
                logger.info(f"✅ VERIFICATION: Total submissions after store: {len(game.current_round.submissions)}")
            else:
                logger.error(f"❌ CRITICAL ERROR: Submission NOT stored for user {user_id}!")
                logger.error(f"❌ CRITICAL ERROR: Available submissions: {game.current_round.submissions}")
                logger.error(f"❌ CRITICAL ERROR: Game ID: {game_id}")
                logger.error(f"❌ CRITICAL ERROR: Current round phase: {game.current_round.phase.value}")
            
            # CRITICAL: Verify game instance identity
            logger.info(f"🎯 GAME INSTANCE DEBUG: Game ID in memory: {id(game)}")
            logger.info(f"🎯 GAME INSTANCE DEBUG: Game ID value: {game.game_id}")
            logger.info(f"🎯 GAME INSTANCE DEBUG: Current round ID: {id(game.current_round)}")
            logger.info(f"🎯 GAME INSTANCE DEBUG: Submissions dict ID: {id(game.current_round.submissions)}")
            
            # Also update player's submitted_card field for consistency
            if user_id in game.players:
                game.players[user_id].submitted_card = white_card
                logger.info(f"🎯 Updated player {user_id} submitted_card field")
            
            # Deal new card to player
            await self._deal_card_to_player(game, player)
            
            # Check if all players have submitted
            non_judge_players = [p for p in game.players.values() if p.user_id != game.current_round.judge_id]
            logger.info(f"🎯 Phase transition check: submissions={len(game.current_round.submissions)}, non_judge_players={len(non_judge_players)}")
            logger.info(f"🎯 Current submissions: {game.current_round.submissions}")
            logger.info(f"🎯 Non-judge players: {[p.user_id for p in non_judge_players]}")
            logger.info(f"🎯 Judge ID type: {type(game.current_round.judge_id)}, value: {game.current_round.judge_id}")
            logger.info(f"🎯 Player IDs types: {[type(p.user_id) for p in game.players.values()]}")
            logger.info(f"🎯 Submissions keys: {list(game.current_round.submissions.keys())}")
            logger.info(f"🎯 Submissions key types: {[type(k) for k in game.current_round.submissions.keys()]}")
            
            if len(game.current_round.submissions) == len(non_judge_players):
                # Move to judging phase
                game.current_round.phase = RoundPhase.JUDGING
                logger.info(f"🎯 SUCCESS: All players submitted, moving to judging phase for game {game_id}")
                logger.info(f"🎯 New phase: {game.current_round.phase.value}")
                
                # Get judge info for the broadcast
                judge_username = "Unknown"
                if game.current_round.judge_id in game.players:
                    judge_username = game.players[game.current_round.judge_id].username
                
                # Broadcast phase change to all players with submissions visible
                try:
                    phase_change_message = {
                        "type": "phase_changed",
                        "game_id": game_id,
                        "new_phase": "judging",
                        "message": f"All cards submitted! {judge_username} can now judge the round.",
                        "judge_id": str(game.current_round.judge_id),
                        "judge_username": judge_username,
                        "submissions_count": len(game.current_round.submissions),
                        "submissions": [
                            {
                                "card": card,
                                "player_username": game.players[player_id].username if player_id in game.players else "Unknown",
                                "player_id": str(player_id)
                            }
                            for player_id, card in game.current_round.submissions.items()
                        ]
                    }
                    
                    await self.broadcast_to_game(game_id, phase_change_message)
                    logger.info(f"🎯 Broadcasted phase change to judging with {len(game.current_round.submissions)} submissions")
                    
                    # ENHANCED: Send special judge notification with immediate submissions visibility
                    try:
                        judge_game_state = self.get_game_state_for_user(game_id, game.current_round.judge_id)
                        if judge_game_state:
                            judge_message = {
                                "type": "judge_ready_to_judge",
                                "game_id": game_id,
                                "message": f"All cards are in! You can now judge the round.",
                                "game_state": judge_game_state,
                                "submissions_for_judging": self.get_submissions_for_judging(game_id, game.current_round.judge_id)
                            }
                            
                            # Send directly to judge via WebSocket
                            websocket_connections = self.get_websocket_connections(game_id)
                            if game.current_round.judge_id in websocket_connections:
                                judge_websocket = websocket_connections[game.current_round.judge_id]
                                await judge_websocket.send_text(json.dumps(judge_message))
                                logger.info(f"🎯 ✅ Sent special judge notification to judge {game.current_round.judge_id}")
                            
                            # Also try main.py WebSocket connections for judge
                            try:
                                import sys
                                if 'main' in sys.modules:
                                    from ..api.main import active_websocket_connections
                                    if game_id in active_websocket_connections:
                                        main_connections = active_websocket_connections[game_id]
                                        judge_id_str = str(game.current_round.judge_id)
                                        if judge_id_str in main_connections:
                                            judge_main_websocket = main_connections[judge_id_str]
                                            await judge_main_websocket.send_text(json.dumps(judge_message))
                                            logger.info(f"🎯 ✅ Sent judge notification via main.py to judge {judge_id_str}")
                            except Exception as main_error:
                                logger.debug(f"Could not send via main.py: {main_error}")
                                            
                    except Exception as judge_error:
                        logger.error(f"❌ Failed to send judge notification: {judge_error}")
                    
                except Exception as broadcast_error:
                    logger.error(f"❌ Failed to broadcast phase change: {broadcast_error}")
            else:
                logger.info(f"🎯 Still waiting: {len(game.current_round.submissions)}/{len(non_judge_players)} players submitted")
            
            logger.info(f"User {user_id} submitted card '{white_card}' in game {game_id}. Remaining hand: {len(player.hand)} cards")
            
            # Broadcast updated game state to all players
            try:
                await self.broadcast_to_game(game_id, {
                    "type": "game_state_updated",
                    "game_id": game_id,
                    "message": f"Player {player.username} submitted a card"
                })
                logger.info(f"🎯 Broadcasted game state update after card submission")
                
                # Also broadcast refreshed game state to all players to ensure they see the current state
                for player_id, game_player in game.players.items():
                    try:
                        player_game_state = self.get_game_state_for_user(game_id, player_id)
                        if player_game_state:
                            websocket_connections = self.get_websocket_connections(game_id)
                            if player_id in websocket_connections:
                                websocket = websocket_connections[player_id]
                                await websocket.send_text(json.dumps({
                                    "type": "game_state_refreshed",
                                    "game_state": player_game_state,
                                    "message": f"Game state updated after {player.username} submitted a card"
                                }))
                                logger.info(f"🎯 Sent refreshed game state to player {player_id}")
                    except Exception as player_refresh_error:
                        logger.warning(f"⚠️ Could not refresh game state for player {player_id}: {player_refresh_error}")
                        
            except Exception as broadcast_error:
                logger.error(f"❌ Failed to broadcast game state update: {broadcast_error}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to submit card for user {user_id} in game {game_id}: {e}")
            # Log the current game state for debugging
            if game_id in self.games:
                game = self.games[game_id]
                logger.error(f"Game state after error: players={len(game.players) if game.players else 0}, current_round={game.current_round.phase.value if game.current_round else 'None'}")
            return False
    
    async def judge_round(self, db: Session, game_id: str, judge_id: int, winning_card: str) -> bool:
        """Judge selects the winning card"""
        try:
            if game_id not in self.games:
                logger.error(f"Game {game_id} not found in judge_round")
                return False
            
            game = self.games[game_id]
            if not game.current_round:
                logger.error(f"No current round in game {game_id}")
                return False
                
            if game.current_round.phase != RoundPhase.JUDGING:
                logger.error(f"Game {game_id} not in judging phase. Current phase: {game.current_round.phase.value}")
                return False
            
            if game.current_round.judge_id != judge_id:
                logger.error(f"User {judge_id} is not the judge for game {game_id}. Judge is: {game.current_round.judge_id}")
                return False
            
            # Validate that the winning card exists in submissions
            if not game.current_round.submissions:
                logger.error(f"No submissions found in game {game_id}")
                return False
            
            # Find the user who submitted the winning card
            winning_user_id = None
            logger.info(f"🎯 JUDGING: Looking for winning card '{winning_card}' in submissions: {game.current_round.submissions}")
            
            for user_id, card in game.current_round.submissions.items():
                logger.info(f"🎯 JUDGING: Checking submission - user_id: {user_id}, card: '{card}', matches: {card == winning_card}")
                if card == winning_card:
                    winning_user_id = user_id
                    logger.info(f"🎯 JUDGING: Found winning card! User {user_id} wins with '{card}'")
                    break
            
            if winning_user_id is None:
                logger.error(f"🎯 JUDGING ERROR: Winning card '{winning_card}' not found in submissions: {game.current_round.submissions}")
                logger.error(f"🎯 JUDGING ERROR: Available submissions: {list(game.current_round.submissions.values())}")
                return False
            
            # Record winner
            game.current_round.winner_id = winning_user_id
            game.current_round.winning_card = winning_card
            game.current_round.phase = RoundPhase.RESULTS
            
            # Update scores
            if winning_user_id in game.players:
                game.players[winning_user_id].score += 1
            
            # Store learning data for the winning player
            await self._store_learning_data(db, game, winning_user_id, game.current_round)
            
            # Check if game is over
            if any(player.score >= game.max_score for player in game.players.values()):
                game.status = GameStatus.FINISHED
            else:
                # Start next round
                await self._start_new_round(game)
            
            logger.info(f"Round judged in game {game_id}, winner: {winning_user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to judge round: {e}")
            return False
    
    async def _store_learning_data(self, db: Session, game: GameState, winning_user_id: str, round_data: GameRound):
        """Store learning data for the winning player"""
        try:
            # Create user feedback record
            persona_name = self._get_best_persona_for_player(db, game.players[winning_user_id])
            feedback = UserFeedback(
                user_id=str(winning_user_id),
                persona_name=persona_name,
                feedback_score=9.0,  # High score for winning
                context=round_data.black_card,
                response_text=round_data.winning_card,
                topic="game_win",
                audience="friends",
                liked=True,
                humor_rating=5,
                created_at=datetime.now()
            )
            
            db.add(feedback)
            
            # Update persona preference
            persona = db.query(Persona).filter(Persona.name == persona_name).first()
            
            if persona:
                # Check if preference exists
                existing_pref = db.query(PersonaPreference).filter(
                    PersonaPreference.user_id == str(winning_user_id),
                    PersonaPreference.persona_id == persona.id
                ).first()
                
                if existing_pref:
                    # Update existing preference
                    existing_pref.preference_score = min(10.0, existing_pref.preference_score + 0.5)
                    existing_pref.interaction_count += 1
                    existing_pref.last_interaction = datetime.now()
                else:
                    # Create new preference
                    new_pref = PersonaPreference(
                        user_id=str(winning_user_id),
                        persona_id=persona.id,
                        preference_score=8.0,
                        interaction_count=1,
                        last_interaction=datetime.now(),
                        created_at=datetime.now(),
                        updated_at=datetime.now()
                    )
                    db.add(new_pref)
            
            db.commit()
            logger.info(f"Learning data stored for user {winning_user_id}")
            
        except Exception as e:
            logger.error(f"Failed to store learning data: {e}")
            db.rollback()
    
    def get_game_state(self, game_id: str) -> Optional[GameState]:
        """Get current game state"""
        return self.games.get(game_id)
    
    def get_player_info(self, game_id: str, user_id: str) -> Optional[AuthenticatedPlayer]:
        """Get player information for a specific user"""
        if game_id in self.games:
            return self.games[game_id].players.get(user_id)
        return None
    
    def leave_game(self, game_id: str, user_id: str) -> bool:
        """Leave a game"""
        try:
            if game_id in self.games:
                game = self.games[game_id]
                if user_id in game.players:
                    del game.players[user_id]
                    
                    # If no players left, remove game
                    if not game.players:
                        del self.games[game_id]
                    # If host left, assign new host
                    elif game.players[user_id].is_host:
                        new_host_id = next(iter(game.players.keys()))
                        game.players[new_host_id].is_host = True
                    
                    logger.info(f"User {user_id} left game {game_id}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to leave game: {e}")
            return False

    def is_user_host(self, game_id: str, user_id: str) -> bool:
        """Check if a user is the host of a specific game"""
        try:
            if game_id not in self.games:
                return False
            
            game = self.games[game_id]
            if user_id not in game.players:
                return False
            
            return game.players[user_id].is_host
            
        except Exception as e:
            logger.error(f"Failed to check if user is host: {e}")
            return False
    
    def get_host_player(self, game_id: str) -> Optional[AuthenticatedPlayer]:
        """Get the host player of a specific game"""
        try:
            if game_id not in self.games:
                return None
            
            game = self.games[game_id]
            for player in game.players.values():
                if player.is_host:
                    return player
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get host player: {e}")
            return None

    def can_user_judge(self, game_id: str, user_id: str) -> bool:
        """Check if a user can currently judge the round"""
        try:
            if game_id not in self.games:
                return False
            
            game = self.games[game_id]
            if not game.current_round:
                return False
            
            # User can judge if they are the judge and we're in judging phase
            return (game.current_round.judge_id == user_id and 
                    game.current_round.phase == RoundPhase.JUDGING)
            
        except Exception as e:
            logger.error(f"Failed to check if user can judge: {e}")
            return False

    def get_current_round_status(self, game_id: str) -> Optional[Dict[str, Any]]:
        """Get the current round status for debugging and frontend use"""
        try:
            if game_id not in self.games:
                return None
            
            game = self.games[game_id]
            if not game.current_round:
                return {
                    "status": "no_round",
                    "message": "No active round"
                }
            
            round_info = {
                "round_number": game.current_round.round_number,
                "phase": game.current_round.phase.value,
                "black_card": game.current_round.black_card,
                "judge_id": str(game.current_round.judge_id),
                "judge_username": game.players[game.current_round.judge_id].username if game.current_round.judge_id in game.players else "Unknown",
                "submissions_count": len(game.current_round.submissions),
                "total_players": len(game.players),
                "non_judge_players": len([p for p in game.players.values() if p.user_id != game.current_round.judge_id]),
                "submissions": [
                    {
                        "player_id": str(player_id),
                        "player_username": game.players[player_id].username if player_id in game.players else "Unknown",
                        "card": card
                    }
                    for player_id, card in game.current_round.submissions.items()
                ]
            }
            
            return round_info
            
        except Exception as e:
            logger.error(f"Failed to get current round status: {e}")
            return None

    def debug_judging_phase(self, game_id: str) -> Dict[str, Any]:
        """Debug method to check the judging phase status"""
        try:
            if game_id not in self.games:
                return {"error": "Game not found"}
            
            game = self.games[game_id]
            if not game.current_round:
                return {"error": "No current round"}
            
            debug_info = {
                "game_id": game_id,
                "round_number": game.current_round.round_number,
                "phase": game.current_round.phase.value,
                "black_card": game.current_round.black_card,
                "judge_id": game.current_round.judge_id,
                "judge_username": game.players[game.current_round.judge_id].username if game.current_round.judge_id in game.players else "Unknown",
                "total_players": len(game.players),
                "non_judge_players": len([p for p in game.players.values() if p.user_id != game.current_round.judge_id]),
                "submissions_count": len(game.current_round.submissions),
                "submissions": game.current_round.submissions,
                "submissions_keys": list(game.current_round.submissions.keys()),
                "submissions_key_types": [type(k) for k in game.current_round.submissions.keys()],
                "player_ids": list(game.players.keys()),
                "player_id_types": [type(p.user_id) for p in game.players.values()],
                "phase_transition_ready": len(game.current_round.submissions) == len([p for p in game.players.values() if p.user_id != game.current_round.judge_id])
            }
            
            return debug_info
            
        except Exception as e:
            logger.error(f"Failed to debug judging phase: {e}")
            return {"error": str(e)}

    def debug_game_state(self, game_id: str) -> Dict[str, Any]:
        """Comprehensive debug method to check the entire game state"""
        try:
            if game_id not in self.games:
                return {"error": "Game not found"}
            
            game = self.games[game_id]
            
            # CRITICAL: Check for multiple game manager instances
            logger.info(f"🎯 DEBUG: Game manager instance ID: {id(self)}")
            logger.info(f"🎯 DEBUG: Games dict instance ID: {id(self.games)}")
            logger.info(f"🎯 DEBUG: Game {game_id} instance ID: {id(game)}")
            logger.info(f"🎯 DEBUG: Game {game_id} in self.games: {game_id in self.games}")
            logger.info(f"🎯 DEBUG: Total games in manager: {len(self.games)}")
            logger.info(f"🎯 DEBUG: Game IDs in manager: {list(self.games.keys())}")
            
            debug_info = {
                "game_id": game_id,
                "game_status": game.status.value,
                "total_players": len(game.players),
                "player_details": [],
                "current_round": None,
                "prepared_cards": {
                    "white_cards_count": len(game.prepared_white_cards),
                    "black_cards_count": len(game.prepared_black_cards),
                    "sample_white_cards": game.prepared_white_cards[:5] if game.prepared_white_cards else [],
                    "sample_black_cards": game.prepared_black_cards[:3] if game.prepared_black_cards else []
                },
                "database_sync": {
                    "game_exists": True,
                    "players_in_memory": len(game.players),
                    "player_ids_in_memory": list(game.players.keys()),
                    "player_emails_in_memory": [p.email for p in game.players.values()] if game.players else []
                }
            }
            
            # Add player details
            for player_id, player in game.players.items():
                player_debug = {
                    "player_id": player_id,
                    "user_id": player.user_id,
                    "email": player.email,
                    "username": player.username,
                    "is_host": player.is_host,
                    "is_judge": player.is_judge,
                    "score": player.score,
                    "hand_count": len(player.hand) if player.hand else 0,
                    "sample_hand": player.hand[:3] if player.hand else [],
                    "connected": player.connected,
                    "submitted_card": player.submitted_card
                }
                debug_info["player_details"].append(player_debug)
            
            # Add current round info
            if game.current_round:
                round_debug = {
                    "round_number": game.current_round.round_number,
                    "phase": game.current_round.phase.value,
                    "black_card": game.current_round.black_card,
                    "judge_id": game.current_round.judge_id,
                    "judge_username": game.players[game.current_round.judge_id].username if game.current_round.judge_id in game.players else "Unknown",
                    "submissions_count": len(game.current_round.submissions),
                    "submissions": game.current_round.submissions,
                    "submissions_keys": list(game.current_round.submissions.keys()),
                    "non_judge_players": [p.user_id for p in game.players.values() if p.user_id != game.current_round.judge_id],
                    "phase_transition_ready": len(game.current_round.submissions) == len([p for p in game.players.values() if p.user_id != game.current_round.judge_id])
                }
                debug_info["current_round"] = round_debug
            
            return debug_info
            
        except Exception as e:
            logger.error(f"Failed to debug game state: {e}")
            return {"error": str(e)}

    async def sync_game_with_database(self, db: Session, game_id: str) -> bool:
        """Sync the in-memory game state with the database to ensure consistency"""
        try:
            if game_id not in self.games:
                logger.error(f"Game {game_id} not found in memory")
                return False
            
            game = self.games[game_id]
            
            # Import database models
            try:
                from ..models.database import Game, GamePlayer
            except ImportError:
                from models.database import Game, GamePlayer
            
            # Get game from database
            db_game = db.query(Game).filter(Game.id == game_id).first()
            if not db_game:
                logger.error(f"Game {game_id} not found in database")
                return False
            
            # Get all players from database
            db_players = db.query(GamePlayer).filter(GamePlayer.game_id == game_id).all()
            logger.info(f"Found {len(db_players)} players in database for game {game_id}")
            
            # Sync players
            synced_players = {}
            for db_player in db_players:
                try:
                    # Get user info
                    from ..models.database import User
                    user = db.query(User).filter(User.id == db_player.user_id).first()
                    if user:
                        # Create or update player in memory
                        if db_player.user_id in game.players:
                            # Update existing player
                            existing_player = game.players[db_player.user_id]
                            existing_player.is_host = db_player.is_host
                            existing_player.is_judge = db_player.is_judge
                            existing_player.score = db_player.score
                            existing_player.username = db_player.username
                            synced_players[db_player.user_id] = existing_player
                            logger.info(f"Updated existing player {db_player.user_id} ({user.email})")
                        else:
                            # Create new player
                            new_player = AuthenticatedPlayer(
                                user_id=user.id,
                                email=user.email,
                                username=db_player.username or user.email.split('@')[0],
                                score=db_player.score,
                                is_host=db_player.is_host,
                                is_judge=db_player.is_judge,
                                hand=[],
                                humor_preferences=user.humor_preferences or {}
                            )
                            synced_players[user.id] = new_player
                            logger.info(f"Created new player {user.id} ({user.email})")
                    else:
                        logger.warning(f"User {db_player.user_id} not found in database")
                        
                except Exception as player_error:
                    logger.error(f"Error syncing player {db_player.user_id}: {player_error}")
            
            # Update game players
            if synced_players:
                game.players = synced_players
                logger.info(f"Synced {len(synced_players)} players for game {game_id}")
                
                # If game is in progress but no current round, start one
                if game.status == GameStatus.IN_PROGRESS and not game.current_round:
                    logger.info(f"Game {game_id} is in progress but has no current round, starting new round")
                    await self._start_new_round(game)
                
                return True
            else:
                logger.warning(f"No players synced for game {game_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to sync game with database: {e}")
            return False

    async def force_refresh_game_state(self, db: Session, game_id: str) -> bool:
        """Force refresh the game state by syncing with database and restarting if needed"""
        try:
            logger.info(f"🔄 Force refreshing game state for {game_id}")
            
            # First sync with database
            sync_success = await self.sync_game_with_database(db, game_id)
            if not sync_success:
                logger.error(f"Failed to sync game {game_id} with database")
                return False
            
            if game_id not in self.games:
                logger.error(f"Game {game_id} not found after sync")
                return False
            
            game = self.games[game_id]
            
            # If game is in progress but no current round, start one
            if game.status == GameStatus.IN_PROGRESS and not game.current_round:
                logger.info(f"🔄 Game {game_id} is in progress but has no current round, starting new round")
                await self._start_new_round(game)
                
                # Deal cards to all players
                for player in game.players.values():
                    if not player.hand or len(player.hand) < 5:
                        await self.replenish_player_hand(db, game_id, player.user_id, 5)
                        logger.info(f"🔄 Replenished hand for player {player.user_id}")
            
            # CRITICAL: DO NOT auto-start games - let the judge decide when to start
            # elif game.status == GameStatus.WAITING and len(game.players) >= 2:
            #     logger.info(f"🔄 Game {game_id} has enough players, starting game")
            #     await self.start_game(db, game_id)
            
            logger.info(f"✅ Game state refreshed successfully for {game_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to force refresh game state: {e}")
            return False

    async def _generate_ai_cards_in_background(self, db: Session, game: GameState):
        """Generate AI white cards in the background for better gameplay (non-blocking)"""
        try:
            # DISABLED: This method is causing timeouts and conflicts with our optimized system
            logger.info(f"🚫 Background AI card generation DISABLED for game {game.game_id} - using optimized system instead")
            return
            
            # OLD CODE - DISABLED
            # if game.game_id in self.background_generation_running:
            #     logger.info(f"Background generation already running for game {game.game_id}")
            #     return
            
            # self.background_generation_running[game.game_id] = True
            # logger.info(f"🚀 Starting background AI card generation for game {game.game_id}")
            
            # # Generate personalized cards for each player
            # for player in game.players.values():
            #     try:
            #         # Generate personalized white cards using player's preferences
            #         request = HumorRequest(
            #             context="Generate a short, funny phrase for Cards Against Humanity white card. Make it edgy and humorous, suitable for adult audiences.",
            #             audience="adults",
            #             topic="general",
            #             user_id=str(player.user_id),
            #             humor_type="edgy",
            #             card_type="white"
            #         )
                    
            #         # Generate cards using player's preferences
            #         generated_cards = await self.humor_orchestrator.generate_and_evaluate_humor(request)
            #         if generated_cards and generated_cards.get('success') and generated_cards.get('results'):
            #             # Add generated cards to prepared white cards
            #             for result in generated_cards['results']:
            #                 if result.get('generation') and result['generation'].text:
            #                     cleaned_card = self._validate_and_clean_card(result['generation'].text)
            #                     if cleaned_card and cleaned_card not in game.prepared_white_cards:
            #                         game.prepared_white_cards.append(cleaned_card)
            #             logger.info(f"✅ Generated {len(generated_cards['results'])} personalized cards for player {player.user_id}")
            #         else:
            #             logger.warning(f"⚠️ No generated cards received for player {player.user_id}: {generated_cards}")
                            
            #         # Small delay to avoid overwhelming the system
            #         await asyncio.sleep(0.5)
                    
            #     except Exception as gen_error:
            #         logger.error(f"❌ Failed to generate cards for player {player.user_id}: {gen_error}")
            #         # Continue with other players
            
            # # Shuffle all cards together
            # random.shuffle(game.prepared_white_cards)
            # logger.info(f"🎯 Total prepared white cards after background generation: {len(game.prepared_white_cards)}")
            
            # # Start additional background task to generate more cards
            # asyncio.create_task(self._generate_additional_white_cards(db, game))
            # logger.info(f"🚀 Started additional background task for game {game.game_id}")
            
        except Exception as e:
            logger.error(f"❌ Background AI card generation failed for game {game.game_id}: {e}")
        finally:
            if game.game_id in self.background_generation_running:
                del self.background_generation_running[game.game_id]
    
    async def _generate_additional_white_cards(self, db: Session, game: GameState):
        """Generate additional white cards in the background for better gameplay"""
        try:
            # DISABLED: This method is causing timeouts and conflicts with our optimized system
            logger.info(f"🚫 Additional white card generation DISABLED for game {game.game_id} - using optimized system instead")
            return
            
            # OLD CODE - DISABLED
            # if game.game_id in self.background_generation_running:
            #     logger.info(f"Additional background generation already running for game {game.game_id}")
            #     return
            
            # self.background_generation_running[game.game_id] = True
            
            # # Generate additional cards for each player
            # for player in game.players.values():
            #     try:
            #         # Generate 3 more personalized cards
            #         request = HumorRequest(
            #             context="Cards Against Humanity game - generate additional short, funny phrases for white cards",
            #             audience="friends",
            #             topic="general",
            #             user_id=str(player.user_id),
            #             humor_type="edgy",
            #             card_type="white"
            #         )
                    
            #         generated_cards = await self.humor_orchestrator.generate_and_evaluate_humor(request)
            #         if generated_cards and generated_cards.get('success') and generated_cards.get('results'):
            #             # Add new cards to player's hand
            #             new_cards = []
            #             for result in generated_cards['results']:
            #                 if result.get('generation') and result['generation'].text:
            #                     new_cards.append(result['generation'].text)
                        
            #             player.hand.extend(new_cards)
            #             logger.info(f"Generated {len(new_cards)} additional cards for player {player.user_id}")
                            
            #             # Shuffle the hand
            #             random.shuffle(player.hand)
                            
            #             # Keep hand size manageable (max 20 cards)
            #             if len(player.hand) > 20:
            #                 player.hand = random.sample(player.hand, 20)
                                
            #     except Exception as player_error:
            #         logger.error(f"Failed to generate additional cards for player {player.user_id}: {player_error}")
            
            # logger.info(f"Background card generation completed for game {game.game_id}")
            
        except Exception as e:
            logger.error(f"Background card generation failed for game {game.game_id}: {e}")
        finally:
            if game.game_id in self.background_generation_running:
                del self.background_generation_running[game.game_id]

    async def _generate_white_card(self, db: Session, user_id: str) -> str:
        """Generate a single white card using the personalized humor system"""
        try:
            # DISABLED: This method is causing conflicts with our optimized system
            logger.info(f"🚫 Single white card generation DISABLED for user {user_id} - using optimized system instead")
            return "Something unexpectedly funny"  # Return fallback card
            
            # OLD CODE - DISABLED
            # # Create a humor request for card generation with personalized context
            # request = HumorRequest(
            #     context="Generate a short, funny phrase for Cards Against Humanity white card. Make it edgy and humorous, suitable for adult audiences.",
            #     audience="adults",
            #     topic="general",
            #     user_id=str(user_id),
            #     humor_type="edgy",
            #     card_type="white"
            # )
            
            # # Generate using humor orchestrator with personalized personas
            # result = await self.humor_orchestrator.generate_and_evaluate_humor(request)
            
            # if result and result.get('success') and result.get('results') and result['results']:
            #     generated_card = result['results'][0]['generation'].text
                
            #     # Clean the generated card
            #     generated_card = self._clean_generated_card(generated_card)
                
            #     # Ensure it's appropriate length for a white card
            #     if len(generated_card) > 100:
            #         generated_card = generated_card[:97] + "..."
                
            #     logger.info(f"Generated personalized white card for user {user_id}: {generated_card}")
            #     return generated_card
            # else:
            #     # Fallback to sample cards
            #     fallback_cards = [
            #         "My questionable life choices",
            #         "The awkward silence",
            #         "Unexpected emotional baggage",
            #         "A disappointing revelation",
            #         "My secret shame"
            #     ]
            #     return random.choice(fallback_cards)
                
        except Exception as e:
            logger.warning(f"Error generating white card for user {user_id}: {e}")
            return random.choice([
                "Something unexpectedly funny",
                "A terrible mistake",
                "My hidden talent",
                "An awkward situation",
                "The wrong answer"
            ])

    def _clean_generated_card(self, text: str) -> str:
        """Clean generated card text to remove unwanted characters and formatting"""
        # Remove common prefixes
        prefixes = ['white card:', 'black card:', 'response:', 'answer:', 'card:']
        for prefix in prefixes:
            if text.lower().startswith(prefix):
                text = text[len(prefix):].strip()
        
        # Remove quotes
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
        elif text.startswith("'") and text.endswith("'"):
            text = text[1:-1]
        
        # Remove underscores and other unwanted characters
        text = text.replace('__', '')  # Remove double underscores
        text = text.replace('_', ' ')  # Replace single underscores with spaces
        text = text.replace('...', '')  # Remove ellipsis
        text = text.replace('..', '')   # Remove double dots
        
        # Clean up multiple spaces
        text = ' '.join(text.split())
        
        # Take only the first line
        text = text.split('\n')[0].strip()
        
        # Ensure it's not empty
        if not text or text.isspace():
            return "Something unexpectedly funny"
        
        return text.strip()

    def _validate_and_clean_card(self, card: str) -> str:
        """Validate and clean a card to ensure it's appropriate for the game"""
        # Basic validation
        if not card or len(card.strip()) == 0:
            return "Something unexpected"
        
        # Clean the card
        cleaned = self._clean_generated_card(card)
        
        # Ensure minimum length
        if len(cleaned) < 3:
            return "Something unexpected"
        
        # Ensure maximum length
        if len(cleaned) > 100:
            cleaned = cleaned[:97] + "..."
        
        return cleaned

    async def replenish_player_hand(self, db: Session, game_id: str, user_id: str, min_cards: int = 5):
        """Replenish a player's hand when they run low on cards using pre-generated personalized cards"""
        try:
            if game_id not in self.games:
                return False
            
            game = self.games[game_id]
            if user_id not in game.players:
                return False
            
            player = game.players[user_id]
            
            # If player has enough cards, no need to replenish
            if len(player.hand) >= min_cards:
                return True
            
            # Check if we have pre-generated personalized cards for this player
            if hasattr(game, 'player_personalized_cards') and user_id in game.player_personalized_cards:
                personalized_cards = game.player_personalized_cards[user_id]
                cards_needed = min_cards - len(player.hand)
                
                # Add cards from personalized deck
                cards_added = 0
                while cards_added < cards_needed and personalized_cards:
                    card = personalized_cards.pop(0)  # Take from the top
                    if card not in player.hand:
                        player.hand.append(card)
                        cards_added += 1
                
                logger.info(f"Replenished {cards_added} personalized cards for player {user_id}")
                
                # If we still need more cards, add fallback cards
                while len(player.hand) < min_cards:
                    fallback_card = f"{player.username}'s hidden talent #{len(player.hand) + 1}"
                    if fallback_card not in player.hand:
                        player.hand.append(fallback_card)
                
                random.shuffle(player.hand)
                return True
                
            else:
                # Fallback to default cards if no personalized cards available
                cards_needed = min_cards - len(player.hand)
                default_cards = self._get_default_white_cards()[:cards_needed]
                player.hand.extend(default_cards)
                random.shuffle(player.hand)
                logger.info(f"Replenished {len(default_cards)} default cards for player {user_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to replenish hand for player {user_id}: {e}")
            return False

    def get_player_hand(self, game_id: str, user_id: str) -> List[str]:
        """Get the current hand for a specific player"""
        try:
            if game_id not in self.games:
                return []
            
            game = self.games[game_id]
            if user_id not in game.players:
                return []
            
            player = game.players[user_id]
            return player.hand.copy()  # Return a copy to prevent external modification
            
        except Exception as e:
            logger.error(f"Failed to get hand for player {user_id}: {e}")
            return []

    def get_available_cards_for_player(self, game_id: str, user_id: str) -> List[str]:
        """Get available cards for a player (excluding already submitted ones)"""
        try:
            if game_id not in self.games:
                return []
            
            game = self.games[game_id]
            if user_id not in game.players:
                return []
            
            player = game.players[user_id]
            
            # If there's a current round and player has submitted, return empty
            if game.current_round and user_id in game.current_round.submissions:
                return []
            
            # Return available cards
            return player.hand.copy()
            
        except Exception as e:
            logger.error(f"Failed to get available cards for player {user_id}: {e}")
            return []

    def get_debug_game_info(self, game_id: str) -> Dict[str, Any]:
        """Get detailed debug information about a game for troubleshooting"""
        try:
            if game_id not in self.games:
                return {"error": "Game not found"}
            
            game = self.games[game_id]
            
            debug_info = {
                "game_id": game_id,
                "status": game.status.value,
                "player_count": len(game.players),
                "total_prepared_white_cards": len(game.prepared_white_cards),
                "total_prepared_black_cards": len(game.prepared_black_cards),
                "sample_white_cards": game.prepared_white_cards[:10] if game.prepared_white_cards else [],
                "sample_black_cards": game.prepared_black_cards[:5] if game.prepared_black_cards else [],
                "players": []
            }
            
            for player in game.players.values():
                player_debug = {
                    "user_id": player.user_id,
                    "username": player.username,
                    "hand_count": len(player.hand),
                    "sample_hand": player.hand[:5] if player.hand else [],
                    "is_host": player.is_host,
                    "is_judge": player.is_judge,
                    "score": player.score
                }
                debug_info["players"].append(player_debug)
            
            return debug_info
            
        except Exception as e:
            logger.error(f"Failed to get debug game info: {e}")
            return {"error": str(e)}

    def get_submissions_for_judging(self, game_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Get submissions specifically formatted for judging (only for judges)"""
        try:
            if game_id not in self.games:
                return None
            
            game = self.games[game_id]
            if not game or not game.current_round:
                return None
            
            # Only judges can see submissions for judging
            if game.current_round.judge_id != user_id:
                return None
            
            # Allow judges to see submissions in both card_submission and judging phases
            if game.current_round.phase not in [RoundPhase.CARD_SUBMISSION, RoundPhase.JUDGING]:
                return None
            
            submissions_data = {
                "round_number": game.current_round.round_number,
                "black_card": game.current_round.black_card,
                "submissions": [
                    {
                        "card": card,
                        "player_username": game.players[player_id].username if player_id in game.players else "Unknown",
                        "player_id": str(player_id)
                    }
                    for player_id, card in game.current_round.submissions.items()
                ],
                "total_submissions": len(game.current_round.submissions),
                "phase": game.current_round.phase.value,
                "can_judge": game.current_round.phase == RoundPhase.JUDGING,
                "waiting_for": len([p for p in game.players.values() if p.user_id != game.current_round.judge_id]) - len(game.current_round.submissions)
            }
            
            return submissions_data
            
        except Exception as e:
            logger.error(f"Failed to get submissions for judging: {e}")
            return None

    def get_submissions_status_for_judge(self, game_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Get submissions status for judges during card submission phase"""
        try:
            if game_id not in self.games:
                return None
            
            game = self.games[game_id]
            if not game or not game.current_round:
                return None
            
            # Only judges can see submissions status
            if game.current_round.judge_id != user_id:
                return None
            
            # Only show in card submission phase
            if game.current_round.phase != RoundPhase.CARD_SUBMISSION:
                return None
            
            non_judge_players = [p for p in game.players.values() if p.user_id != game.current_round.judge_id]
            submitted_players = list(game.current_round.submissions.keys())
            waiting_players = [p.user_id for p in non_judge_players if p.user_id not in submitted_players]
            
            status_data = {
                "round_number": game.current_round.round_number,
                "black_card": game.current_round.black_card,
                "phase": game.current_round.phase.value,
                "total_players": len(game.players),
                "non_judge_players": len(non_judge_players),
                "submissions_received": len(submitted_players),
                "waiting_for": len(waiting_players),
                "submissions": [
                    {
                        "card": card,
                        "player_username": game.players[player_id].username if player_id in game.players else "Unknown",
                        "player_id": str(player_id)
                    }
                    for player_id, card in game.current_round.submissions.items()
                ],
                "waiting_players": [
                    {
                        "player_id": str(player_id),
                        "username": game.players[player_id].username if player_id in game.players else "Unknown"
                    }
                    for player_id in waiting_players
                ]
            }
            
            return status_data
            
        except Exception as e:
            logger.error(f"Failed to get submissions status for judge: {e}")
            return None

    def get_game_state_for_user(self, game_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Get game state formatted for a specific user (hiding other players' hands)"""
        try:
            if game_id not in self.games:
                logger.warning(f"Game {game_id} not found in get_game_state_for_user")
                return None
            
            game = self.games[game_id]
            if not game:
                logger.error(f"Game {game_id} is None in get_game_state_for_user")
                return None
                
            if not game.players:
                logger.error(f"Game {game_id} has no players in get_game_state_for_user")
                return None
            
            # CRITICAL: Log the actual in-memory state to verify data integrity
            logger.info(f"🎯 get_game_state_for_user: In-memory game has {len(game.players)} players")
            logger.info(f"🎯 get_game_state_for_user: In-memory player IDs: {list(game.players.keys())}")
            logger.info(f"🎯 get_game_state_for_user: Game instance ID: {id(game)}")
            logger.info(f"🎯 get_game_state_for_user: Game ID value: {game.game_id}")
            
            if game.current_round:
                logger.info(f"🎯 get_game_state_for_user: In-memory round has {len(game.current_round.submissions)} submissions")
                logger.info(f"🎯 get_game_state_for_user: In-memory submissions: {game.current_round.submissions}")
                logger.info(f"🎯 get_game_state_for_user: Current round instance ID: {id(game.current_round)}")
                logger.info(f"🎯 get_game_state_for_user: Submissions dict instance ID: {id(game.current_round.submissions)}")
                logger.info(f"🎯 get_game_state_for_user: Round phase: {game.current_round.phase.value}")
                logger.info(f"🎯 get_game_state_for_user: Judge ID: {game.current_round.judge_id}")
            else:
                logger.info(f"🎯 get_game_state_for_user: In-memory game has no current round")
            
            # CRITICAL: Verify data integrity - if we have 0 players or submissions, something is wrong
            if len(game.players) == 0:
                logger.error(f"🚨 CRITICAL: In-memory game has 0 players! This indicates a serious corruption!")
                logger.error(f"🚨 Game ID: {game_id}")
                logger.error(f"🚨 User ID: {user_id}")
                return None
                
            if game.current_round and len(game.current_round.submissions) == 0:
                logger.warning(f"⚠️ In-memory game has 0 submissions - this might be normal for new rounds")
            elif game.current_round and len(game.current_round.submissions) > 0:
                logger.info(f"✅ In-memory game has {len(game.current_round.submissions)} submissions - data looks good")
            
            # Convert game state to dict
            game_dict = {
                "game_id": game.game_id,
                "status": game.status.value,
                "max_score": game.max_score,
                "max_players": getattr(game, 'max_players', 6),
                "created_at": game.created_at.isoformat() if game.created_at else None,
                "players": [],
                "current_round": None,
                "round_history": [],
                "debug_info": {
                    "total_prepared_cards": len(game.prepared_white_cards),
                    "total_prepared_black_cards": len(game.prepared_black_cards),
                    "sample_white_cards": game.prepared_white_cards[:5] if game.prepared_white_cards else [],
                    "sample_black_cards": game.prepared_black_cards[:3] if game.prepared_black_cards else []
                }
            }
            
            # Add my_hand and is_my_turn for the current user
            current_player = game.players.get(user_id)
            if current_player:
                game_dict["my_hand"] = current_player.hand.copy()
                
                # Add persona information for current user's cards
                game_dict["my_hand_with_personas"] = []
                if user_id in game.current_round_cards:
                    round_cards = game.current_round_cards[user_id]
                    for card_text in current_player.hand:
                        # Find matching PersonalizedCard
                        card_with_persona = {
                            "text": card_text,
                            "persona_name": "Unknown Persona",
                            "persona_type": "unknown",
                            "is_safe": True
                        }
                        
                        for personalized_card in round_cards:
                            if personalized_card.text == card_text:
                                card_with_persona = {
                                    "text": card_text,
                                    "persona_name": personalized_card.persona_name,
                                    "persona_type": personalized_card.persona_type,
                                    "is_safe": personalized_card.is_safe
                                }
                                break
                        
                        game_dict["my_hand_with_personas"].append(card_with_persona)
                else:
                    # Fallback: just card text without persona info
                    for card_text in current_player.hand:
                        game_dict["my_hand_with_personas"].append({
                            "text": card_text,
                            "persona_name": "Generated Persona",
                            "persona_type": "general",
                            "is_safe": True
                        })
                
                # Determine if it's the current user's turn
                if game.current_round:
                    if game.current_round.phase.value == "card_submission":
                        # In card submission phase: non-judge players can submit cards
                        game_dict["is_my_turn"] = (game.current_round.judge_id != user_id)
                    elif game.current_round.phase.value == "judging":
                        # In judging phase: only the judge can judge
                        game_dict["is_my_turn"] = (game.current_round.judge_id == user_id)
                    else:
                        # In other phases: no one's turn
                        game_dict["is_my_turn"] = False
                else:
                    game_dict["is_my_turn"] = False
                
                logger.info(f"🎯 Player {user_id} ({current_player.username}): is_judge={current_player.is_judge}, is_my_turn={game_dict['is_my_turn']}")
                logger.info(f"🎯 Round phase: {game.current_round.phase.value if game.current_round else 'None'}")
                logger.info(f"🎯 Hand with personas: {len(game_dict['my_hand_with_personas'])} cards")
            
            # Add players (hide hands for other players)
            logger.info(f"🎯 PLAYERS DEBUG: game.players type: {type(game.players)}, content: {game.players}")
            if game.players and isinstance(game.players, dict):
                logger.info(f"🎯 PLAYERS DEBUG: Processing {len(game.players)} players")
                for player_id, player in game.players.items():
                    if not player:
                        logger.warning(f"Skipping None player in game {game_id} at key {player_id}")
                        continue
                        
                    logger.info(f"🎯 PLAYERS DEBUG: Processing player {player_id}: {player.username}")
                    player_dict = {
                        "user_id": str(player.user_id),  # Convert to string for frontend compatibility
                        "email": player.email,
                        "username": player.username,
                        "score": player.score,
                        "is_host": player.is_host,
                        "is_judge": player.is_judge,
                        "connected": player.connected,
                        "submitted_card": player.submitted_card
                    }
                    
                    # Only show hand to the current user
                    if player.user_id == user_id:
                        player_dict["hand"] = player.hand.copy() if player.hand else []
                        player_dict["my_hand"] = player.hand.copy() if player.hand else []  # Frontend expects this
                        player_dict["hand_count"] = len(player.hand) if player.hand else 0
                    else:
                        player_dict["hand"] = []  # Hide other players' hands
                        player_dict["my_hand"] = []
                        player_dict["hand_count"] = len(player.hand) if player.hand else 0  # Show count but not cards
                    
                    game_dict["players"].append(player_dict)
                    logger.info(f"✅ Added player {player_id} to game_dict")
                
                logger.info(f"🎯 FINAL PLAYERS: {len(game_dict['players'])} players added to response")
            else:
                logger.error(f"Game {game_id} players is not a valid dict: {type(game.players)}, value: {game.players}")
                game_dict["players"] = []
            
            # Add current round info
            if game.current_round:
                # Get judge username
                judge_username = "Unknown"
                if game.current_round.judge_id in game.players:
                    judge_username = game.players[game.current_round.judge_id].username
                
                round_dict = {
                    "round_number": game.current_round.round_number,
                    "black_card": game.current_round.black_card,
                    "judge_id": str(game.current_round.judge_id),  # Convert to string for frontend compatibility
                    "judge_username": judge_username,  # Frontend needs this
                    "phase": game.current_round.phase.value,
                    "submissions_count": len(game.current_round.submissions),
                    "winner_id": str(game.current_round.winner_id) if game.current_round.winner_id else None,
                    "winning_card": game.current_round.winning_card,
                    "submissions": [
                        {
                            "card": card,
                            "player_username": game.players[player_id].username if player_id in game.players else "Unknown",
                            "player_id": str(player_id)
                        }
                        for player_id, card in game.current_round.submissions.items()
                    ] if (game.current_round.submissions and 
                          (game.current_round.phase.value == "judging" or 
                           game.current_round.phase.value == "results" or
                           (game.current_round.judge_id == user_id and game.current_round.phase.value == "card_submission"))
                         ) else [],  # Judges can see submissions in all phases when they are the judge
                    "can_judge": (game.current_round.phase.value == "judging" and 
                                 game.current_round.judge_id == user_id),
                    "submissions_visible": (game.current_round.phase.value == "judging" or 
                                          game.current_round.phase.value == "results" or
                                          (game.current_round.judge_id == user_id and game.current_round.phase.value == "card_submission")),
                    # ENHANCED: Add judge-specific visibility for card submission phase
                    "judge_can_see_submissions": (game.current_round.judge_id == user_id and 
                                                game.current_round.phase.value in ["card_submission", "judging"]),
                    "waiting_for_players": [
                        {
                            "user_id": str(p.user_id),
                            "username": p.username,
                            "has_submitted": p.user_id in game.current_round.submissions
                        }
                        for p in game.players.values() 
                        if p.user_id != game.current_round.judge_id
                    ] if game.current_round.judge_id == user_id else []
                }
                game_dict["current_round"] = round_dict
                
                # Debug logging for phase transition
                logger.info(f"🎯 get_game_state_for_user: Round phase = {game.current_round.phase.value}")
                logger.info(f"🎯 get_game_state_for_user: Submissions count = {len(game.current_round.submissions)}")
                logger.info(f"🎯 get_game_state_for_user: Submissions = {game.current_round.submissions}")
                logger.info(f"🎯 get_game_state_for_user: Submissions keys = {list(game.current_round.submissions.keys())}")
                logger.info(f"🎯 get_game_state_for_user: Submissions key types = {[type(k) for k in game.current_round.submissions.keys()]}")
                logger.info(f"🎯 get_game_state_for_user: Player IDs in game.players = {list(game.players.keys())}")
                logger.info(f"🎯 get_game_state_for_user: Player ID types in game.players = {[type(k) for k in game.players.keys()]}")
                logger.info(f"🎯 get_game_state_for_user: Formatted submissions array = {round_dict['submissions']}")
                logger.info(f"🎯 get_game_state_for_user: Formatted submissions length = {len(round_dict['submissions'])}")
                
                # Additional debugging for submissions formatting
                try:
                    for player_id, card in game.current_round.submissions.items():
                        logger.info(f"🎯 Processing submission - player_id: {player_id} (type: {type(player_id)}), card: {card}")
                        if player_id in game.players:
                            logger.info(f"🎯 Player {player_id} found in game.players: {game.players[player_id].username}")
                        else:
                            logger.warning(f"⚠️ Player {player_id} NOT found in game.players!")
                            logger.warning(f"⚠️ Available player IDs: {list(game.players.keys())}")
                except Exception as sub_error:
                    logger.error(f"❌ Error processing submissions: {sub_error}")
            
            # Add round history
            for round_data in game.round_history:
                history_round = {
                    "round_number": round_data.round_number,
                    "black_card": round_data.black_card,
                    "judge_id": str(round_data.judge_id),  # Convert to string for frontend compatibility
                    "winner_id": str(round_data.winner_id) if round_data.winner_id else None,
                    "winning_card": round_data.winning_card
                }
                game_dict["round_history"].append(history_round)
            
            # FINAL VERIFICATION: Ensure we're returning the correct data
            logger.info(f"🎯 get_game_state_for_user: FINAL VERIFICATION")
            logger.info(f"🎯 get_game_state_for_user: Returning {len(game_dict.get('players', []))} players")
            if 'current_round' in game_dict and game_dict['current_round']:
                logger.info(f"🎯 get_game_state_for_user: Returning {len(game_dict['current_round'].get('submissions', []))} submissions")
                logger.info(f"🎯 get_game_state_for_user: Submissions data: {game_dict['current_round'].get('submissions', [])}")
                
                # Additional verification for judge submissions
                if game_dict['current_round'].get('judge_id') == str(user_id):
                    logger.info(f"🎯 JUDGE VERIFICATION: User {user_id} is judge")
                    logger.info(f"🎯 JUDGE VERIFICATION: Phase: {game_dict['current_round'].get('phase')}")
                    logger.info(f"🎯 JUDGE VERIFICATION: Submissions visible: {game_dict['current_round'].get('submissions_visible')}")
                    logger.info(f"🎯 JUDGE VERIFICATION: Judge can see submissions: {game_dict['current_round'].get('judge_can_see_submissions')}")
                    logger.info(f"🎯 JUDGE VERIFICATION: Submissions count: {len(game_dict['current_round'].get('submissions', []))}")
                else:
                    logger.info(f"🎯 USER VERIFICATION: User {user_id} is NOT judge (judge_id: {game_dict['current_round'].get('judge_id')})")
            
            return game_dict
            
        except Exception as e:
            logger.error(f"Failed to get game state for user {user_id}: {e}")
            return None

    async def test_ai_generation(self, db: Session, player: AuthenticatedPlayer):
        """Test method to debug AI card generation"""
        try:
            logger.info(f"🧪 TESTING AI Generation for {player.username}")
            
            # Test single generation call
            request = HumorRequest(
                context="Generate a short, funny phrase for Cards Against Humanity white card. Make it edgy and humorous, suitable for adult audiences.",
                audience="adults",
                topic="general",
                user_id=str(player.user_id),
                humor_type="edgy",
                card_type="white"
            )
            
            logger.info(f"🧪 Sending request: {request}")
            generated_cards = await self.humor_orchestrator.generate_and_evaluate_humor(request)
            
            logger.info(f"🧪 Raw response: {generated_cards}")
            logger.info(f"🧪 Response type: {type(generated_cards)}")
            logger.info(f"🧪 Response length: {len(generated_cards) if isinstance(generated_cards, list) else 'N/A'}")
            
            if generated_cards and isinstance(generated_cards, list):
                for i, card_result in enumerate(generated_cards):
                    logger.info(f"🧪 Card {i+1}: {card_result}")
                    logger.info(f"🧪 Card {i+1} type: {type(card_result)}")
                    logger.info(f"🧪 Card {i+1} has text: {hasattr(card_result, 'text')}")
                    if hasattr(card_result, 'text'):
                        logger.info(f"🧪 Card {i+1} text: '{card_result.text}'")
                        cleaned = self._validate_and_clean_card(card_result.text)
                        logger.info(f"🧪 Card {i+1} cleaned: '{cleaned}'")
            
            return generated_cards
            
        except Exception as e:
            logger.error(f"🧪 AI Generation test failed: {e}")
            return None
