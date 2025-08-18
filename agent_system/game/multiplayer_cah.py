#!/usr/bin/env python3
"""
Multiplayer Cards Against Humanity Game System
Handles game logic, rounds, card preparation, and player management
"""

import asyncio
import json
import random
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

# Import humor generation system
from agent_system.agents.improved_humor_agents import ImprovedHumorOrchestrator, HumorRequest
from agent_system.personas.persona_manager import PersonaManager as FixedPersonaManager

logger = logging.getLogger(__name__)

class GameStatus(Enum):
    WAITING = "waiting"
    STARTING = "starting" 
    IN_PROGRESS = "in_progress"
    JUDGING = "judging"
    ROUND_COMPLETE = "round_complete"
    FINISHED = "finished"

class RoundPhase(Enum):
    CARD_SUBMISSION = "card_submission"
    JUDGING = "judging"
    RESULTS = "results"

@dataclass
class Player:
    user_id: str
    username: str
    score: int = 0
    is_host: bool = False
    is_judge: bool = False
    hand: List[str] = None
    submitted_card: Optional[str] = None
    connected: bool = True
    
    def __post_init__(self):
        if self.hand is None:
            self.hand = []

@dataclass
class GameRound:
    round_number: int
    black_card: str
    judge_id: str
    submissions: Dict[str, str]  # player_id -> white_card
    winner_id: Optional[str] = None
    winning_card: Optional[str] = None
    phase: RoundPhase = RoundPhase.CARD_SUBMISSION
    start_time: datetime = None
    
    def __post_init__(self):
        if self.start_time is None:
            self.start_time = datetime.now()

@dataclass
class GameState:
    game_id: str
    players: Dict[str, Player]
    status: GameStatus
    current_round: Optional[GameRound]
    round_history: List[GameRound]
    settings: Dict[str, Any]
    created_at: datetime
    max_score: int = 5
    max_players: int = 6
    prepared_white_cards: List[str] = None
    prepared_black_cards: List[str] = None
    
    def __post_init__(self):
        if self.prepared_white_cards is None:
            self.prepared_white_cards = []
        if self.prepared_black_cards is None:
            self.prepared_black_cards = []

class MultiplayerCAHGame:
    """Multiplayer Cards Against Humanity game manager"""
    
    def __init__(self, humor_orchestrator: ImprovedHumorOrchestrator, persona_manager: FixedPersonaManager):
        self.games: Dict[str, GameState] = {}
        self.humor_orchestrator = humor_orchestrator
        self.persona_manager = persona_manager
        self.background_generation_running: Dict[str, bool] = {}  # Track background generation per game
        
        # Default black cards for quick start
        self.default_black_cards = [
            "What's the secret to a good relationship? _____",
            "What would grandma find disturbing? _____", 
            "What's the next Happy Meal toy? _____",
            "What did I bring back from Mexico? _____",
            "What's worse than finding a worm in your apple? _____",
            "What ended my last relationship? _____",
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
            "What I wish I knew earlier", "My most embarrassing moment", "The thing I can't stop thinking about"
            "What's my secret power? _____",
            "What will I bring back in time to convince people that I am a powerful wizard? _____",
            "What's the most emo? _____",
            "What gives me uncontrollable gas? _____",
            "What would complete my breakfast? _____",
            "What's the new fad diet? _____",
            "What's that sound? _____",
            "What helps Obama unwind? _____",
            "What never fails to liven up the party? _____"
        ]
    
    async def create_game(self, game_id: str, host_user_id: str, host_username: str, 
                         settings: Dict[str, Any] = None) -> GameState:
        """Create a new multiplayer game"""
        
        if settings is None:
            settings = {
                "max_score": 5,
                "max_players": 6,
                "round_timer": 300,  # 5 minutes per round
                "cards_per_player": 7,
                "allow_spectators": False
            }
        
        # Create host player
        host_player = Player(
            user_id=host_user_id,
            username=host_username,
            is_host=True,
            connected=True
        )
        
        # Create game state
        game_state = GameState(
            game_id=game_id,
            players={host_user_id: host_player},
            status=GameStatus.WAITING,
            current_round=None,
            round_history=[],
            settings=settings,
            created_at=datetime.now(),
            max_score=settings.get("max_score", 5),
            max_players=settings.get("max_players", 6)
        )
        
        self.games[game_id] = game_state
        
        # Pre-generate cards for this game
        await self._prepare_cards_for_game(game_id)
        
        logger.info(f"Created game {game_id} with host {host_username}")
        return game_state
    
    async def join_game(self, game_id: str, user_id: str, username: str) -> GameState:
        """Join an existing game"""
        
        if game_id not in self.games:
            raise ValueError(f"Game {game_id} not found")
        
        game = self.games[game_id]
        
        if game.status not in [GameStatus.WAITING, GameStatus.STARTING]:
            raise ValueError("Cannot join game in progress")
        
        if len(game.players) >= game.max_players:
            raise ValueError("Game is full")
        
        if user_id in game.players:
            # Player reconnecting
            game.players[user_id].connected = True
            logger.info(f"Player {username} reconnected to game {game_id}")
        else:
            # New player joining
            player = Player(
                user_id=user_id,
                username=username,
                connected=True
            )
            game.players[user_id] = player
            logger.info(f"Player {username} joined game {game_id}")
        
        return game
    
    async def start_game(self, game_id: str, host_user_id: str) -> GameState:
        """Start the game (host only)"""
        
        if game_id not in self.games:
            raise ValueError(f"Game {game_id} not found")
        
        game = self.games[game_id]
        
        if not game.players[host_user_id].is_host:
            raise ValueError("Only host can start the game")
        
        if len(game.players) < 2:
            raise ValueError("Need at least 2 players to start")
        
        if game.status != GameStatus.WAITING:
            raise ValueError("Game already started")
        
        # Deal initial hands to all players
        await self._deal_initial_hands(game)
        
        # Start first round
        game.status = GameStatus.IN_PROGRESS
        await self._start_new_round(game)
        
        logger.info(f"Game {game_id} started with {len(game.players)} players")
        return game
    
    async def _deal_initial_hands(self, game: GameState):
        """Deal initial white cards to all players"""
        cards_per_player = game.settings.get("cards_per_player", 7)
        
        for player in game.players.values():
            player.hand = []
            for _ in range(cards_per_player):
                if game.prepared_white_cards:
                    card = game.prepared_white_cards.pop(0)
                    # FIXED: Validate and clean the card before dealing
                    card = self._validate_and_clean_card(card)
                    player.hand.append(card)
                else:
                    # Generate new card if we run out
                    new_card = await self._generate_white_card(game.game_id, player.user_id)
                    player.hand.append(new_card)
    
    async def _start_new_round(self, game: GameState):
        """Start a new round"""
        round_number = len(game.round_history) + 1
        
        # Select judge (rotate through players)
        player_ids = list(game.players.keys())
        judge_index = (round_number - 1) % len(player_ids)
        judge_id = player_ids[judge_index]
        
        # Update judge status
        for player in game.players.values():
            player.is_judge = (player.user_id == judge_id)
            player.submitted_card = None
        
        # Select black card
        if game.prepared_black_cards:
            black_card = game.prepared_black_cards.pop(0)
        else:
            black_card = random.choice(self.default_black_cards)
        
        # Create new round
        game.current_round = GameRound(
            round_number=round_number,
            black_card=black_card,
            judge_id=judge_id,
            submissions={},
            phase=RoundPhase.CARD_SUBMISSION
        )
        
        logger.info(f"Started round {round_number} in game {game.game_id}, judge: {game.players[judge_id].username}")
    
    async def submit_card(self, game_id: str, user_id: str, white_card: str) -> GameState:
        """Submit a white card for the current round"""
        
        logger.info(f"🎯 submit_card called: user={user_id}, game={game_id}, card='{white_card}'")
        
        if game_id not in self.games:
            raise ValueError(f"Game {game_id} not found")
        
        game = self.games[game_id]
        
        if game.status != GameStatus.IN_PROGRESS:
            raise ValueError("Game is not in progress")
        
        if not game.current_round:
            raise ValueError("No active round")
        
        if game.current_round.phase != RoundPhase.CARD_SUBMISSION:
            raise ValueError("Card submission phase is over")
        
        if user_id not in game.players:
            raise ValueError("Player not in game")
        
        player = game.players[user_id]
        
        if player.is_judge:
            raise ValueError("Judge cannot submit cards")
        
        # Check if player has already submitted
        if player.submitted_card is not None:
            raise ValueError("Player has already submitted a card")
        
        # FIXED: More robust card validation - normalize whitespace and handle encoding
        normalized_submitted = white_card.strip()
        card_found = False
        actual_card = None
        
        # Try exact match first
        if normalized_submitted in player.hand:
            card_found = True
            actual_card = normalized_submitted
        else:
            # Try case-insensitive match
            for card in player.hand:
                if card.strip().lower() == normalized_submitted.lower():
                    card_found = True
                    actual_card = card
                    break
        
        if not card_found:
            logger.warning(f"Card '{white_card}' not found in player {user_id}'s hand. Available cards: {player.hand}")
            raise ValueError("Card not in player's hand")
        
        # Remove the actual card from hand and add to submissions
        player.hand.remove(actual_card)
        player.submitted_card = actual_card
        game.current_round.submissions[user_id] = actual_card
        
        logger.info(f"✅ Player {user_id} ({player.username}) submitted card: {actual_card}")
        logger.info(f"📊 Submissions now: {len(game.current_round.submissions)} total")
        
        # Deal new card to player
        await self._deal_card_to_player(game, player)
        
        # Check if all players have submitted
        non_judge_players = [p for p in game.players.values() if not p.is_judge and p.connected]
        logger.info(f"👥 Non-judge players: {len(non_judge_players)}, Submissions: {len(game.current_round.submissions)}")
        
        if len(game.current_round.submissions) == len(non_judge_players):
            game.current_round.phase = RoundPhase.JUDGING
            logger.info(f"🎯 All cards submitted for round {game.current_round.round_number} in game {game_id} - moving to judging phase")
        
        return game
    
    def _validate_and_clean_card(self, card_text: str) -> str:
        """Validate and clean a card text to ensure it's suitable for the game"""
        if not card_text or card_text.isspace():
            return "Something unexpectedly funny"
        
        # Clean the card using the same method as generation
        cleaned = self._clean_generated_card(card_text)
        
        # Additional validation
        if len(cleaned) < 3:
            return "Something unexpectedly funny"
        
        if len(cleaned) > 150:
            cleaned = cleaned[:147] + "..."
        
        return cleaned
    
    async def _deal_card_to_player(self, game: GameState, player: Player):
        """Deal a new white card to a player"""
        try:
            if game.prepared_white_cards:
                new_card = game.prepared_white_cards.pop(0)
                # FIXED: Validate and clean the card before dealing
                new_card = self._validate_and_clean_card(new_card)
                logger.info(f"Dealt prepared card to {player.username}: {new_card}")
            else:
                # Only generate new cards if we don't have prepared ones
                new_card = await self._generate_white_card(game.game_id, player.user_id)
                logger.info(f"Generated and dealt new card to {player.username}: {new_card}")
            
            if new_card:
                player.hand.append(new_card)
                logger.info(f"Player {player.username} now has {len(player.hand)} cards")
            else:
                logger.warning(f"Failed to deal card to {player.username}")
                
        except Exception as e:
            logger.error(f"Error dealing card to {player.username}: {e}")
            # Add a default card as fallback
            fallback_card = "A disappointing birthday party"
            player.hand.append(fallback_card)
            logger.info(f"Added fallback card to {player.username}: {fallback_card}")
    
    async def judge_round(self, game_id: str, judge_user_id: str, winning_card: str) -> GameState:
        """Judge the round and select winner"""
        
        if game_id not in self.games:
            raise ValueError(f"Game {game_id} not found")
        
        game = self.games[game_id]
        
        if not game.current_round:
            raise ValueError("No active round")
        
        if game.current_round.phase != RoundPhase.JUDGING:
            raise ValueError("Not in judging phase")
        
        if game.current_round.judge_id != judge_user_id:
            raise ValueError("Only the judge can select the winner")
        
        if winning_card not in game.current_round.submissions.values():
            raise ValueError("Invalid winning card")
        
        # Find winner
        winner_id = None
        for player_id, card in game.current_round.submissions.items():
            if card == winning_card:
                winner_id = player_id
                break
        
        if not winner_id:
            raise ValueError("Could not find winner")
        
        # Award point and update round
        game.players[winner_id].score += 1
        game.current_round.winner_id = winner_id
        game.current_round.winning_card = winning_card
        game.current_round.phase = RoundPhase.RESULTS
        
        # Check win condition
        if game.players[winner_id].score >= game.max_score:
            game.status = GameStatus.FINISHED
            logger.info(f"Game {game_id} finished! Winner: {game.players[winner_id].username}")
        else:
            # Move to next round after brief delay
            game.round_history.append(game.current_round)
            await asyncio.sleep(3)  # Show results for 3 seconds
            await self._start_new_round(game)
        
        logger.info(f"Round {game.current_round.round_number} won by {game.players[winner_id].username} in game {game_id}")
        return game
    
    async def _prepare_cards_for_game(self, game_id: str):
        """Pre-generate cards for the game, using fallbacks for speed"""
        try:
            game = self.games[game_id]
            game.prepared_black_cards = self.default_black_cards.copy()
            random.shuffle(game.prepared_black_cards)

            # Use default white cards for quick start, but clean them
            game.prepared_white_cards = []
            for card in self.default_white_cards:
                cleaned_card = self._validate_and_clean_card(card)
                game.prepared_white_cards.append(cleaned_card)
            random.shuffle(game.prepared_white_cards)
            logger.info(f"Initially prepared {len(game.prepared_white_cards)} cleaned default white cards for game {game_id}")

            # Start background task to generate more AI cards
            asyncio.create_task(self._generate_additional_white_cards(game_id))
            logger.info(f"Started background task to generate additional AI white cards for game {game_id}")

        except Exception as e:
            logger.error(f"Error preparing cards for game {game_id}: {e}")
            # Fallback to ensure game can still start
            game.prepared_white_cards = []
            for card in self.default_white_cards:
                cleaned_card = self._validate_and_clean_card(card)
                game.prepared_white_cards.append(cleaned_card)
            random.shuffle(game.prepared_white_cards)
    
    async def _generate_white_card(self, game_id: str, user_id: str) -> str:
        """Generate a single white card using the personalized humor system"""
        try:
            # Create a humor request for card generation with personalized context
            request = HumorRequest(
                context="Generate a short, funny phrase for Cards Against Humanity white card. Make it edgy and humorous, suitable for adult audiences.",
                audience="adults",
                topic="general",
                user_id=user_id,
                humor_type="edgy"
            )
            
            # Generate using humor orchestrator with personalized personas
            result = await self.humor_orchestrator.generate_and_evaluate_humor(request, num_generators=1)
            
            if result.get('success') and result.get('top_results'):
                generated_card = result['top_results'][0]['generation']['text']
                
                # FIXED: Enhanced cleaning to remove __ and other unwanted characters
                generated_card = self._clean_generated_card(generated_card)
                
                # Ensure it's appropriate length for a white card
                if len(generated_card) > 100:
                    generated_card = generated_card[:97] + "..."
                
                logger.info(f"Generated personalized white card for user {user_id}: {generated_card}")
                return generated_card
            else:
                # Fallback to sample cards
                fallback_cards = [
                    "My questionable life choices",
                    "The awkward silence",
                    "Unexpected emotional baggage",
                    "A disappointing revelation",
                    "My secret shame"
                ]
                return random.choice(fallback_cards)
                
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
        
        # FIXED: Remove underscores and other unwanted characters
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
        
        return text
    
    async def _generate_additional_white_cards(self, game_id: str):
        """Generate additional personalized white cards in the background"""
        try:
            game = self.games[game_id]
            if not game:
                return
            
            # Set flag to indicate background generation is running
            self.background_generation_running[game_id] = True
            
            logger.info(f"Starting background generation of personalized white cards for game {game_id}")
            
            # Create a copy of players to avoid dictionary modification during iteration
            players = list(game.players.values())
            
            # Generate personalized cards for each player
            for player in players:
                try:
                    # Check if game is in active gameplay - if so, be more conservative
                    is_active_gameplay = (game.status == GameStatus.IN_PROGRESS and 
                                        game.current_round and 
                                        game.current_round.phase == RoundPhase.CARD_SUBMISSION)
                    
                    # Skip generation during active gameplay to avoid interference
                    if is_active_gameplay:
                        logger.info(f"Skipping background generation for {player.username} during active gameplay")
                        continue
                    
                    # Generate 3 personalized cards per player
                    for _ in range(3):
                        personalized_card = await self._generate_white_card(game_id, player.user_id)
                        if personalized_card and personalized_card not in game.prepared_white_cards:
                            # FIXED: Validate and clean the card before adding
                            cleaned_card = self._validate_and_clean_card(personalized_card)
                            if cleaned_card and cleaned_card not in game.prepared_white_cards:
                                game.prepared_white_cards.append(cleaned_card)
                                logger.info(f"Added personalized card for {player.username}: {cleaned_card}")
                    
                    # Small delay to avoid overwhelming the system
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    logger.warning(f"Error generating personalized cards for {player.username}: {e}")
            
            logger.info(f"Completed background generation for game {game_id}. Total cards: {len(game.prepared_white_cards)}")
            
        except Exception as e:
            logger.error(f"Error in background card generation for game {game_id}: {e}")
        finally:
            # Clear the flag when done
            self.background_generation_running[game_id] = False
    
    def get_game_state(self, game_id: str) -> Optional[GameState]:
        """Get current game state"""
        return self.games.get(game_id)
    
    def get_player_view(self, game_id: str, user_id: str) -> Dict[str, Any]:
        """Get game state from a specific player's perspective"""
        if game_id not in self.games:
            return None
        
        game = self.games[game_id]
        if user_id not in game.players:
            return None
        
        player = game.players[user_id]
        
        # FIXED: Enhanced logging for judge specifically
        is_judge = player.is_judge if game.current_round else False
        if is_judge:
            logger.info(f"🔍 Getting JUDGE view for {user_id} ({player.username})")
            if game.current_round:
                logger.info(f"   Current round: {game.current_round.round_number}")
                logger.info(f"   Submissions: {len(game.current_round.submissions)}")
                for pid, card in game.current_round.submissions.items():
                    submitter_name = game.players[pid].username if pid in game.players else "Unknown"
                    logger.info(f"     {submitter_name}: {card}")
        
        # Create player-specific view
        view = {
            "game_id": game_id,
            "status": game.status.value,
            "players": [
                {
                    "user_id": p.user_id,
                    "username": p.username,
                    "score": p.score,
                    "is_host": p.is_host,
                    "is_judge": p.is_judge,
                    "connected": p.connected,
                    "has_submitted": p.submitted_card is not None if game.current_round else False
                }
                for p in game.players.values()
            ],
            "my_hand": player.hand,
            "is_my_turn": not player.is_judge if game.current_round else False,
            "settings": game.settings
        }
        
        # Add current round info
        if game.current_round:
            view["current_round"] = {
                "round_number": game.current_round.round_number,
                "black_card": game.current_round.black_card,
                "phase": game.current_round.phase.value,
                "judge_id": game.current_round.judge_id,
                "judge_username": game.players[game.current_round.judge_id].username,
                "submissions_count": len(game.current_round.submissions),
                "my_submission": player.submitted_card
            }
            
            # FIXED: Enhanced judge view - show submission status even during card submission phase
            if player.is_judge:
                # Judge can see who has submitted during both phases
                view["current_round"]["player_submission_status"] = [
                    {
                        "player_id": p.user_id,
                        "username": p.username,
                        "has_submitted": p.submitted_card is not None
                    }
                    for p in game.players.values() if not p.is_judge
                ]
                
                # Show actual submissions only during judging phase
                if game.current_round.phase == RoundPhase.JUDGING:
                    view["current_round"]["submissions"] = [
                        {
                            "card": card,
                            "player_username": game.players[player_id].username
                        }
                        for player_id, card in game.current_round.submissions.items()
                    ]
        
        # FIXED: Enhanced logging for judge view
        if is_judge:
            submitted_players = [p for p in view["players"] if p.get("has_submitted", False)]
            logger.info(f"🔍 JUDGE view created: {len(submitted_players)} players have submitted")
            for p in view["players"]:
                if not p["is_judge"]:
                    logger.info(f"   {p['username']}: has_submitted={p.get('has_submitted', False)}")
        
        # Add round history
        view["round_history"] = [
            {
                "round_number": round.round_number,
                "black_card": round.black_card,
                "winner_username": game.players[round.winner_id].username if round.winner_id else None,
                "winning_card": round.winning_card
            }
            for round in game.round_history
        ]
        
        return view
    
    async def leave_game(self, game_id: str, user_id: str):
        """Player leaves the game"""
        if game_id not in self.games:
            return
        
        game = self.games[game_id]
        if user_id in game.players:
            game.players[user_id].connected = False
            
            # If host leaves, transfer to another player
            if game.players[user_id].is_host and len([p for p in game.players.values() if p.connected]) > 1:
                for player in game.players.values():
                    if player.connected and player.user_id != user_id:
                        player.is_host = True
                        break
            
            # End game if not enough players
            connected_players = [p for p in game.players.values() if p.connected]
            if len(connected_players) < 2 and game.status == GameStatus.IN_PROGRESS:
                game.status = GameStatus.FINISHED
    
    def delete_game(self, game_id: str):
        """Delete a game and clean up resources"""
        if game_id in self.games:
            del self.games[game_id]
            # Clean up background generation flag
            if game_id in self.background_generation_running:
                del self.background_generation_running[game_id]
            logger.info(f"Deleted game {game_id}")
    
    def get_all_games(self) -> List[Dict[str, Any]]:
        """Get list of all games (for admin/debugging)"""
        return [
            {
                "game_id": game_id,
                "status": game.status.value,
                "player_count": len(game.players),
                "created_at": game.created_at.isoformat()
            }
            for game_id, game in self.games.items()
        ]

# Game Manager Singleton
game_manager: Optional[MultiplayerCAHGame] = None

def initialize_game_manager(humor_orchestrator: ImprovedHumorOrchestrator, 
                          persona_manager: FixedPersonaManager):
    """Initialize the global game manager"""
    global game_manager
    game_manager = MultiplayerCAHGame(humor_orchestrator, persona_manager)
    return game_manager

def get_game_manager() -> MultiplayerCAHGame:
    """Get the global game manager"""
    if game_manager is None:
        raise RuntimeError("Game manager not initialized")
    return game_manager 