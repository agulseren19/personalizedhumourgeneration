from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import json

# Module load test - this should appear in logs when module is imported
print(f"🎯 MULTIPLAYER_ROUTES MODULE LOADED AT {__import__('datetime').datetime.now().strftime('%H:%M:%S')}")
from sqlalchemy.orm import Session
from typing import Dict, List, Any, Optional
import uuid
import logging

try:
    from .auth_routes import get_current_user
    from ..models.database import User, get_db
    from ..game.authenticated_multiplayer_cah import AuthenticatedMultiplayerCAHGame, GameState
    from ..agents.improved_humor_agents import ImprovedHumorOrchestrator
    from ..personas.persona_manager import PersonaManager
except ImportError:
    # Fallback to absolute imports
    from agent_system.api.auth_routes import get_current_user
    from agent_system.models.database import User, get_db
    from agent_system.game.authenticated_multiplayer_cah import AuthenticatedMultiplayerCAHGame, GameState
    from agent_system.agents.improved_humor_agents import ImprovedHumorOrchestrator
    from agent_system.personas.persona_manager import PersonaManager

router = APIRouter(prefix="/multiplayer", tags=["multiplayer"])

# Global game manager (in production, use Redis or database)
game_manager: Optional[AuthenticatedMultiplayerCAHGame] = None
humor_orchestrator: Optional[ImprovedHumorOrchestrator] = None
persona_manager: Optional[PersonaManager] = None

# CRITICAL: Track instance creation to prevent multiple instances
_game_manager_created = False

# CRITICAL: Lock to prevent race conditions during initialization
import threading
_game_manager_lock = threading.Lock()

logger = logging.getLogger(__name__)

def get_game_manager(db: Session = Depends(get_db)) -> AuthenticatedMultiplayerCAHGame:
    """Get or create the global game manager"""
    global game_manager, humor_orchestrator, persona_manager, _game_manager_created
    
    # CRITICAL: Use lock to prevent race conditions during initialization
    with _game_manager_lock:
        if game_manager is None and not _game_manager_created:
            # Initialize components only once
            if humor_orchestrator is None:
                humor_orchestrator = ImprovedHumorOrchestrator(use_crewai_agents=False)  # Use standard agents for multiplayer
            if persona_manager is None:
                persona_manager = PersonaManager(db)
            
            # Create game manager with initial database session, but don't recreate it
            game_manager = AuthenticatedMultiplayerCAHGame(humor_orchestrator, persona_manager)
            _game_manager_created = True
            logger.info(f"✅ Created new game manager instance. Current games: {list(game_manager.games.keys())}")
            logger.info(f"✅ Game manager instance ID: {id(game_manager)}")
            
            # Sync WebSocket connections with the new game manager
            try:
                from .main import sync_websocket_connections_with_game_manager
                sync_websocket_connections_with_game_manager(game_manager)
                logger.info(f"✅ Synced WebSocket connections with new game manager")
            except Exception as sync_error:
                logger.warning(f"⚠️ Could not sync WebSocket connections: {sync_error}")
        elif game_manager is None and _game_manager_created:
            logger.error(f"❌ CRITICAL: Game manager was created but is now None! This indicates a serious bug!")
            logger.error(f"❌ CRITICAL: Recreating game manager as emergency fix...")
            # Emergency recreation
            game_manager = AuthenticatedMultiplayerCAHGame(humor_orchestrator, persona_manager)
            _game_manager_created = True
        else:
            logger.info(f"🔍 Using existing game manager. Current games: {list(game_manager.games.keys())}")
    
    # CRITICAL: Verify this is the same instance and has the expected games
    logger.info(f"🔍 DEBUG: Game manager instance ID: {id(game_manager)}")
    logger.info(f"🔍 DEBUG: Game manager has {len(game_manager.games)} games")
    
    # CRITICAL: If we have a game manager but it's empty, something is wrong
    if game_manager and len(game_manager.games) == 0 and _game_manager_created:
        logger.error(f"🚨 CRITICAL: Game manager exists but has 0 games! This indicates state loss!")
        logger.error(f"🚨 CRITICAL: This should NEVER happen with a proper singleton!")
        logger.error(f"🚨 CRITICAL: Game manager instance ID: {id(game_manager)}")
        logger.error(f"🚨 CRITICAL: _game_manager_created flag: {_game_manager_created}")
    
    for game_id, game_state in game_manager.games.items():
        logger.info(f"🔍 DEBUG: Game {game_id}: {len(game_state.players)} players, status: {game_state.status.value}")
        if game_state.current_round:
            logger.info(f"🔍 DEBUG: Game {game_id} round: {game_state.current_round.phase.value}, submissions: {len(game_state.current_round.submissions)}")
    
    return game_manager

@router.post("/create-game")
async def create_game(
    settings: Dict[str, Any] = {},
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new multiplayer game"""
    try:
        game_manager = get_game_manager(db)
        
        # Generate unique game ID
        game_id = str(uuid.uuid4())[:8]
        
        # Create game
        logger.info(f"🎮 Creating game {game_id} for user {current_user.id}")
        logger.info(f"🔍 BEFORE: Total games: {len(game_manager.games)}")
        logger.info(f"🚨 FORCED RELOAD TEST - This should appear in logs!")
        
        # Debug: Check if game manager is properly initialized
        logger.info(f"🔍 DEBUG: Game manager type: {type(game_manager)}")
        logger.info(f"🔍 DEBUG: Game manager games attribute: {hasattr(game_manager, 'games')}")
        
        game_state = await game_manager.create_game(db, game_id, current_user.id, settings)
        logger.info(f"✅ Game {game_id} created successfully. Total games: {len(game_manager.games)}")
        logger.info(f"🔍 AFTER: Available games: {list(game_manager.games.keys())}")
        
        # Debug: Verify game was actually stored
        if game_id in game_manager.games:
            logger.info(f"✅ DEBUG: Game {game_id} confirmed in manager")
            stored_game = game_manager.games[game_id]
            logger.info(f"✅ DEBUG: Stored game has {len(stored_game.players)} players")
        else:
            logger.error(f"❌ DEBUG: Game {game_id} NOT found in manager after creation!")
            logger.error(f"❌ DEBUG: Available games: {list(game_manager.games.keys())}")
        
        return {
            "success": True,
            "game_id": game_id,
            "message": "Game created successfully",
            "game_state": {
                "status": game_state.status.value,
                "players": [
                    {
                        "user_id": str(player.user_id),  # Convert to string for frontend compatibility
                        "username": player.username,
                        "is_host": player.is_host,
                        "score": player.score
                    }
                    for player in game_state.players.values()
                ],
                "max_players": game_state.max_players,
                "max_score": game_state.max_score
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to create game: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create game: {str(e)}"
        )

@router.post("/force-create-game")
async def force_create_game(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Force create a new game for debugging host assignment"""
    try:
        game_manager = get_game_manager(db)
        
        # Generate unique game ID
        import uuid
        game_id = str(uuid.uuid4())[:8]
        
        # Create game with current user as host
        logger.info(f"🎮 Force creating game {game_id} for user {current_user.id}")
        
        game_state = await game_manager.create_game(db, game_id, current_user.id, {
            "max_score": 5,
            "max_players": 6,
            "round_timer": 300
        })
        
        # Verify host assignment
        host_player = game_manager.get_host_player(game_id)
        is_current_user_host = game_manager.is_user_host(game_id, current_user.id)
        
        logger.info(f"✅ Game {game_id} created. Host: {host_player.username if host_player else 'None'}")
        logger.info(f"🔍 Current user is host: {is_current_user_host}")
        
        return {
            "success": True,
            "game_id": game_id,
            "message": "Game force created successfully",
            "debug_info": {
                "current_user_id": str(current_user.id),  # Convert to string for frontend compatibility
                "current_user_email": current_user.email,
                "current_user_username": current_user.username,
                "host_player_id": str(host_player.user_id) if host_player else None,  # Convert to string for frontend compatibility
                "host_player_username": host_player.username if host_player else None,
                "is_current_user_host": is_current_user_host,
                "total_players": len(game_state.players)
            },
            "game_state": {
                "status": game_state.status.value,
                "players": [
                    {
                        "user_id": str(player.user_id),  # Convert to string for frontend compatibility
                        "username": player.username,
                        "is_host": player.is_host,
                        "score": player.score
                    }
                    for player in game_state.players.values()
                ],
                "max_players": game_state.max_players,
                "max_score": game_state.max_score
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to force create game: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to force create game: {str(e)}"
        )

@router.get("/games")
async def list_games(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List all available games (for debugging)"""
    try:
        game_manager = get_game_manager(db)
        logger.info(f"🔍 DEBUG: Listing games. Total games: {len(game_manager.games)}")
        logger.info(f"🔍 DEBUG: Game IDs: {list(game_manager.games.keys())}")
        
        games = []
        for game_id, game_state in game_manager.games.items():
            games.append({
                "game_id": game_id,
                "status": game_state.status.value,
                "player_count": len(game_state.players),
                "max_players": game_state.max_players,
                "created_at": game_state.created_at.isoformat() if game_state.created_at else None
            })
        
        return {
            "success": True,
            "games": games,
            "total_games": len(games)
        }
        
    except Exception as e:
        logger.error(f"Failed to list games: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list games: {str(e)}"
        )

@router.post("/join-game/{game_id}")
async def join_game(
    game_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Join an existing multiplayer game"""
    logger.info(f"🎯 NEW CODE ACTIVE: join_game called for game {game_id} by user {current_user.id}")
    print(f"🎯 DEBUG: join_game function called for game {game_id} by user {current_user.id}")
    try:
        game_manager = get_game_manager(db)
        print(f"🎯 DEBUG: Got game manager successfully")
        
        # Join game
        logger.info(f"🎮 User {current_user.id} attempting to join game {game_id}")
        logger.info(f"🔍 Available games: {list(game_manager.games.keys())}")
        logger.info(f"🔍 Game {game_id} exists: {game_id in game_manager.games}")
        
        # Debug: Check game manager state
        logger.info(f"🔍 DEBUG: Game manager has {len(game_manager.games)} games")
        for gid, gs in game_manager.games.items():
            logger.info(f"🔍 DEBUG: Game {gid}: {len(gs.players)} players, status: {gs.status.value}")
        
        if game_id not in game_manager.games:
            logger.error(f"❌ DEBUG: Game {game_id} not found in manager!")
            logger.error(f"❌ DEBUG: Available games: {list(game_manager.games.keys())}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Game {game_id} not found"
            )
        
        success = await game_manager.join_game(db, game_id, current_user.id)
        logger.info(f"🔍 Join result: {success}")
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Could not join game"
            )
        
        # Get updated game state
        game_state = game_manager.get_game_state(game_id)
        logger.info(f"🎯 ABOUT TO BROADCAST: game_state retrieved, proceeding to broadcast")
        print(f"🎯 DEBUG: About to broadcast for game {game_id}")
        print(f"🎯 DEBUG: Game state has {len(game_state.players)} players")
        print(f"🎯 DEBUG: Current user ID: {current_user.id}")
        print(f"🎯 DEBUG: Game manager type: {type(game_manager)}")
        print(f"🎯 DEBUG: Has broadcast_to_game method: {hasattr(game_manager, 'broadcast_to_game')}")
        
        # Broadcast player joined to all WebSocket connections using the authenticated game manager
        logger.info(f"🚀 Starting broadcast for game {game_id}")
        print(f"🎯 BROADCAST DEBUG: Starting broadcast for game {game_id}")
        try:
            # Use the authenticated game manager's broadcast method
            if hasattr(game_manager, 'broadcast_to_game'):
                print(f"🎯 BROADCAST DEBUG: Calling game_manager.broadcast_to_game")
                
                # Create a comprehensive player joined message with updated game state
                player_joined_message = {
                    "type": "player_joined",
                    "new_player_id": str(current_user.id),  # Convert to string for frontend compatibility
                    "new_player_username": current_user.username or current_user.email.split('@')[0],
                    "game_id": game_id,
                    "game_state": {
                        "status": game_state.status.value,
                        "players": [
                            {
                                "user_id": str(player.user_id),  # Convert to string for frontend compatibility
                                "username": player.username,
                                "is_host": player.is_host,
                                "is_judge": player.is_judge,
                                "score": player.score,
                                "connected": player.connected
                            }
                            for player in game_state.players.values()
                        ],
                        "max_players": game_state.max_players,
                        "max_score": game_state.max_score,
                        "current_round": None  # Game hasn't started yet, so no current round
                    }
                }
                
                await game_manager.broadcast_to_game(game_id, player_joined_message)
                logger.info(f"✅ Broadcasted player join for user {current_user.id} using authenticated game manager")
                print(f"🎯 BROADCAST DEBUG: Broadcast successful!")
                
                # Also broadcast updated game state to ensure all players see the current state
                try:
                    # Get fresh game state for all players
                    for player_id, player in game_state.players.items():
                        player_game_state = game_manager.get_game_state_for_user(game_id, player_id)
                        if player_game_state:
                            # Send updated state to each player
                            websocket_connections = game_manager.get_websocket_connections(game_id)
                            if player_id in websocket_connections:
                                websocket = websocket_connections[player_id]
                                await websocket.send_text(json.dumps({
                                    "type": "game_state_refreshed",
                                    "game_state": player_game_state,
                                    "message": f"Game state updated after {current_user.username} joined"
                                }))
                                logger.info(f"✅ Sent refreshed game state to player {player_id}")
                except Exception as refresh_error:
                    logger.warning(f"⚠️ Could not refresh game state for all players: {refresh_error}")
            else:
                logger.warning("❌ Game manager doesn't have broadcast_to_game method - falling back to old system")
                print(f"🎯 BROADCAST DEBUG: Falling back to old system")
                # Fallback to old broadcast system
                import agent_system.api.cah_crewai_api as cah_api
                if hasattr(cah_api, 'broadcast_function') and cah_api.broadcast_function:
                    await cah_api.broadcast_function(game_id, {
                        "type": "player_joined",
                        "new_player_id": str(current_user.id),  # Convert to string for frontend compatibility
                        "new_player_username": current_user.username or current_user.email.split('@')[0],
                        "game_id": game_id
                    })
                    logger.info(f"✅ Fallback broadcast successful for user {current_user.id}")
                else:
                    logger.error(f"❌ No broadcast method available - players won't see real-time updates")
        
        except Exception as broadcast_error:
            logger.error(f"❌ Failed to broadcast player join: {broadcast_error}")
            import traceback
            logger.error(f"❌ Broadcast error traceback: {traceback.format_exc()}")
            print(f"🎯 BROADCAST DEBUG: Exception caught: {broadcast_error}")
            print(f"🎯 BROADCAST DEBUG: Traceback: {traceback.format_exc()}")
            

        
        return {
            "success": True,
            "message": "Joined game successfully",
            "game_state": {
                "status": game_state.status.value,
                "players": [
                    {
                        "user_id": str(player.user_id),  # Convert to string for frontend compatibility
                        "username": player.username,
                        "is_host": player.is_host,
                        "score": player.score
                    }
                    for player in game_state.players.values()
                ],
                "max_players": game_state.max_players,
                "max_score": game_state.max_score
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to join game: {e}")
        import traceback
        logger.error(f"❌ Join game error traceback: {traceback.format_exc()}")
        print(f"🎯 EXCEPTION CAUGHT: {e}")
        print(f"🎯 EXCEPTION TRACEBACK: {traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to join game: {str(e)}"
        )

@router.post("/start-game/{game_id}")
async def start_game(
    game_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Start a multiplayer game"""
    try:
        game_manager = get_game_manager(db)
        
        # Check if game exists
        game_state = game_manager.get_game_state(game_id)
        if not game_state:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Game not found"
            )
        
        # Check if user is in the game
        player = game_manager.get_player_info(game_id, current_user.id)
        if not player:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You are not in this game"
            )
        
        # Check if user is host using the improved method
        if not game_manager.is_user_host(game_id, current_user.id):
            # Get host player info for better error message
            host_player = game_manager.get_host_player(game_id)
            host_info = f"Host: {host_player.username} (ID: {host_player.user_id})" if host_player else "Unknown host"
            current_info = f"Current user: {current_user.username} (ID: {current_user.id})"
            
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Only the host can start the game. {host_info}. {current_info}"
            )
        
        # Start game
        success = await game_manager.start_game(db, game_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Could not start game"
            )
        
        # Get updated game state
        updated_game_state = game_manager.get_game_state(game_id)
        
        # Broadcast game started message to all players
        try:
            if hasattr(game_manager, 'broadcast_to_game'):
                # Get the full game state for each user (with their own hands)
                game_started_message = {
                    "type": "game_started",
                    "game_id": game_id,
                    "message": "Game started successfully! Check your hand for white cards."
                }
                
                # Broadcast to each player individually with their own hand
                for player in updated_game_state.players.values():
                    player_game_state = game_manager.get_game_state_for_user(game_id, player.user_id)
                    if player_game_state:
                        player_message = {
                            **game_started_message,
                            "game_state": player_game_state
                        }
                        
                        # Send to this specific player
                        await game_manager.broadcast_to_game(game_id, player_message, exclude_user=player.user_id)
                        
                        # Also send directly to their WebSocket if available
                        try:
                            websocket_connections = game_manager.get_websocket_connections(game_id)
                            if player.user_id in websocket_connections:
                                websocket = websocket_connections[player.user_id]
                                await websocket.send_text(json.dumps(player_message))
                                logger.info(f"✅ Sent game state directly to player {player.user_id}")
                        except Exception as ws_error:
                            logger.warning(f"⚠️ Could not send direct WebSocket message to player {player.user_id}: {ws_error}")
                
                logger.info(f"✅ Broadcasted game started message with individual game states for game {game_id}")
            else:
                logger.warning("❌ Game manager doesn't have broadcast_to_game method")
        except Exception as broadcast_error:
            logger.error(f"❌ Failed to broadcast game started: {broadcast_error}")
        
        return {
            "success": True,
            "message": "Game started successfully",
            "game_state": {
                "status": updated_game_state.status.value,
                "current_round": {
                    "round_number": updated_game_state.current_round.round_number,
                    "black_card": updated_game_state.current_round.black_card,
                    "judge_id": str(updated_game_state.current_round.judge_id),  # Convert to string for frontend compatibility
                    "phase": updated_game_state.current_round.phase.value
                } if updated_game_state.current_round else None,
                "players": [
                    {
                        "user_id": str(player.user_id),  # Convert to string for frontend compatibility
                        "username": player.username,
                        "is_host": player.is_host,
                        "score": player.score,
                        "is_judge": player.is_judge,
                        "hand_size": len(player.hand)
                    }
                    for player in updated_game_state.players.values()
                ]
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to start game: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start game: {str(e)}"
        )

@router.get("/game-state/{game_id}")
async def get_game_state(
    game_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get current game state"""
    try:
        game_manager = get_game_manager(db)
        
        game_state = game_manager.get_game_state(game_id)
        if not game_state:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Game not found"
            )
        
        # Check if user is in the game
        player = game_manager.get_player_info(game_id, current_user.id)
        if not player:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You are not in this game"
            )
        
        # Get host player info
        host_player = game_manager.get_host_player(game_id)
        
        # Debug: Check host identification logic
        debug_host_info = {
            "current_user_id": str(current_user.id),  # Convert to string for frontend compatibility
            "current_user_email": current_user.email,
            "current_user_username": current_user.username,
            "is_current_user_host": player.is_host,
            "host_player_id": str(host_player.user_id) if host_player else None,  # Convert to string for frontend compatibility
            "host_player_username": host_player.username if host_player else None,
            "host_player_email": host_player.email if host_player else None,
            "total_players": len(game_state.players),
            "all_players_debug": []
        }
        
        # Add debug info for all players
        for player_id, p in game_state.players.items():
            debug_host_info["all_players_debug"].append({
                "player_id": str(player_id),  # Convert to string for frontend compatibility
                "email": p.email,
                "username": p.username,
                "is_host": p.is_host,
                "is_judge": p.is_judge
            })
        
        # Use the new method to get game state with proper hand management
        user_game_state = game_manager.get_game_state_for_user(game_id, current_user.id)
        if not user_game_state:
            # Try to sync with database and get state again
            logger.warning(f"Failed to get game state for user {current_user.id}, attempting to sync with database")
            sync_success = await game_manager.sync_game_with_database(db, game_id)
            if sync_success:
                user_game_state = game_manager.get_game_state_for_user(game_id, current_user.id)
                if user_game_state:
                    logger.info(f"Successfully synced game state for user {current_user.id}")
                else:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="Failed to format game state even after database sync"
                    )
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to format game state and database sync failed"
                )
        
        # ENHANCED: If user is judge, send special judge notifications for both phases
        if (game_state.current_round and 
            game_state.current_round.judge_id == current_user.id and 
            game_state.current_round.phase.value in ["card_submission", "judging"]):
            
            try:
                # Get submissions for judging (works in both phases)
                submissions_for_judging = game_manager.get_submissions_for_judging(game_id, current_user.id)
                
                # Determine message based on phase
                if game_state.current_round.phase.value == "judging":
                    message = "You can now judge the round!"
                    notification_type = "judge_ready_notification"
                else:
                    message = f"Waiting for cards... ({len(game_state.current_round.submissions)} submitted)"
                    notification_type = "judge_waiting_notification"
                
                # Send WebSocket update to judge with submissions
                if hasattr(game_manager, 'get_websocket_connections'):
                    websocket_connections = game_manager.get_websocket_connections(game_id)
                    if current_user.id in websocket_connections:
                        judge_websocket = websocket_connections[current_user.id]
                        await judge_websocket.send_text(json.dumps({
                            "type": notification_type,
                            "game_id": game_id,
                            "message": message,
                            "game_state": user_game_state,
                            "submissions_for_judging": submissions_for_judging,
                            "submissions_count": len(game_state.current_round.submissions),
                            "phase": game_state.current_round.phase.value
                        }))
                        logger.info(f"🎯 ✅ Sent {notification_type} to judge {current_user.id}")
                        
                # Also add submissions directly to the response for judges
                if submissions_for_judging:
                    user_game_state["judge_submissions"] = submissions_for_judging
                    
            except Exception as judge_error:
                logger.warning(f"⚠️ Could not send judge notification: {judge_error}")
        
        # CRITICAL: Always get the latest in-memory state, not database state
        logger.info(f"🎯 Getting latest in-memory game state for user {current_user.id}")
        
        # CRITICAL: Verify we're using the same game manager instance
        logger.info(f"🎯 Game manager instance ID: {id(game_manager)}")
        logger.info(f"🎯 Game {game_id} in manager: {game_id in game_manager.games}")
        
        # CRITICAL: Verify game manager has the expected games
        if game_id not in game_manager.games:
            logger.error(f"🚨 CRITICAL: Game {game_id} NOT found in game manager!")
            logger.error(f"🚨 CRITICAL: Available games: {list(game_manager.games.keys())}")
            logger.error(f"🚨 CRITICAL: Game manager instance ID: {id(game_manager)}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Game {game_id} not found in game manager"
            )
        
        # Get the fresh in-memory state directly (no need to force refresh every time)
        user_game_state = game_manager.get_game_state_for_user(game_id, current_user.id)
        
        if not user_game_state:
            logger.error(f"❌ Failed to get in-memory game state for user {current_user.id}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to get game state"
            )
        
        # Log the actual data being returned
        logger.info(f"🎯 Returning game state with {len(user_game_state.get('players', []))} players")
        if 'current_round' in user_game_state and user_game_state['current_round']:
            logger.info(f"🎯 Current round: phase={user_game_state['current_round'].get('phase')}, submissions={len(user_game_state['current_round'].get('submissions', []))}")
        
        # Verify data integrity
        if len(user_game_state.get('players', [])) == 0:
            logger.error(f"🚨 CRITICAL: Game state has 0 players! This indicates a serious issue.")
            logger.error(f"🚨 Game ID: {game_id}")
            logger.error(f"🚨 User ID: {current_user.id}")
            logger.error(f"🚨 In-memory players count: {len(game_manager.games.get(game_id, {}).players if game_id in game_manager.games else 'N/A')}")
        
        if 'current_round' in user_game_state and user_game_state['current_round']:
            submissions_count = len(user_game_state['current_round'].get('submissions', []))
            if submissions_count == 0 and game_manager.games.get(game_id, {}).current_round:
                actual_submissions = len(game_manager.games[game_id].current_round.submissions)
                logger.error(f"🚨 CRITICAL: Game state shows 0 submissions but in-memory has {actual_submissions}!")
                logger.error(f"🚨 In-memory submissions: {game_manager.games[game_id].current_round.submissions}")
        
        return {
            "success": True,
            "game_state": user_game_state,
            "debug_info": debug_host_info
        }
        
    except Exception as e:
        logger.error(f"Failed to get game state: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get game state: {str(e)}"
        )

@router.get("/debug-game/{game_id}")
async def debug_game(
    game_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Debug endpoint to get comprehensive game information"""
    try:
        game_manager = get_game_manager(db)
        
        # CRITICAL: Add game manager instance debugging
        game_manager_debug = {
            "instance_id": id(game_manager),
            "total_games": len(game_manager.games),
            "game_ids": list(game_manager.games.keys()),
            "games_details": {}
        }
        
        for gid, gs in game_manager.games.items():
            game_manager_debug["games_details"][gid] = {
                "players_count": len(gs.players),
                "status": gs.status.value,
                "has_current_round": gs.current_round is not None,
                "current_round_phase": gs.current_round.phase.value if gs.current_round else None,
                "submissions_count": len(gs.current_round.submissions) if gs.current_round else 0
            }
        
        # Get debug information
        debug_info = game_manager.debug_game_state(game_id)
        
        # Also get the current user's game state
        user_game_state = game_manager.get_game_state_for_user(game_id, current_user.id)
        
        return {
            "success": True,
            "game_manager_debug": game_manager_debug,
            "debug_info": debug_info,
            "user_game_state": user_game_state,
            "current_user_id": current_user.id,
            "current_user_email": current_user.email
        }
        
    except Exception as e:
        logger.error(f"Failed to debug game: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to debug game: {str(e)}"
        )

@router.get("/judge-status/{game_id}")
async def get_judge_status(
    game_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get submissions status for judges during card submission phase"""
    try:
        game_manager = get_game_manager(db)
        
        # Check if user is in the game
        player = game_manager.get_player_info(game_id, current_user.id)
        if not player:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You are not in this game"
            )
        
        # Get current game state for judge
        game_state = game_manager.get_game_state(game_id)
        if not game_state or not game_state.current_round:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No active round"
            )
        
        # Check if user is judge
        if game_state.current_round.judge_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You are not the judge for this round"
            )
        
        # Get judge status based on current phase
        if game_state.current_round.phase.value == "card_submission":
            # During card submission: show submission status
            judge_status = game_manager.get_submissions_status_for_judge(game_id, current_user.id)
            if judge_status:
                return {
                    "success": True,
                    "type": "card_submission",
                    "data": judge_status
                }
        elif game_state.current_round.phase.value == "judging":
            # During judging: show submissions for judging
            judging_data = game_manager.get_submissions_for_judging(game_id, current_user.id)
            if judging_data:
                # ENHANCED: Also send game state update to judge when they poll for submissions
                try:
                    judge_game_state = game_manager.get_game_state_for_user(game_id, current_user.id)
                    if judge_game_state and hasattr(game_manager, 'get_websocket_connections'):
                        websocket_connections = game_manager.get_websocket_connections(game_id)
                        if current_user.id in websocket_connections:
                            judge_websocket = websocket_connections[current_user.id]
                            await judge_websocket.send_text(json.dumps({
                                "type": "judge_submissions_update",
                                "game_id": game_id,
                                "message": "Updated submissions for judging",
                                "game_state": judge_game_state,
                                "submissions_for_judging": judging_data
                            }))
                            logger.info(f"🎯 ✅ Sent judge submissions update via WebSocket to judge {current_user.id}")
                except Exception as ws_error:
                    logger.warning(f"⚠️ Could not send WebSocket update to judge: {ws_error}")
                
                return {
                    "success": True,
                    "type": "judging",
                    "data": judging_data,
                    "game_state": game_manager.get_game_state_for_user(game_id, current_user.id)
                }
        
        # Fallback: no submissions available yet
        return {
            "success": True,
            "type": "waiting",
            "data": {
                "message": "Waiting for submissions",
                "phase": game_state.current_round.phase.value,
                "submissions_count": len(game_state.current_round.submissions)
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get judge status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get judge status: {str(e)}"
        )

@router.get("/judge-submissions/{game_id}")
async def get_judge_submissions(
    game_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all submissions visible to the judge (both during submission and judging phases)"""
    try:
        game_manager = get_game_manager(db)
        
        # Check if user is in the game and is judge
        player = game_manager.get_player_info(game_id, current_user.id)
        if not player:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You are not in this game"
            )
        
        game_state = game_manager.get_game_state(game_id)
        if not game_state or not game_state.current_round:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No active round"
            )
        
        if game_state.current_round.judge_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You are not the judge for this round"
            )
        
        # Get submissions data for judge
        submissions_data = game_manager.get_submissions_for_judging(game_id, current_user.id)
        judge_game_state = game_manager.get_game_state_for_user(game_id, current_user.id)
        
        # Send real-time update to judge via WebSocket
        try:
            if hasattr(game_manager, 'get_websocket_connections'):
                websocket_connections = game_manager.get_websocket_connections(game_id)
                if current_user.id in websocket_connections:
                    judge_websocket = websocket_connections[current_user.id]
                    await judge_websocket.send_text(json.dumps({
                        "type": "judge_submissions_refresh",
                        "game_id": game_id,
                        "message": "Judge submissions refreshed",
                        "game_state": judge_game_state,
                        "submissions_for_judging": submissions_data
                    }))
                    logger.info(f"🎯 ✅ Sent judge submissions refresh to judge {current_user.id}")
        except Exception as ws_error:
            logger.warning(f"⚠️ Could not send WebSocket refresh to judge: {ws_error}")
        
        return {
            "success": True,
            "submissions_data": submissions_data,
            "game_state": judge_game_state,
            "phase": game_state.current_round.phase.value,
            "can_judge": game_state.current_round.phase.value == "judging"
        }
        
    except Exception as e:
        logger.error(f"Failed to get judge submissions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get judge submissions: {str(e)}"
        )

@router.post("/refresh-game/{game_id}")
async def refresh_game(
    game_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Force refresh the game state"""
    try:
        game_manager = get_game_manager(db)
        
        # Force refresh the game state
        success = await game_manager.force_refresh_game_state(db, game_id)
        
        if success:
            # Get updated debug info
            debug_info = game_manager.debug_game_state(game_id)
            user_game_state = game_manager.get_game_state_for_user(game_id, current_user.id)
            
            # Broadcast the refreshed state to all players
            try:
                await game_manager.broadcast_to_game(game_id, {
                    "type": "game_state_refreshed",
                    "game_id": game_id,
                    "message": "Game state has been refreshed by admin",
                    "action": "force_refresh"
                })
                logger.info(f"✅ Broadcasted game state refresh for game {game_id}")
            except Exception as broadcast_error:
                logger.warning(f"⚠️ Could not broadcast refresh: {broadcast_error}")
            
            return {
                "success": True,
                "message": "Game state refreshed successfully",
                "debug_info": debug_info,
                "user_game_state": user_game_state
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to refresh game state"
            )
        
    except Exception as e:
        logger.error(f"Failed to refresh game: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to refresh game: {str(e)}"
        )

@router.get("/player-hand/{game_id}")
async def get_player_hand(
    game_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get the current player's hand for the game"""
    try:
        game_manager = get_game_manager(db)
        
        # Check if user is in the game
        player = game_manager.get_player_info(game_id, current_user.id)
        if not player:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You are not in this game"
            )
        
        # Get the player's hand
        hand = game_manager.get_player_hand(game_id, current_user.id)
        
        return {
            "success": True,
            "hand": hand,
            "hand_size": len(hand)
        }
        
    except Exception as e:
        logger.error(f"Failed to get player hand: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get player hand: {str(e)}"
        )

@router.post("/replenish-hand/{game_id}")
async def replenish_hand(
    game_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Replenish the current player's hand"""
    try:
        game_manager = get_game_manager(db)
        
        # Check if user is in the game
        player = game_manager.get_player_info(game_id, current_user.id)
        if not player:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You are not in this game"
            )
        
        # Replenish the player's hand
        success = await game_manager.replenish_player_hand(db, game_id, current_user.id, 5)
        
        if success:
            # Get the updated hand
            hand = game_manager.get_player_hand(game_id, current_user.id)
            
            return {
                "success": True,
                "message": "Hand replenished successfully",
                "hand": hand,
                "hand_size": len(hand)
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to replenish hand"
            )
        
    except Exception as e:
        logger.error(f"Failed to replenish hand: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to replenish hand: {str(e)}"
        )

@router.post("/leave-game/{game_id}")
async def leave_game(
    game_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Leave a multiplayer game"""
    try:
        game_manager = get_game_manager(db)
        
        # Get game state before leaving
        game_state = game_manager.get_game_state(game_id)
        if not game_state:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Game not found"
            )
        
        # Leave game
        success = game_manager.leave_game(game_id, current_user.id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Could not leave game"
            )
        
        # Broadcast player left message to remaining players
        try:
            if hasattr(game_manager, 'broadcast_to_game'):
                player_left_message = {
                    "type": "player_left",
                    "game_id": game_id,
                    "user_id": str(current_user.id),  # Convert to string for frontend compatibility
                    "username": current_user.username or current_user.email.split('@')[0],
                    "game_state": {
                        "status": game_state.status.value,
                        "players": [
                            {
                                "user_id": str(player.user_id),  # Convert to string for frontend compatibility
                                "username": player.username,
                                "is_host": player.is_host,
                                "is_judge": player.is_judge,
                                "score": player.score,
                                "connected": player.connected
                            }
                            for player in game_state.players.values()
                        ],
                        "max_players": game_state.max_players,
                        "max_score": game_state.max_score
                    }
                }
                
                await game_manager.broadcast_to_game(game_id, player_left_message)
                logger.info(f"✅ Broadcasted player left message for game {game_id}")
            else:
                logger.warning("❌ Game manager doesn't have broadcast_to_game method")
        except Exception as broadcast_error:
            logger.error(f"❌ Failed to broadcast player left: {broadcast_error}")
        
        return {
            "success": True,
            "message": "Left game successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to leave game: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to leave game: {str(e)}"
        )

@router.get("/available-games")
async def get_available_games(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get list of available games to join"""
    try:
        game_manager = get_game_manager(db)
        
        available_games = []
        for game_id, game_state in game_manager.games.items():
            if game_state.status.value == "waiting" and current_user.id not in game_state.players:
                available_games.append({
                    "game_id": game_id,
                    "host": next(p.username for p in game_state.players.values() if p.is_host),
                    "player_count": len(game_state.players),
                    "max_players": game_state.max_players,
                    "created_at": game_state.created_at.isoformat()
                })
        
        return {
            "success": True,
            "available_games": available_games
        }
        
    except Exception as e:
        logger.error(f"Failed to get available games: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get available games: {str(e)}"
        )

@router.get("/debug/users")
async def debug_users(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Debug endpoint to list all users and their games"""
    try:
        # Get all users
        all_users = db.query(User).all()
        users_info = []
        for user in all_users:
            users_info.append({
                "id": str(user.id),  # Convert to string for frontend compatibility
                "email": user.email,
                "username": user.username,
                "created_at": user.created_at.isoformat() if user.created_at else None
            })
        
        # Get current user's games from both memory and database
        game_manager = get_game_manager(db)
        
        # Get games from memory
        memory_games = []
        for game_id, game_state in game_manager.games.items():
            if current_user.id in game_state.players:
                memory_games.append({
                    "game_id": game_id,
                    "is_host": game_state.players[current_user.id].is_host,
                    "player_count": len(game_state.players),
                    "status": game_state.status.value,
                    "source": "memory"
                })
        
        # Get games from database
        try:
            from models.database import Game, GamePlayer
            db_games = db.query(Game).join(GamePlayer).filter(GamePlayer.user_id == current_user.id).all()
            db_games_info = []
            for game in db_games:
                player = db.query(GamePlayer).filter(
                    GamePlayer.game_id == game.id,
                    GamePlayer.user_id == current_user.id
                ).first()
                if player:
                    db_games_info.append({
                        "game_id": game.id,
                        "is_host": player.is_host,
                        "player_count": db.query(GamePlayer).filter(GamePlayer.game_id == game.id).count(),
                        "status": game.status,
                        "source": "database"
                    })
        except Exception as db_error:
            logger.error(f"Failed to query database games: {db_error}")
            db_games_info = []
        
        # Combine both sources
        all_current_user_games = memory_games + db_games_info
        
        return {
            "success": True,
            "debug_info": {
                "current_user": {
                    "id": str(current_user.id),  # Convert to string for frontend compatibility
                    "email": current_user.email,
                    "username": current_user.username
                },
                "all_users": users_info,
                "current_user_games": all_current_user_games,
                "memory_games": len(game_manager.games),
                "database_games": len(db_games_info)
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get debug info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get debug info: {str(e)}"
        )

@router.get("/debug/game/{game_id}")
async def debug_game(
    game_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Debug endpoint to check a specific game's details"""
    try:
        game_manager = get_game_manager(db)
        game_state = game_manager.get_game_state(game_id)
        
        if not game_state:
            return {
                "success": False,
                "error": "Game not found"
            }
        
        # Check if current user is in the game
        current_player = game_manager.get_player_info(game_id, current_user.id)
        
        # Get all players with their details
        players_info = []
        for player_id, player in game_state.players.items():
            players_info.append({
                "user_id": player.user_id,
                "email": player.email,
                "username": player.username,
                "is_host": player.is_host,
                "is_judge": player.is_judge,
                "score": player.score,
                "connected": player.connected
            })
        
        # Also try to get game info from database
        db_game_info = None
        try:
            from models.database import Game, GamePlayer
            db_game = db.query(Game).filter(Game.id == game_id).first()
            if db_game:
                db_players = db.query(GamePlayer).filter(GamePlayer.game_id == game_id).all()
                db_game_info = {
                    "game_id": db_game.id,
                    "status": db_game.status,
                    "created_at": db_game.created_at.isoformat() if db_game.created_at else None,
                    "players": [
                        {
                            "player_id": p.id,
                            "user_id": p.user_id,
                            "username": p.username,
                            "is_host": p.is_host,
                            "is_judge": p.is_judge,
                            "score": p.score,
                            "joined_at": p.joined_at.isoformat() if p.joined_at else None
                        }
                        for p in db_players
                    ]
                }
        except Exception as db_error:
            logger.error(f"Failed to query database game: {db_error}")
        
        return {
            "success": True,
            "debug_info": {
                "game_id": game_id,
                "game_status": game_state.status.value if game_state else None,
                "current_user": {
                    "id": current_user.id,
                    "email": current_user.email,
                    "username": current_user.username
                },
                "current_player_in_game": current_player is not None,
                "current_player_is_host": current_player.is_host if current_player else False,
                "all_players": players_info,
                "total_players": len(game_state.players) if game_state else None,
                "host_player": next((p for p in players_info if p["is_host"]), None),
                "database_game": db_game_info
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get game debug info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get game debug info: {str(e)}"
        )

@router.get("/debug/game-manager")
async def debug_game_manager(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Debug endpoint to check game manager state"""
    try:
        game_manager = get_game_manager(db)
        
        debug_info = {
            "total_games": len(game_manager.games),
            "games": []
        }
        
        for game_id, game_state in game_manager.games.items():
            game_debug = {
                "game_id": game_id,
                "status": game_state.status.value,
                "created_at": game_state.created_at.isoformat() if game_state.created_at else None,
                "players": []
            }
            
            for player_id, player in game_state.players.items():
                game_debug["players"].append({
                    "player_id": player_id,
                    "email": player.email,
                    "username": player.username,
                    "is_host": player.is_host,
                    "is_judge": player.is_judge,
                    "score": player.score
                })
            
            debug_info["games"].append(game_debug)
        
        # Also get games from database
        db_games_info = []
        try:
            from models.database import Game, GamePlayer
            db_games = db.query(Game).all()
            for db_game in db_games:
                db_players = db.query(GamePlayer).filter(GamePlayer.game_id == db_game.id).all()
                db_game_debug = {
                    "game_id": db_game.id,
                    "status": db_game.status,
                    "created_at": db_game.created_at.isoformat() if db_game.created_at else None,
                    "source": "database",
                    "players": [
                        {
                            "player_id": p.id,
                            "user_id": p.user_id,
                            "username": p.username,
                            "is_host": p.is_host,
                            "is_judge": p.is_judge,
                            "score": p.score
                        }
                        for p in db_players
                    ]
                }
                db_games_info.append(db_game_debug)
        except Exception as db_error:
            logger.error(f"Failed to query database games: {db_error}")
        
        # Combine both sources
        all_games = debug_info["games"] + db_games_info
        
        return {
            "success": True,
            "debug_info": {
                "total_games": len(all_games),
                "memory_games": len(game_manager.games),
                "database_games": len(db_games_info),
                "games": all_games
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get game manager debug info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get game manager debug info: {str(e)}"
        )

@router.get("/debug/game-cards/{game_id}")
async def debug_game_cards(
    game_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Debug endpoint to check game cards and player hands"""
    try:
        game_manager = get_game_manager(db)
        
        # Check if user is in the game
        player = game_manager.get_player_info(game_id, current_user.id)
        if not player:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You are not in this game"
            )
        
        # Get game state
        game_state = game_manager.get_game_state(game_id)
        if not game_state:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Game not found"
            )
        
        # Get detailed card information
        card_debug_info = {
            "game_id": game_id,
            "status": game_state.status.value,
            "players": [],
            "current_round": None,
            "debug_info": {
                "total_players": len(game_state.players),
                "cards_prepared": bool(game_state.prepared_white_cards),
                "black_cards_count": len(game_state.prepared_black_cards) if game_state.prepared_black_cards else 0
            }
        }
        
        # Add player information with hand details
        for player_id, p in game_state.players.items():
            player_info = {
                "user_id": str(player_id),
                "username": p.username,
                "email": p.email,
                "is_host": p.is_host,
                "is_judge": p.is_judge,
                "score": p.score,
                "hand_size": len(p.hand),
                "hand_sample": p.hand[:5] if p.hand else [],  # Show first 5 cards
                "has_hand": bool(p.hand),
                "hand_type": type(p.hand).__name__
            }
            card_debug_info["players"].append(player_info)
        
        # Add current round information
        if game_state.current_round:
            card_debug_info["current_round"] = {
                "round_number": game_state.current_round.round_number,
                "black_card": game_state.current_round.black_card,
                "judge_id": str(game_state.current_round.judge_id),
                "phase": game_state.current_round.phase.value,
                "submissions_count": len(game_state.current_round.submissions),
                "submissions": [
                    {
                        "user_id": str(user_id),
                        "card": card
                    }
                    for user_id, card in game_state.current_round.submissions.items()
                ]
            }
        
        return {
            "success": True,
            "debug_info": card_debug_info
        }
        
    except Exception as e:
        logger.error(f"Failed to debug game cards: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to debug game cards: {str(e)}"
        )

@router.post("/submit-card/{game_id}")
async def submit_card(
    game_id: str,
    white_card: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Submit a white card for the current round"""
    try:
        game_manager = get_game_manager(db)
        
        # Submit card
        success = await game_manager.submit_card(game_id, current_user.id, white_card)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Could not submit card"
            )
        
        # Get updated game state
        game_state = game_manager.get_game_state(game_id)
        
        # Broadcast card submitted message to all players
        try:
            if hasattr(game_manager, 'broadcast_to_game'):
                card_submitted_message = {
                    "type": "card_submitted",
                    "game_id": game_id,
                    "user_id": str(current_user.id),  # Convert to string for frontend compatibility
                    "username": current_user.username or current_user.email.split('@')[0],
                    "game_state": {
                        "status": game_state.status.value,
                        "current_round": {
                            "round_number": game_state.current_round.round_number,
                            "black_card": game_state.current_round.black_card,
                            "judge_id": str(game_state.current_round.judge_id),  # Convert to string for frontend compatibility
                            "phase": game_state.current_round.phase.value,
                            "submissions_count": len(game_state.current_round.submissions)
                        } if game_state.current_round else None
                    }
                }
                
                await game_manager.broadcast_to_game(game_id, card_submitted_message)
                logger.info(f"✅ Broadcasted card submitted message for game {game_id}")
            else:
                logger.warning("❌ Game manager doesn't have broadcast_to_game method")
        except Exception as broadcast_error:
            logger.error(f"❌ Failed to broadcast card submitted: {broadcast_error}")
        
        return {
            "success": True,
            "message": "Card submitted successfully",
            "game_state": {
                "status": game_state.status.value,
                "current_round": None  # Game hasn't started yet, so no current round
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to submit card: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit card: {str(e)}"
        )

@router.post("/judge-round/{game_id}")
async def judge_round(
    game_id: str,
    winning_card: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Judge selects the winning card"""
    try:
        game_manager = get_game_manager(db)
        
        # Check if user is judge
        game_state = game_manager.get_game_state(game_id)
        if not game_state or not game_state.current_round:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No active round"
            )
        
        if game_state.current_round.judge_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only the judge can select the winner"
            )
        
        # Judge round
        success = await game_manager.judge_round(db, game_id, current_user.id, winning_card)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Could not judge round"
            )
        
        # Get updated game state
        updated_game_state = game_manager.get_game_state(game_id)
        
        # Get the winning user ID from the updated game state
        winning_user_id = updated_game_state.current_round.winner_id if updated_game_state.current_round else None
        
        # Broadcast round complete message to all players
        try:
            if hasattr(game_manager, 'broadcast_to_game'):
                round_complete_message = {
                    "type": "round_complete",
                    "game_id": game_id,
                    "winner_id": str(winning_user_id) if winning_user_id else None,  # Convert to string for frontend compatibility
                    "winning_card": updated_game_state.current_round.winning_card if updated_game_state.current_round else None,
                    "game_state": {
                        "status": updated_game_state.status.value,
                        "current_round": {
                            "round_number": updated_game_state.current_round.round_number,
                            "winner_id": str(updated_game_state.current_round.winner_id),  # Convert to string for frontend compatibility
                            "winning_card": updated_game_state.current_round.winning_card,
                            "phase": updated_game_state.current_round.phase.value
                        } if updated_game_state.current_round else None,
                        "players": [
                            {
                                "user_id": str(player.user_id),  # Convert to string for frontend compatibility
                                "username": player.username,
                                "score": player.score
                            }
                            for player in updated_game_state.players.values()
                        ]
                    }
                }
                
                await game_manager.broadcast_to_game(game_id, round_complete_message)
                logger.info(f"✅ Broadcasted round complete message for game {game_id}")
            else:
                logger.warning("❌ Game manager doesn't have broadcast_to_game method")
        except Exception as broadcast_error:
            logger.error(f"❌ Failed to broadcast round complete: {broadcast_error}")
        
        return {
            "success": True,
            "message": "Round judged successfully",
            "game_state": {
                "status": updated_game_state.status.value,
                "current_round": {
                    "round_number": updated_game_state.current_round.round_number,
                    "winner_id": str(updated_game_state.current_round.winner_id),  # Convert to string for frontend compatibility
                    "winning_card": updated_game_state.current_round.winning_card,
                    "phase": updated_game_state.current_round.phase.value
                } if updated_game_state.current_round else None,
                "players": [
                    {
                        "user_id": str(player.user_id),  # Convert to string for frontend compatibility
                        "username": player.username,
                        "score": player.score
                    }
                    for player in updated_game_state.players.values()
                ]
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to judge round: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to judge round: {str(e)}"
        )
