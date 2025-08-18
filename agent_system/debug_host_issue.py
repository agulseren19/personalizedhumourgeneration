#!/usr/bin/env python3
"""
Debug script to identify host identification issues in multiplayer CAH games
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def debug_host_issue():
    """Debug the host identification issue"""
    try:
        # Import necessary modules
        from models.database import get_db, User
        from game.authenticated_multiplayer_cah import AuthenticatedMultiplayerCAHGame
        from agents.improved_humor_agents import ImprovedHumorOrchestrator
        from personas.persona_manager import PersonaManager
        
        print("🔍 Starting host identification debug...")
        
        # Get database session
        db = next(get_db())
        try:
            # Get all users
            print("\n📊 All Users in Database:")
            users = db.query(User).all()
            for user in users:
                print(f"  - ID: {user.id}, Email: {user.email}, Username: {user.username}")
            
            # Check if there are multiple users with similar emails
            print("\n🔍 Checking for email similarities:")
            emails = [user.email for user in users]
            for i, email1 in enumerate(emails):
                for j, email2 in enumerate(emails[i+1:], i+1):
                    if email1.split('@')[0] == email2.split('@')[0]:
                        print(f"  ⚠️  Similar usernames: {email1} and {email2}")
            
            # Initialize game manager
            print("\n🎮 Initializing game manager...")
            humor_orchestrator = ImprovedHumorOrchestrator()
            persona_manager = PersonaManager(db)
            game_manager = AuthenticatedMultiplayerCAHGame(humor_orchestrator, persona_manager)
            
            # Check existing games
            print(f"\n🎯 Existing games: {list(game_manager.games.keys())}")
            for game_id, game_state in game_manager.games.items():
                print(f"\n  Game {game_id}:")
                print(f"    Status: {game_state.status.value}")
                print(f"    Players: {len(game_state.players)}")
                for player_id, player in game_state.players.items():
                    print(f"      - {player.username} (ID: {player.user_id}) {'👑 HOST' if player.is_host else ''}")
            
            # Test host identification methods
            print("\n🧪 Testing host identification methods:")
            for game_id in game_manager.games:
                host_player = game_manager.get_host_player(game_id)
                if host_player:
                    print(f"  Game {game_id} host: {host_player.username} (ID: {host_player.user_id})")
                else:
                    print(f"  Game {game_id}: No host found!")
                
                # Test host check for each player
                game_state = game_manager.get_game_state(game_id)
                for player_id, player in game_state.players.items():
                    is_host = game_manager.is_user_host(game_id, player_id)
                    print(f"    Player {player.username} (ID: {player_id}) is host: {is_host}")
            
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Debug failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_host_issue())
