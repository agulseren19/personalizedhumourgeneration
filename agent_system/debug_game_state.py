#!/usr/bin/env python3
"""
Debug script to check current game state and host assignment
"""

import os
import sys
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    print("🔍 Checking current game state and host assignment...")
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Test database connection
    from models.database import get_db, User
    
    print("✅ Database models imported successfully")
    
    # Get database session
    db = next(get_db())
    try:
        print("✅ Database connection successful")
        
        # Check the specific users mentioned in the issue
        print("\n👥 Checking specific users:")
        
        # User ID 3 (aslihangulseren@hotmail.com)
        user3 = db.query(User).filter(User.id == 3).first()
        if user3:
            print(f"  User ID 3:")
            print(f"    Email: {user3.email}")
            print(f"    Username: {user3.username}")
            print(f"    Email prefix: {user3.email.split('@')[0]}")
        else:
            print("  User ID 3: Not found")
        
        # User ID 7 (isilgulseren@hotmail.com)
        user7 = db.query(User).filter(User.id == 7).first()
        if user7:
            print(f"  User ID 7:")
            print(f"    Email: {user7.email}")
            print(f"    Username: {user7.username}")
            print(f"    Email prefix: {user7.email.split('@')[0]}")
        else:
            print("  User ID 7: Not found")
        
        # Check if there are any active games
        print("\n🎮 Checking for active games...")
        try:
            from game.authenticated_multiplayer_cah import AuthenticatedMultiplayerCAHGame
            from agents.improved_humor_agents import ImprovedHumorOrchestrator
            from personas.persona_manager import PersonaManager
            
            humor_orchestrator = ImprovedHumorOrchestrator()
            persona_manager = PersonaManager(db)
            game_manager = AuthenticatedMultiplayerCAHGame(humor_orchestrator, persona_manager)
            
            print(f"🎯 Games in memory: {list(game_manager.games.keys())}")
            
            if game_manager.games:
                for game_id, game_state in game_manager.games.items():
                    print(f"\n  Game {game_id}:")
                    print(f"    Status: {game_state.status.value}")
                    print(f"    Players: {len(game_state.players)}")
                    print(f"    Created at: {game_state.created_at}")
                    
                    for player_id, player in game_state.players.items():
                        print(f"      Player {player_id}:")
                        print(f"        Email: {player.email}")
                        print(f"        Username: {player.username}")
                        print(f"        Is Host: {player.is_host}")
                        print(f"        Is Judge: {player.is_judge}")
                        print(f"        Score: {player.score}")
                        
                    # Check host specifically
                    host_player = game_manager.get_host_player(game_id)
                    if host_player:
                        print(f"    👑 Host: {host_player.username} (ID: {host_player.user_id})")
                    else:
                        print(f"    ⚠️  No host found!")
                        
            else:
                print("  No games currently in memory")
                
        except Exception as e:
            print(f"⚠️  Could not check games: {e}")
            import traceback
            traceback.print_exc()
            
    finally:
        db.close()
        print("✅ Database connection closed")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
