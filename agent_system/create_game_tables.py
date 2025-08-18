#!/usr/bin/env python3
"""
Create multiplayer game tables in existing database
"""

import os
import sys
from pathlib import Path

# Add current directory to path for imports
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️  python-dotenv not installed, trying to load env vars manually")

def create_game_tables():
    """Create the new multiplayer game tables"""
    
    print("🔧 Creating multiplayer game tables...")
    
    try:
        from models.database import create_database, get_session_local, User, Game, GamePlayer, GameRound, SubmittedCard
        
        # Get database URL
        database_url = os.getenv("DATABASE_URL", "postgresql://aslihangulseren@localhost:5432/cah_db")
        print(f"   Database URL: {database_url}")
        
        # Create tables
        print("\n📋 Creating tables...")
        engine = create_database(database_url)
        
        # Test the new tables
        print("\n🧪 Testing new tables...")
        SessionLocal = get_session_local(database_url)
        db = SessionLocal()
        
        try:
            # Test Game table
            games = db.query(Game).all()
            print(f"   ✅ Games table: {len(games)} games found")
            
            # Test GamePlayer table
            players = db.query(GamePlayer).all()
            print(f"   ✅ GamePlayer table: {len(players)} players found")
            
            # Test GameRound table
            rounds = db.query(GameRound).all()
            print(f"   ✅ GameRound table: {len(rounds)} rounds found")
            
            # Test SubmittedCard table
            cards = db.query(SubmittedCard).all()
            print(f"   ✅ SubmittedCard table: {len(cards)} cards found")
            
            print("\n🎉 All game tables created successfully!")
            
        except Exception as e:
            print(f"   ❌ Error testing tables: {e}")
        finally:
            db.close()
            
    except Exception as e:
        print(f"❌ Failed to create game tables: {e}")
        raise

if __name__ == "__main__":
    create_game_tables()
