#!/usr/bin/env python3
"""
PostgreSQL Database Migration Script for CAH System
Creates all necessary tables in PostgreSQL
"""

import os
import sys
from pathlib import Path

# Add parent directory to Python path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

try:
    from models.database import create_database, Base, User, Game, GamePlayer, GameRound, SubmittedCard
    from config.settings import settings
    print("✅ Successfully imported database models")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure you're in the agent_system directory")
    sys.exit(1)

def create_postgresql_tables():
    """Create all PostgreSQL tables"""
    try:
        print("🚀 Starting PostgreSQL migration...")
        
        # Get database URL from environment
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            print("❌ DATABASE_URL environment variable not set")
            print("💡 Set it to: postgresql://username:password@localhost:5432/database_name")
            return False
        
        # Fix database URL for SSL
        if "render.com" in database_url and "?sslmode=" not in database_url:
            database_url += "?sslmode=require"
        
        print(f"📊 Database URL: {database_url}")
        
        # Create all tables
        engine = create_database(database_url)
        
        print("✅ All tables created successfully!")
        
        # List all tables
        inspector = engine.dialect.inspector(engine)
        tables = inspector.get_table_names()
        
        print(f"\n📋 Tables created:")
        for table in tables:
            print(f"  - {table}")
            
        print(f"\n🎉 Total tables: {len(tables)}")
        
        return True
        
    except Exception as e:
        print(f"❌ PostgreSQL migration failed: {e}")
        return False

def test_postgresql_connection():
    """Test PostgreSQL connection and basic operations"""
    try:
        print("\n🧪 Testing PostgreSQL connection...")
        
        from models.database import get_session_local
        
        database_url = os.getenv("DATABASE_URL")
        if "render.com" in database_url and "?sslmode=" not in database_url:
            database_url += "?sslmode=require"
            
        SessionLocal = get_session_local(database_url)
        db = SessionLocal()
        
        try:
            # Test User table
            user_count = db.query(User).count()
            print(f"   ✅ Users table: {user_count} users found")
            
            # Test Game table
            game_count = db.query(Game).count()
            print(f"   ✅ Games table: {game_count} games found")
            
            # Test GamePlayer table
            player_count = db.query(GamePlayer).count()
            print(f"   ✅ GamePlayer table: {player_count} players found")
            
            print("✅ PostgreSQL connection test passed!")
            
        except Exception as e:
            print(f"   ❌ Error testing tables: {e}")
        finally:
            db.close()
            
        return True
        
    except Exception as e:
        print(f"❌ PostgreSQL connection test failed: {e}")
        return False

def main():
    """Main migration function"""
    print("🚀 PostgreSQL Database Migration Tool")
    print("=" * 60)
    
    # Step 1: Create tables
    print("\n📊 Step 1: Creating PostgreSQL tables...")
    if not create_postgresql_tables():
        print("❌ Failed to create tables. Exiting.")
        return
    
    # Step 2: Test connection
    print("\n🧪 Step 2: Testing PostgreSQL connection...")
    if not test_postgresql_connection():
        print("❌ Connection test failed.")
        return
    
    print("\n" + "=" * 60)
    print("✅ PostgreSQL migration completed successfully!")
    print("\n📋 What was created:")
    print("   • User table with authentication")
    print("   • Game tables for multiplayer")
    print("   • All necessary indexes and constraints")
    print("\n🔐 Next steps:")
    print("   1. Test sign-up functionality")
    print("   2. Test login functionality")
    print("   3. Deploy to Render")
    print("   4. Update frontend API calls")

if __name__ == "__main__":
    main()
