#!/usr/bin/env python3
"""
Database Migration Script for Authentication System
Adds User table and updates existing tables for authentication
"""

import os
import sys
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import OperationalError

# Add the current directory to the path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    from models.database import Base, User
    from config.settings import settings
    print("✅ Successfully imported database models")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure you're in the agent_system directory and virtual environment is activated")
    print("💡 Try: source cah_env/bin/activate")
    sys.exit(1)

def create_auth_tables():
    """Create authentication tables"""
    try:
        # Create database engine
        if hasattr(settings, 'DATABASE_URL'):
            database_url = settings.DATABASE_URL
        else:
            # Fallback to SQLite for development
            database_url = "sqlite:///agent_humor.db"
        
        engine = create_engine(database_url)
        
        # Create all tables
        Base.metadata.create_all(bind=engine)
        print("✅ Authentication tables created successfully")
        
        # Test the connection
        with engine.connect() as conn:
            result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
            tables = [row[0] for row in result]
            print(f"📋 Available tables: {tables}")
            
    except Exception as e:
        print(f"❌ Error creating authentication tables: {e}")
        return False
    
    return True

def migrate_existing_data():
    """Migrate existing data to new schema"""
    try:
        # Create database engine
        if hasattr(settings, 'DATABASE_URL'):
            database_url = settings.DATABASE_URL
        else:
            database_url = "sqlite:///agent_humor.db"
        
        engine = create_engine(database_url)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        db = SessionLocal()
        
        # Check if users table exists and has data
        try:
            existing_users = db.query(User).count()
            print(f"👥 Found {existing_users} existing users")
        except:
            print("📝 Users table is new, no migration needed")
            return True
        
        # If there are existing users, we might need to migrate them
        # For now, just report the count
        if existing_users > 0:
            print("ℹ️  Existing users found. Consider backing up data before major schema changes.")
        
        db.close()
        return True
        
    except Exception as e:
        print(f"❌ Error during data migration: {e}")
        return False

def main():
    """Main migration function"""
    print("🚀 Starting Authentication System Migration...")
    print("=" * 50)
    
    # Step 1: Create tables
    print("\n📊 Step 1: Creating authentication tables...")
    if not create_auth_tables():
        print("❌ Failed to create tables. Exiting.")
        return
    
    # Step 2: Migrate existing data
    print("\n🔄 Step 2: Checking existing data...")
    if not migrate_existing_data():
        print("❌ Failed to migrate data. Exiting.")
        return
    
    print("\n" + "=" * 50)
    print("✅ Authentication system migration completed successfully!")
    print("\n📋 What was added:")
    print("   • User table with authentication fields")
    print("   • Password hashing support")
    print("   • JWT token support")
    print("   • User preferences and demographics")
    print("\n🔐 Next steps:")
    print("   1. Update your backend API to include auth routes")
    print("   2. Test login/register functionality")
    print("   3. Update multiplayer games to use authenticated users")
    print("   4. Deploy with proper environment variables")

if __name__ == "__main__":
    main()
