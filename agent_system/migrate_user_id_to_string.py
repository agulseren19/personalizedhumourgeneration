#!/usr/bin/env python3
"""
Database Migration: Convert User.id from Integer to String
This script migrates existing User table to use String IDs instead of Integer IDs
"""

import os
import sys
from pathlib import Path

# Add the agent_system directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

def migrate_user_id_to_string():
    """Migrate User.id from Integer to String"""
    
    # Get database URL from environment
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("❌ DATABASE_URL environment variable not found")
        print("💡 Using default PostgreSQL URL for local testing")
        database_url = "postgresql://aslihangulseren@localhost:5432/cah_db"
    
    print(f"🔄 Connecting to database: {database_url[:50]}...")
    
    try:
        engine = create_engine(database_url)
        
        with engine.connect() as conn:
            print("✅ Database connection successful")
            
            # Start transaction
            trans = conn.begin()
            
            try:
                print("🔍 Checking current User table structure...")
                
                # Check if users table exists and has Integer ID
                result = conn.execute(text("""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_name = 'users' AND column_name = 'id'
                """))
                
                row = result.fetchone()
                if not row:
                    print("⚠️  Users table or id column not found")
                    return False
                
                current_type = row[1]
                print(f"📊 Current User.id type: {current_type}")
                
                if current_type == 'character varying':
                    print("✅ User.id is already String type - no migration needed")
                    return True
                
                if current_type != 'integer':
                    print(f"⚠️  Unexpected data type: {current_type}")
                    return False
                
                print("🚀 Starting migration from Integer to String...")
                
                # Step 1: Create backup table
                print("📋 Creating backup table...")
                conn.execute(text("""
                    CREATE TABLE users_backup AS SELECT * FROM users
                """))
                
                # Step 2: Drop foreign key constraints
                print("🔗 Dropping foreign key constraints...")
                
                # Drop foreign keys in dependent tables
                conn.execute(text("""
                    ALTER TABLE game_players DROP CONSTRAINT IF EXISTS game_players_user_id_fkey
                """))
                conn.execute(text("""
                    ALTER TABLE game_rounds DROP CONSTRAINT IF EXISTS game_rounds_judge_user_id_fkey
                """))
                
                # Step 3: Convert existing integer IDs to strings
                print("🔄 Converting User IDs to strings...")
                
                # Alter the users table
                conn.execute(text("""
                    ALTER TABLE users ALTER COLUMN id TYPE VARCHAR USING id::VARCHAR
                """))
                
                # Update dependent tables
                conn.execute(text("""
                    ALTER TABLE game_players ALTER COLUMN user_id TYPE VARCHAR USING user_id::VARCHAR
                """))
                conn.execute(text("""
                    ALTER TABLE game_rounds ALTER COLUMN judge_user_id TYPE VARCHAR USING judge_user_id::VARCHAR
                """))
                
                # Step 4: Recreate foreign key constraints
                print("🔗 Recreating foreign key constraints...")
                conn.execute(text("""
                    ALTER TABLE game_players 
                    ADD CONSTRAINT game_players_user_id_fkey 
                    FOREIGN KEY (user_id) REFERENCES users(id)
                """))
                conn.execute(text("""
                    ALTER TABLE game_rounds 
                    ADD CONSTRAINT game_rounds_judge_user_id_fkey 
                    FOREIGN KEY (judge_user_id) REFERENCES users(id)
                """))
                
                # Commit transaction
                trans.commit()
                print("✅ Migration completed successfully!")
                
                # Verify migration
                print("🔍 Verifying migration...")
                result = conn.execute(text("""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_name = 'users' AND column_name = 'id'
                """))
                
                row = result.fetchone()
                new_type = row[1]
                print(f"📊 New User.id type: {new_type}")
                
                if 'character' in new_type.lower() or 'varchar' in new_type.lower():
                    print("✅ Migration verification successful!")
                    return True
                else:
                    print(f"❌ Migration verification failed - type is still: {new_type}")
                    return False
                
            except Exception as e:
                print(f"❌ Migration failed: {e}")
                trans.rollback()
                print("🔄 Transaction rolled back")
                return False
                
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

if __name__ == "__main__":
    print("🎯 USER ID MIGRATION TOOL")
    print("=" * 50)
    
    success = migrate_user_id_to_string()
    
    if success:
        print("\n🎉 Migration completed successfully!")
        print("✅ User.id is now String type")
        print("✅ Foreign key constraints updated")
        print("✅ Backup table created (users_backup)")
    else:
        print("\n❌ Migration failed!")
        print("🔧 Please check the errors above and try again")
        sys.exit(1)
