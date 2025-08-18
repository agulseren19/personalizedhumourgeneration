#!/usr/bin/env python3
"""
Database migration to fix user_id fields from Integer to String
This fixes the analytics bug where PersonaPreference.persona was not accessible
"""

import sqlite3
import os
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def migrate_database():
    """Migrate the database to fix user_id field types"""
    
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
        logger.error("Database file not found!")
        return False
    
    logger.info(f"🔄 Migrating database: {db_path}")
    
    # Create backup
    backup_path = f"{db_path}.backup"
    import shutil
    shutil.copy2(db_path, backup_path)
    logger.info(f"📦 Created backup: {backup_path}")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Check if tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        logger.info(f"📋 Found tables: {tables}")
        
        # 1. Fix PersonaPreference table
        if 'persona_preferences' in tables:
            logger.info("🔧 Migrating persona_preferences table...")
            
            # Create new table with correct schema
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS persona_preferences_new (
                    id INTEGER PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    persona_id INTEGER NOT NULL,
                    preference_score REAL,
                    interaction_count INTEGER DEFAULT 0,
                    last_interaction TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    context_preferences JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (persona_id) REFERENCES personas (id)
                )
            """)
            
            # Copy data, converting user_id to string
            cursor.execute("""
                INSERT OR IGNORE INTO persona_preferences_new 
                (id, user_id, persona_id, preference_score, interaction_count, 
                 last_interaction, context_preferences, created_at, updated_at)
                SELECT id, CAST(user_id AS TEXT), persona_id, preference_score, 
                       interaction_count, last_interaction, context_preferences,
                       created_at, updated_at
                FROM persona_preferences
            """)
            
            # Drop old table and rename new one
            cursor.execute("DROP TABLE persona_preferences")
            cursor.execute("ALTER TABLE persona_preferences_new RENAME TO persona_preferences")
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS ix_persona_preferences_user_id ON persona_preferences (user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS ix_persona_preferences_persona_id ON persona_preferences (persona_id)")
            
            logger.info("✅ persona_preferences table migrated")
        
        # 2. Fix HumorGenerationRequest table
        if 'humor_generation_requests' in tables:
            logger.info("🔧 Migrating humor_generation_requests table...")
            
            # Create new table with correct schema
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS humor_generation_requests_new (
                    id INTEGER PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    context TEXT,
                    target_audience TEXT,
                    humor_type TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Copy data, converting user_id to string
            cursor.execute("""
                INSERT OR IGNORE INTO humor_generation_requests_new 
                (id, user_id, context, target_audience, humor_type, created_at)
                SELECT id, CAST(user_id AS TEXT), context, target_audience, 
                       humor_type, created_at
                FROM humor_generation_requests
            """)
            
            # Drop old table and rename new one
            cursor.execute("DROP TABLE humor_generation_requests")
            cursor.execute("ALTER TABLE humor_generation_requests_new RENAME TO humor_generation_requests")
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS ix_humor_generation_requests_user_id ON humor_generation_requests (user_id)")
            
            logger.info("✅ humor_generation_requests table migrated")
        
        # Commit all changes
        conn.commit()
        logger.info("✅ Database migration completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        conn.rollback()
        return False
        
    finally:
        conn.close()

if __name__ == "__main__":
    migrate_database() 