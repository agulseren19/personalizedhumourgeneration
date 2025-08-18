#!/usr/bin/env python3
"""
Database Migration Script
Update user_feedback table to support string user_ids and persona_name
"""

import sqlite3
import os
from datetime import datetime

def migrate_database():
    """Migrate the database to the new schema"""
    print("🔄 Migrating Database Schema")
    print("=" * 50)
    
    db_path = "agent_system/agent_humor.db"
    if not os.path.exists(db_path):
        print(f"❌ Database not found: {db_path}")
        return False
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Check current schema
        cursor.execute("PRAGMA table_info(user_feedback);")
        columns = cursor.fetchall()
        column_names = [col[1] for col in columns]
        
        print(f"📋 Current columns: {column_names}")
        
        # Check if migration is needed
        needs_migration = (
            'user_id' in column_names and 
            'persona_name' not in column_names
        )
        
        if not needs_migration:
            print("✅ Database schema is up to date")
            return True
        
        print("🔄 Migration needed - updating schema...")
        
        # Create backup of existing data
        cursor.execute("SELECT * FROM user_feedback;")
        existing_data = cursor.fetchall()
        print(f"📊 Found {len(existing_data)} existing feedback records")
        
        # Create new table with updated schema
        cursor.execute("""
            CREATE TABLE user_feedback_new (
                id INTEGER PRIMARY KEY,
                user_id TEXT,
                generation_id INTEGER,
                persona_name TEXT,
                feedback_score REAL,
                context TEXT,
                response_text TEXT,
                topic TEXT,
                audience TEXT,
                liked BOOLEAN,
                humor_rating INTEGER,
                feedback_text_legacy TEXT,
                created_at DATETIME
            );
        """)
        
        # Copy existing data if any
        if existing_data:
            print("📋 Copying existing data...")
            for row in existing_data:
                # Map old columns to new schema
                # Assuming old schema: id, user_id, generation_id, liked, humor_rating, feedback_text, created_at
                if len(row) >= 7:
                    cursor.execute("""
                        INSERT INTO user_feedback_new 
                        (id, user_id, generation_id, liked, humor_rating, feedback_text_legacy, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, row)
        
        # Drop old table and rename new table
        cursor.execute("DROP TABLE user_feedback;")
        cursor.execute("ALTER TABLE user_feedback_new RENAME TO user_feedback;")
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_feedback_user_id ON user_feedback(user_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_feedback_persona_name ON user_feedback(persona_name);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_feedback_created_at ON user_feedback(created_at);")
        
        conn.commit()
        
        print("✅ Database migration completed successfully!")
        
        # Verify new schema
        cursor.execute("PRAGMA table_info(user_feedback);")
        new_columns = cursor.fetchall()
        print(f"📋 New columns: {[col[1] for col in new_columns]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        conn.rollback()
        return False
    
    finally:
        conn.close()

def test_migrated_schema():
    """Test the migrated schema"""
    print("\n🧪 Testing Migrated Schema")
    print("=" * 50)
    
    db_path = "agent_system/agent_humor.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Test inserting a record with new schema
        test_data = {
            'user_id': 'test_user_123',
            'persona_name': 'Dark Humor Connoisseur',
            'feedback_score': 8.5,
            'context': 'Test context',
            'response_text': 'Test response',
            'topic': 'general',
            'audience': 'friends',
            'created_at': datetime.now().isoformat()
        }
        
        cursor.execute("""
            INSERT INTO user_feedback 
            (user_id, persona_name, feedback_score, context, response_text, topic, audience, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            test_data['user_id'],
            test_data['persona_name'],
            test_data['feedback_score'],
            test_data['context'],
            test_data['response_text'],
            test_data['topic'],
            test_data['audience'],
            test_data['created_at']
        ))
        
        conn.commit()
        
        # Test retrieval
        cursor.execute("""
            SELECT user_id, persona_name, feedback_score 
            FROM user_feedback 
            WHERE user_id = ?
        """, (test_data['user_id'],))
        
        result = cursor.fetchone()
        
        if result:
            print("✅ Migration test: PASSED")
            print(f"   - Retrieved user_id: {result[0]}")
            print(f"   - Retrieved persona_name: {result[1]}")
            print(f"   - Retrieved score: {result[2]}")
        else:
            print("❌ Migration test: FAILED")
        
        # Clean up test data
        cursor.execute("DELETE FROM user_feedback WHERE user_id = ?", (test_data['user_id'],))
        conn.commit()
        
    except Exception as e:
        print(f"❌ Migration test failed: {e}")
    
    finally:
        conn.close()

def main():
    """Run database migration"""
    print("🔄 Database Migration Tool")
    print("=" * 60)
    
    # Run migration
    success = migrate_database()
    
    if success:
        # Test the migrated schema
        test_migrated_schema()
        print("\n✅ Database migration and testing complete!")
    else:
        print("\n❌ Database migration failed!")

if __name__ == "__main__":
    main() 