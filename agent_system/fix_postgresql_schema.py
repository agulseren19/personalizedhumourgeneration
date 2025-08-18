#!/usr/bin/env python3
"""
Fix PostgreSQL Schema Constraints
This script fixes foreign key constraint issues after migrating from SQLite to PostgreSQL
"""

import os
import sys
from sqlalchemy import create_engine, text

# Add the current directory to the path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    from config.settings import settings
    print("✅ Successfully imported settings")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure you're in the agent_system directory and virtual environment is activated")
    sys.exit(1)

def fix_postgresql_schema():
    """Fix PostgreSQL schema constraints"""
    try:
        print("🔧 Fixing PostgreSQL schema constraints...")
        
        # Create engine
        engine = create_engine(settings.database_url)
        conn = engine.connect()
        
        # Create sequences for auto-incrementing IDs
        print("📋 Creating sequences...")
        conn.execute(text("CREATE SEQUENCE IF NOT EXISTS personas_id_seq START 13;"))
        conn.execute(text("CREATE SEQUENCE IF NOT EXISTS evaluator_personas_id_seq START 3;"))
        conn.execute(text("CREATE SEQUENCE IF NOT EXISTS user_feedback_id_seq START 3;"))
        conn.execute(text("CREATE SEQUENCE IF NOT EXISTS persona_preferences_id_seq START 2;"))
        
        # Check which tables exist and fix them
        print("📋 Checking existing tables...")
        result = conn.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';"))
        existing_tables = [row[0] for row in result.fetchall()]
        print(f"📋 Found tables: {existing_tables}")
        
        # Fix personas table - ensure id is properly set as primary key
        if 'personas' in existing_tables:
            print("📋 Fixing personas table...")
            conn.execute(text("""
                ALTER TABLE personas 
                ALTER COLUMN id SET NOT NULL,
                ALTER COLUMN id SET DEFAULT nextval('personas_id_seq');
            """))
        
        # Fix evaluator_personas table
        if 'evaluator_personas' in existing_tables:
            print("📋 Fixing evaluator_personas table...")
            conn.execute(text("""
                ALTER TABLE evaluator_personas 
                ALTER COLUMN id SET NOT NULL,
                ALTER COLUMN id SET DEFAULT nextval('evaluator_personas_id_seq');
            """))
        
        # Fix user_feedback table
        if 'user_feedback' in existing_tables:
            print("📋 Fixing user_feedback table...")
            conn.execute(text("""
                ALTER TABLE user_feedback 
                ALTER COLUMN id SET NOT NULL,
                ALTER COLUMN id SET DEFAULT nextval('user_feedback_id_seq');
            """))
        
        # Fix persona_preferences table
        if 'persona_preferences' in existing_tables:
            print("📋 Fixing persona_preferences table...")
            conn.execute(text("""
                ALTER TABLE persona_preferences 
                ALTER COLUMN id SET NOT NULL,
                ALTER COLUMN id SET DEFAULT nextval('persona_preferences_id_seq');
            """))
        
        # Commit changes
        conn.commit()
        conn.close()
        
        print("✅ PostgreSQL schema constraints fixed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error fixing PostgreSQL schema: {e}")
        return False

def main():
    """Main function"""
    print("🚀 Fixing PostgreSQL Schema Constraints...")
    print("=" * 50)
    
    if fix_postgresql_schema():
        print("\n🎉 Schema fixes completed successfully!")
        print("\n📋 What was fixed:")
        print("   • Primary key constraints on all tables")
        print("   • Sequence defaults for auto-incrementing IDs")
        print("   • Foreign key reference integrity")
        print("\n🔐 Next steps:")
        print("   1. Run: python migrate_auth_system.py")
        print("   2. Start your system: python start_cah_working.py")
    else:
        print("\n❌ Schema fixes failed. Check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
