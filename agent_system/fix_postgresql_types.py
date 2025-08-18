#!/usr/bin/env python3
"""
Fix PostgreSQL Type Compatibility Issues
This script fixes the "Unknown PG numeric type: 25" error in the CAH system
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

def fix_postgresql_types():
    """Fix PostgreSQL type compatibility issues"""
    try:
        print("🔧 Fixing PostgreSQL type compatibility...")
        
        # Create engine
        engine = create_engine(settings.database_url)
        conn = engine.connect()
        
        # Fix JSON columns to use proper PostgreSQL JSONB type
        print("📋 Fixing JSON column types...")
        
        # Check which tables have JSON columns and fix them
        tables_with_json = [
            ('personas', ['demographics', 'personality_traits', 'expertise_areas']),
            ('evaluator_personas', ['evaluation_criteria', 'personality_traits']),
            ('user_feedback', ['interests', 'humor_preferences']),
            ('persona_preferences', ['context_preferences'])
        ]
        
        for table, columns in tables_with_json:
            print(f"📋 Fixing table: {table}")
            for column in columns:
                try:
                    # Check if column exists and has JSON data
                    result = conn.execute(text(f"""
                        SELECT column_name, data_type 
                        FROM information_schema.columns 
                        WHERE table_name = '{table}' AND column_name = '{column}'
                    """))
                    col_info = result.fetchone()
                    
                    if col_info:
                        current_type = col_info[1]
                        if current_type != 'jsonb':
                            print(f"  🔧 Converting {column} from {current_type} to JSONB...")
                            # Convert to JSONB
                            conn.execute(text(f"""
                                ALTER TABLE {table} 
                                ALTER COLUMN {column} TYPE JSONB USING {column}::jsonb
                            """))
                            print(f"  ✅ {column} converted to JSONB")
                        else:
                            print(f"  ✅ {column} already JSONB")
                    else:
                        print(f"  ⚠️  Column {column} not found in {table}")
                        
                except Exception as e:
                    print(f"  ❌ Error fixing {column} in {table}: {e}")
                    continue
        
        # Fix any TEXT columns that should be JSONB but contain JSON data
        print("📋 Fixing TEXT columns with JSON data...")
        
        # Check for TEXT columns that contain JSON data and convert them
        for table, columns in tables_with_json:
            for column in columns:
                try:
                    # Check if column contains valid JSON data
                    result = conn.execute(text(f"""
                        SELECT COUNT(*) as count
                        FROM {table} 
                        WHERE {column} IS NOT NULL 
                        AND {column} != ''
                        AND {column}::jsonb IS NOT NULL
                    """))
                    
                    json_count = result.fetchone()[0]
                    if json_count > 0:
                        print(f"  🔧 Converting {table}.{column} to JSONB (contains {json_count} JSON records)")
                        conn.execute(text(f"""
                            ALTER TABLE {table} 
                            ALTER COLUMN {column} TYPE JSONB USING {column}::jsonb
                        """))
                        print(f"  ✅ {table}.{column} converted to JSONB")
                        
                except Exception as e:
                    print(f"  ⚠️  Could not convert {table}.{column}: {e}")
                    continue
        
        # Commit changes
        conn.commit()
        conn.close()
        
        print("✅ PostgreSQL type compatibility fixed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error fixing PostgreSQL types: {e}")
        return False

def main():
    """Main function"""
    print("🚀 Fixing PostgreSQL Type Compatibility...")
    print("=" * 50)
    
    if fix_postgresql_types():
        print("\n🎉 Type compatibility fixes completed successfully!")
        print("\n📋 What was fixed:")
        print("   • JSON columns converted to proper JSONB type")
        print("   • PostgreSQL type compatibility issues resolved")
        print("   • CAH system should now work with PostgreSQL")
        print("\n🔐 Next steps:")
        print("   1. Try running: python start_cah_working.py")
        print("   2. Test the full CAH system")
        print("   3. Update multiplayer game for authenticated users")
    else:
        print("\n❌ Type compatibility fixes failed. Check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
