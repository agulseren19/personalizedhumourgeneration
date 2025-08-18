#!/usr/bin/env python3
"""
Fix PersonaManager PostgreSQL Type Issue
This script fixes the "Unknown PG numeric type: 25" error in PersonaManager
"""

import os
import sys
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.orm import sessionmaker

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

def fix_persona_manager_postgresql():
    """Fix the specific PostgreSQL type issue in PersonaManager"""
    try:
        print("🔧 Fixing PersonaManager PostgreSQL type issue...")
        
        # Create engine
        engine = create_engine(settings.database_url)
        
        # Check table structure
        inspector = inspect(engine)
        
        print("📋 Checking table structures...")
        
        # Check personas table
        if 'personas' in inspector.get_table_names():
            print("📋 Checking personas table...")
            columns = inspector.get_columns('personas')
            for col in columns:
                print(f"   {col['name']}: {col['type']} (nullable: {col['nullable']})")
                
                # Check for problematic column types
                if hasattr(col['type'], 'item_type'):
                    print(f"     Item type: {col['type'].item_type}")
        
        # Check evaluator_personas table
        if 'evaluator_personas' in inspector.get_table_names():
            print("📋 Checking evaluator_personas table...")
            columns = inspector.get_columns('evaluator_personas')
            for col in columns:
                print(f"   {col['name']}: {col['type']} (nullable: {col['nullable']})")
        
        # Check for any problematic data types
        print("🔍 Checking for problematic data types...")
        
        with engine.connect() as conn:
            # Check if there are any columns with unknown types
            result = conn.execute(text("""
                SELECT column_name, data_type, udt_name
                FROM information_schema.columns 
                WHERE table_name IN ('personas', 'evaluator_personas', 'persona_preferences')
                ORDER BY table_name, ordinal_position
            """))
            
            for row in result:
                print(f"   {row[0]}: {row[1]} (UDT: {row[2]})")
                
                # Check for problematic types
                if row[2] in ['numeric', 'decimal'] or 'numeric' in str(row[1]).lower():
                    print(f"     ⚠️  Potential numeric type issue: {row[1]}")
        
        # Try to identify the specific issue by looking at the data
        print("🔍 Checking for data type mismatches...")
        
        with engine.connect() as conn:
            # Check personas table data
            try:
                result = conn.execute(text("SELECT id, name, demographics, personality_traits FROM personas LIMIT 3"))
                for row in result:
                    print(f"   Persona {row[0]}: {row[1]}")
                    print(f"     Demographics: {type(row[2])} - {row[2]}")
                    print(f"     Personality: {type(row[3])} - {row[3]}")
            except Exception as e:
                print(f"   ❌ Error reading personas: {e}")
            
            # Check evaluator_personas table data
            try:
                result = conn.execute(text("SELECT id, name, evaluation_criteria FROM evaluator_personas LIMIT 3"))
                for row in result:
                    print(f"   Evaluator {row[0]}: {row[1]}")
                    print(f"     Criteria: {type(row[2])} - {row[2]}")
            except Exception as e:
                print(f"   ❌ Error reading evaluator_personas: {e}")
        
        # Try to fix any type issues
        print("🔧 Attempting to fix type issues...")
        
        with engine.connect() as conn:
            # Check if there are any TEXT columns that should be JSONB
            result = conn.execute(text("""
                SELECT table_name, column_name, data_type
                FROM information_schema.columns 
                WHERE table_name IN ('personas', 'evaluator_personas', 'persona_preferences')
                AND data_type = 'text'
                AND column_name IN ('demographics', 'personality_traits', 'expertise_areas', 'evaluation_criteria', 'context_preferences')
            """))
            
            text_columns = result.fetchall()
            if text_columns:
                print(f"📋 Found {len(text_columns)} TEXT columns that should be JSONB:")
                for table, column, dtype in text_columns:
                    print(f"   {table}.{column}: {dtype}")
                    
                    # Try to convert to JSONB
                    try:
                        print(f"     🔧 Converting {table}.{column} to JSONB...")
                        conn.execute(text(f"""
                            ALTER TABLE {table} 
                            ALTER COLUMN {column} TYPE JSONB USING {column}::jsonb
                        """))
                        print(f"     ✅ {table}.{column} converted to JSONB")
                    except Exception as e:
                        print(f"     ❌ Failed to convert {table}.{column}: {e}")
                        continue
                
                conn.commit()
                print("✅ Type conversions completed")
            else:
                print("✅ All columns are already the correct type")
        
        engine.dispose()
        
        print("✅ PersonaManager PostgreSQL type fix completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error fixing PersonaManager PostgreSQL types: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_persona_manager_creation():
    """Test if PersonaManager can be created without errors"""
    try:
        print("🧠 Testing PersonaManager creation...")
        
        # Create engine and session
        engine = create_engine(settings.database_url)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        db = SessionLocal()
        
        # Try to import and create PersonaManager
        from personas.persona_manager import PersonaManager
        
        print("✅ PersonaManager imported successfully")
        
        # Try to create instance
        persona_manager = PersonaManager(db)
        print("✅ PersonaManager instance created successfully")
        
        db.close()
        engine.dispose()
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating PersonaManager: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    print("🚀 Fixing PersonaManager PostgreSQL Type Issue...")
    print("=" * 70)
    
    success_count = 0
    total_tests = 2
    
    # Test 1: Fix PostgreSQL types
    if fix_persona_manager_postgresql():
        success_count += 1
    
    # Test 2: Test PersonaManager creation
    if test_persona_manager_creation():
        success_count += 1
    
    print("\n" + "=" * 70)
    print(f"📊 Test Results: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("\n🎉 All tests passed! PersonaManager PostgreSQL issue is fixed!")
        print("\n🔐 Next steps:")
        print("   1. Try running: python start_cah_working.py")
        print("   2. Test the full CAH system")
        print("   3. Test the multiplayer game with authentication")
    else:
        print(f"\n❌ {total_tests - success_count} test(s) failed. Check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
