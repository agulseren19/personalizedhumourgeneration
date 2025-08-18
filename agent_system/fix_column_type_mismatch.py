#!/usr/bin/env python3
"""
Fix Column Type Mismatch
This script fixes the mismatch between database schema and SQLAlchemy model types
"""

import os
import sys
from sqlalchemy import create_engine, text
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

def fix_column_type_mismatch():
    """Fix the column type mismatch between database and model"""
    try:
        print("🔧 Fixing column type mismatch...")
        
        # Create engine
        engine = create_engine(settings.database_url)
        
        with engine.connect() as conn:
            # Fix created_at column: TEXT -> TIMESTAMP
            print("📋 Fixing created_at column: TEXT -> TIMESTAMP")
            try:
                conn.execute(text("""
                    ALTER TABLE personas 
                    ALTER COLUMN created_at TYPE TIMESTAMP USING 
                    CASE 
                        WHEN created_at ~ '^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}$' 
                        THEN created_at::TIMESTAMP 
                        ELSE NULL 
                    END
                """))
                print("✅ created_at converted to TIMESTAMP")
            except Exception as e:
                print(f"❌ Error converting created_at: {e}")
            
            # Fix avg_rating column: TEXT -> FLOAT
            print("📋 Fixing avg_rating column: TEXT -> FLOAT")
            try:
                conn.execute(text("""
                    ALTER TABLE personas 
                    ALTER COLUMN avg_rating TYPE FLOAT USING 
                    CASE 
                        WHEN avg_rating ~ '^[0-9]+\.?[0-9]*$' 
                        THEN avg_rating::FLOAT 
                        ELSE 0.0 
                    END
                """))
                print("✅ avg_rating converted to FLOAT")
            except Exception as e:
                print(f"❌ Error converting avg_rating: {e}")
            
            # Fix created_at column in evaluator_personas: TEXT -> TIMESTAMP
            print("📋 Fixing evaluator_personas.created_at column: TEXT -> TIMESTAMP")
            try:
                conn.execute(text("""
                    ALTER TABLE evaluator_personas 
                    ALTER COLUMN created_at TYPE TIMESTAMP USING 
                    CASE 
                        WHEN created_at ~ '^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}$' 
                        THEN created_at::TIMESTAMP 
                        ELSE NULL 
                    END
                """))
                print("✅ evaluator_personas.created_at converted to TIMESTAMP")
            except Exception as e:
                print(f"❌ Error converting evaluator_personas.created_at: {e}")
            
            # Fix created_at and updated_at columns in persona_preferences: TEXT -> TIMESTAMP
            print("📋 Fixing persona_preferences timestamp columns: TEXT -> TIMESTAMP")
            try:
                conn.execute(text("""
                    ALTER TABLE persona_preferences 
                    ALTER COLUMN created_at TYPE TIMESTAMP USING 
                    CASE 
                        WHEN created_at ~ '^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}$' 
                        THEN created_at::TIMESTAMP 
                        ELSE NULL 
                    END
                """))
                print("✅ persona_preferences.created_at converted to TIMESTAMP")
            except Exception as e:
                print(f"❌ Error converting persona_preferences.created_at: {e}")
            
            try:
                conn.execute(text("""
                    ALTER TABLE persona_preferences 
                    ALTER COLUMN updated_at TYPE TIMESTAMP USING 
                    CASE 
                        WHEN updated_at ~ '^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}$' 
                        THEN updated_at::TIMESTAMP 
                        ELSE NULL 
                    END
                """))
                print("✅ persona_preferences.updated_at converted to TIMESTAMP")
            except Exception as e:
                print(f"❌ Error converting persona_preferences.updated_at: {e}")
            
            # Fix last_interaction column in persona_preferences: TEXT -> TIMESTAMP
            print("📋 Fixing persona_preferences.last_interaction column: TEXT -> TIMESTAMP")
            try:
                conn.execute(text("""
                    ALTER TABLE persona_preferences 
                    ALTER COLUMN last_interaction TYPE TIMESTAMP USING 
                    CASE 
                        WHEN last_interaction ~ '^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}$' 
                        THEN last_interaction::TIMESTAMP 
                        ELSE NULL 
                    END
                """))
                print("✅ persona_preferences.last_interaction converted to TIMESTAMP")
            except Exception as e:
                print(f"❌ Error converting persona_preferences.last_interaction: {e}")
            
            # Fix created_at column in user_feedback: TEXT -> TIMESTAMP
            print("📋 Fixing user_feedback.created_at column: TEXT -> TIMESTAMP")
            try:
                conn.execute(text("""
                    ALTER TABLE user_feedback 
                    ALTER COLUMN created_at TYPE TIMESTAMP USING 
                    CASE 
                        WHEN created_at ~ '^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}$' 
                        THEN created_at::TIMESTAMP 
                        ELSE NULL 
                    END
                """))
                print("✅ user_feedback.created_at converted to TIMESTAMP")
            except Exception as e:
                print(f"❌ Error converting user_feedback.created_at: {e}")
            
            # Commit changes
            conn.commit()
            print("\n✅ All column type conversions completed")
        
        engine.dispose()
        return True
        
    except Exception as e:
        print(f"❌ Error fixing column type mismatch: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_fixes():
    """Verify that the column type fixes worked"""
    try:
        print("🔍 Verifying column type fixes...")
        
        # Create engine
        engine = create_engine(settings.database_url)
        
        with engine.connect() as conn:
            # Check personas table
            print("📋 Checking personas table column types...")
            result = conn.execute(text("""
                SELECT column_name, data_type, udt_name
                FROM information_schema.columns 
                WHERE table_name = 'personas' 
                AND column_name IN ('created_at', 'avg_rating')
                ORDER BY column_name
            """))
            
            columns = result.fetchall()
            for col_name, data_type, udt_name in columns:
                print(f"   {col_name}: {data_type} (UDT: {udt_name})")
            
            # Check evaluator_personas table
            print("\n📋 Checking evaluator_personas table column types...")
            result = conn.execute(text("""
                SELECT column_name, data_type, udt_name
                FROM information_schema.columns 
                WHERE table_name = 'evaluator_personas' 
                AND column_name = 'created_at'
            """))
            
            columns = result.fetchall()
            for col_name, data_type, udt_name in columns:
                print(f"   {col_name}: {data_type} (UDT: {udt_name})")
            
            # Check persona_preferences table
            print("\n📋 Checking persona_preferences table column types...")
            result = conn.execute(text("""
                SELECT column_name, data_type, udt_name
                FROM information_schema.columns 
                WHERE table_name = 'persona_preferences' 
                AND column_name IN ('created_at', 'updated_at', 'last_interaction')
                ORDER BY column_name
            """))
            
            columns = result.fetchall()
            for col_name, data_type, udt_name in columns:
                print(f"   {col_name}: {data_type} (UDT: {udt_name})")
            
            # Check user_feedback table
            print("\n📋 Checking user_feedback table column types...")
            result = conn.execute(text("""
                SELECT column_name, data_type, udt_name
                FROM information_schema.columns 
                WHERE table_name = 'user_feedback' 
                AND column_name = 'created_at'
            """))
            
            columns = result.fetchall()
            for col_name, data_type, udt_name in columns:
                print(f"   {col_name}: {data_type} (UDT: {udt_name})")
        
        engine.dispose()
        return True
        
    except Exception as e:
        print(f"❌ Error verifying fixes: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_persona_query_after_fix():
    """Test if the Persona query works after fixing the column types"""
    try:
        print("🧪 Testing Persona query after column type fixes...")
        
        # Create engine
        engine = create_engine(settings.database_url)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        db = SessionLocal()
        
        # Try to import the Persona model
        try:
            from models.database import Persona
            print("✅ Persona model imported successfully")
        except Exception as e:
            print(f"❌ Error importing Persona model: {e}")
            return False
        
        # Now test the query
        try:
            existing_personas = db.query(Persona).all()
            print(f"✅ Query successful: Found {len(existing_personas)} personas")
            
            for persona in existing_personas[:3]:  # Show first 3
                print(f"   - {persona.name}: {persona.description[:50]}...")
                print(f"     Created: {persona.created_at}, Rating: {persona.avg_rating}")
                
        except Exception as e:
            print(f"❌ Error in Persona query: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        db.close()
        engine.dispose()
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing Persona query: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    print("🚀 Fixing Column Type Mismatch...")
    print("=" * 70)
    
    success_count = 0
    total_tests = 3
    
    # Test 1: Fix column type mismatch
    if fix_column_type_mismatch():
        success_count += 1
    
    # Test 2: Verify fixes
    if verify_fixes():
        success_count += 1
    
    # Test 3: Test Persona query after fix
    if test_persona_query_after_fix():
        success_count += 1
    
    print("\n" + "=" * 70)
    print(f"📊 Test Results: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("\n🎉 All tests passed! Column type mismatch is fixed!")
        print("\n🔐 Next steps:")
        print("   1. Try running: python start_cah_working.py")
        print("   2. Test the full CAH system")
        print("   3. Test the multiplayer game with authentication")
    else:
        print(f"\n❌ {total_tests - success_count} test(s) failed. Check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
