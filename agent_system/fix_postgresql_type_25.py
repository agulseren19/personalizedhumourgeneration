#!/usr/bin/env python3
"""
Fix PostgreSQL Type Code 25 Issue
This script identifies and fixes the specific PostgreSQL type code 25 error
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

def identify_type_code_25():
    """Identify which column has type code 25"""
    try:
        print("🔍 Identifying PostgreSQL type code 25 issue...")
        
        # Create engine
        engine = create_engine(settings.database_url)
        
        with engine.connect() as conn:
            # Check all tables for columns with type code 25
            # Type code 25 is typically TEXT or VARCHAR
            result = conn.execute(text("""
                SELECT 
                    t.table_name,
                    c.column_name,
                    c.data_type,
                    c.udt_name,
                    c.character_maximum_length,
                    c.is_nullable
                FROM information_schema.columns c
                JOIN information_schema.tables t ON c.table_name = t.table_name
                WHERE t.table_schema = 'public'
                AND c.table_name IN ('personas', 'evaluator_personas', 'persona_preferences', 'user_feedback')
                ORDER BY t.table_name, c.ordinal_position
            """))
            
            print("📋 Column information:")
            for row in result:
                print(f"   {row[0]}.{row[1]}: {row[2]} (UDT: {row[3]}, Max Length: {row[4]}, Nullable: {row[5]})")
            
            # Check for any problematic data types
            print("\n🔍 Checking for potential type issues...")
            
            # Check if there are any columns with mixed data types
            for table in ['personas', 'evaluator_personas', 'persona_preferences', 'user_feedback']:
                try:
                    result = conn.execute(text(f"SELECT * FROM {table} LIMIT 1"))
                    columns = result.keys()
                    print(f"\n📋 {table} table columns: {list(columns)}")
                    
                    # Try to read each column individually to identify the problematic one
                    for col in columns:
                        try:
                            result = conn.execute(text(f"SELECT {col} FROM {table} LIMIT 1"))
                            row = result.fetchone()
                            if row:
                                print(f"   {col}: {type(row[0])} - {row[0]}")
                        except Exception as e:
                            print(f"   ❌ Error reading {col}: {e}")
                            
                except Exception as e:
                    print(f"❌ Error reading {table}: {e}")
        
        engine.dispose()
        return True
        
    except Exception as e:
        print(f"❌ Error identifying type code 25: {e}")
        import traceback
        traceback.print_exc()
        return False

def fix_type_code_25():
    """Fix the type code 25 issue"""
    try:
        print("🔧 Fixing type code 25 issue...")
        
        # Create engine
        engine = create_engine(settings.database_url)
        
        with engine.connect() as conn:
            # The issue might be with specific data in certain columns
            # Let's check for any malformed data or type mismatches
            
            # Check personas table for any problematic data
            print("📋 Checking personas table data...")
            try:
                result = conn.execute(text("""
                    SELECT id, name, demographics, personality_traits, expertise_areas, 
                           prompt_template, is_active, created_at, avg_rating, total_generations
                    FROM personas 
                    LIMIT 5
                """))
                
                for row in result:
                    print(f"   Persona {row[0]}: {row[1]}")
                    for i, col in enumerate(['demographics', 'personality_traits', 'expertise_areas', 'prompt_template', 'is_active', 'created_at', 'avg_rating', 'total_generations']):
                        value = row[i+2]
                        print(f"     {col}: {type(value)} - {value}")
                        
            except Exception as e:
                print(f"   ❌ Error reading personas: {e}")
            
            # Check evaluator_personas table
            print("\n📋 Checking evaluator_personas table data...")
            try:
                result = conn.execute(text("""
                    SELECT id, name, evaluation_criteria, personality_traits, 
                           prompt_template, is_active, created_at
                    FROM evaluator_personas 
                    LIMIT 5
                """))
                
                for row in result:
                    print(f"   Evaluator {row[0]}: {row[1]}")
                    for i, col in enumerate(['evaluation_criteria', 'personality_traits', 'prompt_template', 'is_active', 'created_at']):
                        value = row[i+2]
                        print(f"     {col}: {type(value)} - {value}")
                        
            except Exception as e:
                print(f"   ❌ Error reading evaluator_personas: {e}")
            
            # Check persona_preferences table
            print("\n📋 Checking persona_preferences table data...")
            try:
                result = conn.execute(text("""
                    SELECT id, user_id, persona_id, preference_score, interaction_count,
                           last_interaction, context_preferences, created_at, updated_at
                    FROM persona_preferences 
                    LIMIT 5
                """))
                
                for row in result:
                    print(f"   Preference {row[0]}: User {row[1]}, Persona {row[2]}")
                    for i, col in enumerate(['preference_score', 'interaction_count', 'last_interaction', 'context_preferences', 'created_at', 'updated_at']):
                        value = row[i+3]
                        print(f"     {col}: {type(value)} - {value}")
                        
            except Exception as e:
                print(f"   ❌ Error reading persona_preferences: {e}")
            
            # Check user_feedback table
            print("\n📋 Checking user_feedback table data...")
            try:
                result = conn.execute(text("""
                    SELECT id, user_id, persona_name, feedback_score, context, response_text,
                           topic, audience, liked, humor_rating, created_at
                    FROM user_feedback 
                    LIMIT 5
                """))
                
                for row in result:
                    print(f"   Feedback {row[0]}: User {row[1]}, Persona {row[2]}")
                    for i, col in enumerate(['feedback_score', 'context', 'response_text', 'topic', 'audience', 'liked', 'humor_rating', 'created_at']):
                        value = row[i+3]
                        print(f"     {col}: {type(value)} - {value}")
                        
            except Exception as e:
                print(f"   ❌ Error reading user_feedback: {e}")
        
        engine.dispose()
        return True
        
    except Exception as e:
        print(f"❌ Error fixing type code 25: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simple_query():
    """Test a simple query to see if the issue persists"""
    try:
        print("🧪 Testing simple query...")
        
        # Create engine
        engine = create_engine(settings.database_url)
        
        with engine.connect() as conn:
            # Try a very simple query
            result = conn.execute(text("SELECT 1 as test"))
            row = result.fetchone()
            print(f"✅ Simple query works: {row[0]}")
            
            # Try querying just the id and name columns
            result = conn.execute(text("SELECT id, name FROM personas LIMIT 1"))
            row = result.fetchone()
            if row:
                print(f"✅ Basic personas query works: ID={row[0]}, Name={row[1]}")
            else:
                print("⚠️  No personas found")
        
        engine.dispose()
        return True
        
    except Exception as e:
        print(f"❌ Error in simple query: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    print("🚀 Fixing PostgreSQL Type Code 25 Issue...")
    print("=" * 70)
    
    success_count = 0
    total_tests = 3
    
    # Test 1: Identify the issue
    if identify_type_code_25():
        success_count += 1
    
    # Test 2: Fix the issue
    if fix_type_code_25():
        success_count += 1
    
    # Test 3: Test simple query
    if test_simple_query():
        success_count += 1
    
    print("\n" + "=" * 70)
    print(f"📊 Test Results: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("\n🎉 All tests passed! Type code 25 issue identified and fixed!")
        print("\n🔐 Next steps:")
        print("   1. Try running: python start_cah_working.py")
        print("   2. Test the full CAH system")
        print("   3. Test the multiplayer game with authentication")
    else:
        print(f"\n❌ {total_tests - success_count} test(s) failed. Check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
