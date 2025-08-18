#!/usr/bin/env python3
"""
Fix Duplicate Data Issues
This script fixes duplicate data that's preventing primary key constraints
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

def check_duplicate_data():
    """Check for duplicate data in tables"""
    try:
        print("🔍 Checking for duplicate data...")
        
        # Create engine
        engine = create_engine(settings.database_url)
        
        with engine.connect() as conn:
            # Check personas table for duplicates
            print("📋 Checking personas table for duplicates...")
            result = conn.execute(text("""
                SELECT id, COUNT(*) as count
                FROM personas 
                GROUP BY id 
                HAVING COUNT(*) > 1
                ORDER BY id
            """))
            
            duplicates = result.fetchall()
            if duplicates:
                print(f"❌ Found {len(duplicates)} duplicate IDs in personas table:")
                for dup_id, count in duplicates:
                    print(f"   ID {dup_id}: {count} occurrences")
            else:
                print("✅ No duplicate IDs found in personas table")
            
            # Check evaluator_personas table for duplicates
            print("\n📋 Checking evaluator_personas table for duplicates...")
            result = conn.execute(text("""
                SELECT id, COUNT(*) as count
                FROM evaluator_personas 
                GROUP BY id 
                HAVING COUNT(*) > 1
                ORDER BY id
            """))
            
            duplicates = result.fetchall()
            if duplicates:
                print(f"❌ Found {len(duplicates)} duplicate IDs in evaluator_personas table:")
                for dup_id, count in duplicates:
                    print(f"   ID {dup_id}: {count} occurrences")
            else:
                print("✅ No duplicate IDs found in evaluator_personas table")
            
            # Check other tables
            tables_to_check = ['persona_preferences', 'user_feedback']
            for table in tables_to_check:
                print(f"\n📋 Checking {table} table for duplicates...")
                try:
                    result = conn.execute(text(f"""
                        SELECT id, COUNT(*) as count
                        FROM {table} 
                        GROUP BY id 
                        HAVING COUNT(*) > 1
                        ORDER BY id
                    """))
                    
                    duplicates = result.fetchall()
                    if duplicates:
                        print(f"❌ Found {len(duplicates)} duplicate IDs in {table} table:")
                        for dup_id, count in duplicates:
                            print(f"   ID {dup_id}: {count} occurrences")
                    else:
                        print(f"✅ No duplicate IDs found in {table} table")
                        
                except Exception as e:
                    print(f"   ⚠️  Could not check {table}: {e}")
        
        engine.dispose()
        return True
        
    except Exception as e:
        print(f"❌ Error checking duplicate data: {e}")
        import traceback
        traceback.print_exc()
        return False

def fix_duplicate_data():
    """Fix duplicate data by keeping only one row per ID"""
    try:
        print("🔧 Fixing duplicate data...")
        
        # Create engine
        engine = create_engine(settings.database_url)
        
        with engine.connect() as conn:
            # Fix personas table duplicates
            print("📋 Fixing personas table duplicates...")
            try:
                # Delete duplicates, keeping the first occurrence
                conn.execute(text("""
                    DELETE FROM personas 
                    WHERE id IN (
                        SELECT id 
                        FROM (
                            SELECT id, ROW_NUMBER() OVER (PARTITION BY id ORDER BY id) as rn
                            FROM personas
                        ) t 
                        WHERE t.rn > 1
                    )
                """))
                print("✅ personas table duplicates removed")
            except Exception as e:
                print(f"❌ Error removing personas duplicates: {e}")
            
            # Fix evaluator_personas table duplicates
            print("📋 Fixing evaluator_personas table duplicates...")
            try:
                conn.execute(text("""
                    DELETE FROM evaluator_personas 
                    WHERE id IN (
                        SELECT id 
                        FROM (
                            SELECT id, ROW_NUMBER() OVER (PARTITION BY id ORDER BY id) as rn
                            FROM evaluator_personas
                        ) t 
                        WHERE t.rn > 1
                    )
                """))
                print("✅ evaluator_personas table duplicates removed")
            except Exception as e:
                print(f"❌ Error removing evaluator_personas duplicates: {e}")
            
            # Fix persona_preferences table duplicates
            print("📋 Fixing persona_preferences table duplicates...")
            try:
                conn.execute(text("""
                    DELETE FROM persona_preferences 
                    WHERE id IN (
                        SELECT id 
                        FROM (
                            SELECT id, ROW_NUMBER() OVER (PARTITION BY id ORDER BY id) as rn
                            FROM persona_preferences
                        ) t 
                        WHERE t.rn > 1
                    )
                """))
                print("✅ persona_preferences table duplicates removed")
            except Exception as e:
                print(f"❌ Error removing persona_preferences duplicates: {e}")
            
            # Fix user_feedback table duplicates
            print("📋 Fixing user_feedback table duplicates...")
            try:
                conn.execute(text("""
                    DELETE FROM user_feedback 
                    WHERE id IN (
                        SELECT id 
                        FROM (
                            SELECT id, ROW_NUMBER() OVER (PARTITION BY id ORDER BY id) as rn
                            FROM user_feedback
                        ) t 
                        WHERE t.rn > 1
                    )
                """))
                print("✅ user_feedback table duplicates removed")
            except Exception as e:
                print(f"❌ Error removing user_feedback duplicates: {e}")
            
            # Commit changes
            conn.commit()
            print("\n✅ Duplicate data removal completed")
        
        engine.dispose()
        return True
        
    except Exception as e:
        print(f"❌ Error fixing duplicate data: {e}")
        import traceback
        traceback.print_exc()
        return False

def add_primary_key_constraints():
    """Add primary key constraints after fixing duplicates"""
    try:
        print("🔧 Adding primary key constraints...")
        
        # Create engine
        engine = create_engine(settings.database_url)
        
        with engine.connect() as conn:
            # Add primary key to personas table
            print("📋 Adding primary key to personas table...")
            try:
                conn.execute(text("""
                    ALTER TABLE personas 
                    ADD CONSTRAINT personas_pkey PRIMARY KEY (id)
                """))
                print("✅ personas primary key constraint added")
            except Exception as e:
                print(f"❌ Error adding personas primary key: {e}")
            
            # Add primary key to evaluator_personas table
            print("📋 Adding primary key to evaluator_personas table...")
            try:
                conn.execute(text("""
                    ALTER TABLE evaluator_personas 
                    ADD CONSTRAINT evaluator_personas_pkey PRIMARY KEY (id)
                """))
                print("✅ evaluator_personas primary key constraint added")
            except Exception as e:
                print(f"❌ Error adding evaluator_personas primary key: {e}")
            
            # Add primary key to persona_preferences table
            print("📋 Adding primary key to persona_preferences table...")
            try:
                conn.execute(text("""
                    ALTER TABLE persona_preferences 
                    ADD CONSTRAINT persona_preferences_pkey PRIMARY KEY (id)
                """))
                print("✅ persona_preferences primary key constraint added")
            except Exception as e:
                print(f"❌ Error adding persona_preferences primary key: {e}")
            
            # Add primary key to user_feedback table
            print("📋 Adding primary key to user_feedback table...")
            try:
                conn.execute(text("""
                    ALTER TABLE user_feedback 
                    ADD CONSTRAINT user_feedback_pkey PRIMARY KEY (id)
                """))
                print("✅ user_feedback primary key constraint added")
            except Exception as e:
                print(f"❌ Error adding user_feedback primary key: {e}")
            
            # Commit changes
            conn.commit()
            print("\n✅ Primary key constraints added")
        
        engine.dispose()
        return True
        
    except Exception as e:
        print(f"❌ Error adding primary key constraints: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_fixes():
    """Verify that the fixes worked"""
    try:
        print("🔍 Verifying fixes...")
        
        # Create engine
        engine = create_engine(settings.database_url)
        
        with engine.connect() as conn:
            # Check if primary keys exist
            tables_to_check = ['personas', 'evaluator_personas', 'persona_preferences', 'user_feedback']
            
            for table in tables_to_check:
                print(f"📋 Checking {table} table...")
                try:
                    result = conn.execute(text(f"""
                        SELECT constraint_name, constraint_type
                        FROM information_schema.table_constraints 
                        WHERE table_name = '{table}' AND constraint_type = 'PRIMARY KEY'
                    """))
                    
                    primary_keys = result.fetchall()
                    if primary_keys:
                        print(f"   ✅ {table} has primary key constraint")
                        for pk in primary_keys:
                            print(f"     {pk[0]}: {pk[1]}")
                    else:
                        print(f"   ❌ {table} missing primary key constraint")
                        
                except Exception as e:
                    print(f"   ❌ Error checking {table}: {e}")
            
            # Check for duplicates again
            print("\n📋 Checking for remaining duplicates...")
            for table in tables_to_check:
                try:
                    result = conn.execute(text(f"""
                        SELECT id, COUNT(*) as count
                        FROM {table} 
                        GROUP BY id 
                        HAVING COUNT(*) > 1
                    """))
                    
                    duplicates = result.fetchall()
                    if duplicates:
                        print(f"   ❌ {table} still has {len(duplicates)} duplicate IDs")
                    else:
                        print(f"   ✅ {table} has no duplicate IDs")
                        
                except Exception as e:
                    print(f"   ⚠️  Could not check {table}: {e}")
        
        engine.dispose()
        return True
        
    except Exception as e:
        print(f"❌ Error verifying fixes: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    print("🚀 Fixing Duplicate Data Issues...")
    print("=" * 70)
    
    success_count = 0
    total_tests = 4
    
    # Test 1: Check duplicate data
    if check_duplicate_data():
        success_count += 1
    
    # Test 2: Fix duplicate data
    if fix_duplicate_data():
        success_count += 1
    
    # Test 3: Add primary key constraints
    if add_primary_key_constraints():
        success_count += 1
    
    # Test 4: Verify fixes
    if verify_fixes():
        success_count += 1
    
    print("\n" + "=" * 70)
    print(f"📊 Test Results: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("\n🎉 All tests passed! Duplicate data issues are fixed!")
        print("\n🔐 Next steps:")
        print("   1. Try running: python start_cah_working.py")
        print("   2. Test the full CAH system")
        print("   3. Test the multiplayer game with authentication")
    else:
        print(f"\n❌ {total_tests - success_count} test(s) failed. Check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
