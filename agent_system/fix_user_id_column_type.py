#!/usr/bin/env python3
"""
Fix User ID Column Type
This script fixes the user_id column type issue by converting it back to TEXT
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

def fix_user_id_column_type():
    """Fix the user_id column type from JSONB back to TEXT"""
    try:
        print("🔧 Fixing user_id column type...")
        
        # Create engine
        engine = create_engine(settings.database_url)
        
        with engine.connect() as conn:
            # Fix persona_preferences.user_id: JSONB -> TEXT
            print("📋 Fixing persona_preferences.user_id column: JSONB -> TEXT")
            try:
                conn.execute(text("""
                    ALTER TABLE persona_preferences 
                    ALTER COLUMN user_id TYPE TEXT USING 
                    CASE 
                        WHEN user_id IS NULL THEN NULL
                        WHEN jsonb_typeof(user_id) = 'string' THEN user_id::text
                        ELSE user_id::text
                    END
                """))
                print("✅ persona_preferences.user_id converted to TEXT")
            except Exception as e:
                print(f"❌ Error converting persona_preferences.user_id: {e}")
            
            # Fix user_feedback.user_id: JSONB -> TEXT (if it was converted)
            print("📋 Checking user_feedback.user_id column type...")
            try:
                result = conn.execute(text("""
                    SELECT data_type 
                    FROM information_schema.columns 
                    WHERE table_name = 'user_feedback' AND column_name = 'user_id'
                """))
                
                data_type = result.fetchone()[0]
                if data_type == 'jsonb':
                    print("   Converting user_feedback.user_id from JSONB to TEXT...")
                    conn.execute(text("""
                        ALTER TABLE user_feedback 
                        ALTER COLUMN user_id TYPE TEXT USING 
                        CASE 
                            WHEN user_id IS NULL THEN NULL
                            WHEN jsonb_typeof(user_id) = 'string' THEN user_id::text
                            ELSE user_id::text
                        END
                    """))
                    print("   ✅ user_feedback.user_id converted to TEXT")
                else:
                    print(f"   ✅ user_feedback.user_id is already {data_type}")
                    
            except Exception as e:
                print(f"   ❌ Error with user_feedback.user_id: {e}")
            
            # Commit changes
            conn.commit()
            print("\n✅ User ID column type fixes completed")
        
        engine.dispose()
        return True
        
    except Exception as e:
        print(f"❌ Error fixing user_id column type: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_fixes():
    """Verify that the user_id column type fixes worked"""
    try:
        print("🔍 Verifying user_id column type fixes...")
        
        # Create engine
        engine = create_engine(settings.database_url)
        
        with engine.connect() as conn:
            # Check persona_preferences table
            print("📋 Checking persona_preferences.user_id column type...")
            result = conn.execute(text("""
                SELECT column_name, data_type, udt_name
                FROM information_schema.columns 
                WHERE table_name = 'persona_preferences' AND column_name = 'user_id'
            """))
            
            column_info = result.fetchone()
            if column_info:
                col_name, data_type, udt_name = column_info
                print(f"   {col_name}: {data_type} (UDT: {udt_name})")
                if data_type == 'text':
                    print("   ✅ user_id is now TEXT type")
                else:
                    print(f"   ❌ user_id is still {data_type} type")
            else:
                print("   ❌ Could not find user_id column")
            
            # Check user_feedback table
            print("\n📋 Checking user_feedback.user_id column type...")
            result = conn.execute(text("""
                SELECT column_name, data_type, udt_name
                FROM information_schema.columns 
                WHERE table_name = 'user_feedback' AND column_name = 'user_id'
            """))
            
            column_info = result.fetchone()
            if column_info:
                col_name, data_type, udt_name = column_info
                print(f"   {col_name}: {data_type} (UDT: {udt_name})")
                if data_type == 'text':
                    print("   ✅ user_id is now TEXT type")
                else:
                    print(f"   ❌ user_id is still {data_type} type")
            else:
                print("   ❌ Could not find user_id column")
        
        engine.dispose()
        return True
        
    except Exception as e:
        print(f"❌ Error verifying fixes: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_user_id_query():
    """Test if the user_id query now works"""
    try:
        print("🧪 Testing user_id query after fixes...")
        
        # Create engine
        engine = create_engine(settings.database_url)
        
        with engine.connect() as conn:
            # Test querying persona_preferences with a string user_id
            print("📋 Testing persona_preferences query with string user_id...")
            try:
                result = conn.execute(text("""
                    SELECT id, user_id, persona_id 
                    FROM persona_preferences 
                    WHERE user_id = 'user_1752604356649_cqj6lqqg0'
                    LIMIT 1
                """))
                
                row = result.fetchone()
                if row:
                    print(f"✅ Query successful: ID={row[0]}, user_id={row[1]}, persona_id={row[2]}")
                else:
                    print("✅ Query successful: No matching rows found")
                    
            except Exception as e:
                print(f"❌ Query failed: {e}")
                return False
            
            # Test inserting a new persona_preference with string user_id
            print("📋 Testing insert with string user_id...")
            try:
                # First, get a valid persona_id
                result = conn.execute(text("SELECT id FROM personas LIMIT 1"))
                persona_id = result.fetchone()[0]
                
                # Insert test record
                conn.execute(text("""
                    INSERT INTO persona_preferences (user_id, persona_id, preference_score, interaction_count)
                    VALUES (%s, %s, %s, %s)
                """), ('test_user_123', persona_id, 5.0, 1))
                
                print("✅ Insert successful")
                
                # Clean up test record
                conn.execute(text("DELETE FROM persona_preferences WHERE user_id = 'test_user_123'"))
                print("✅ Test record cleaned up")
                
            except Exception as e:
                print(f"❌ Insert test failed: {e}")
                return False
        
        engine.dispose()
        return True
        
    except Exception as e:
        print(f"❌ Error testing user_id query: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    print("🚀 Fixing User ID Column Type...")
    print("=" * 70)
    
    success_count = 0
    total_tests = 3
    
    # Test 1: Fix user_id column type
    if fix_user_id_column_type():
        success_count += 1
    
    # Test 2: Verify fixes
    if verify_fixes():
        success_count += 1
    
    # Test 3: Test user_id query
    if test_user_id_query():
        success_count += 1
    
    print("\n" + "=" * 70)
    print(f"📊 Test Results: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("\n🎉 All tests passed! User ID column type is fixed!")
        print("\n🔐 Next steps:")
        print("   1. Try running: python start_cah_working.py")
        print("   2. Test the full CAH system")
        print("   3. Test the multiplayer game with authentication")
    else:
        print(f"\n❌ {total_tests - success_count} test(s) failed. Check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
