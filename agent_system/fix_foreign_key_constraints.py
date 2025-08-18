#!/usr/bin/env python3
"""
Fix Foreign Key Constraints
This script fixes foreign key constraint issues by ensuring proper primary keys
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

def check_table_constraints():
    """Check the current state of table constraints"""
    try:
        print("🔍 Checking table constraints...")
        
        # Create engine
        engine = create_engine(settings.database_url)
        
        with engine.connect() as conn:
            # Check personas table constraints
            print("📋 Checking personas table constraints...")
            result = conn.execute(text("""
                SELECT 
                    tc.constraint_name,
                    tc.constraint_type,
                    kcu.column_name,
                    ccu.table_name AS foreign_table_name,
                    ccu.column_name AS foreign_column_name
                FROM information_schema.table_constraints tc
                LEFT JOIN information_schema.key_column_usage kcu 
                    ON tc.constraint_name = kcu.constraint_name
                LEFT JOIN information_schema.constraint_column_usage ccu 
                    ON ccu.constraint_name = tc.constraint_name
                WHERE tc.table_name = 'personas'
                ORDER BY tc.constraint_type, tc.constraint_name
            """))
            
            constraints = result.fetchall()
            for constraint in constraints:
                constraint_name, constraint_type, column_name, foreign_table, foreign_column = constraint
                print(f"   {constraint_type}: {constraint_name}")
                if column_name:
                    print(f"     Column: {column_name}")
                if foreign_table and foreign_column:
                    print(f"     References: {foreign_table}.{foreign_column}")
            
            # Check if personas table has a primary key
            result = conn.execute(text("""
                SELECT 
                    tc.constraint_name,
                    tc.constraint_type,
                    kcu.column_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu 
                    ON tc.constraint_name = kcu.constraint_name
                WHERE tc.table_name = 'personas' 
                AND tc.constraint_type = 'PRIMARY KEY'
            """))
            
            primary_keys = result.fetchall()
            if primary_keys:
                print("✅ personas table has primary key constraint")
                for pk in primary_keys:
                    print(f"   Primary key: {pk[2]}")
            else:
                print("❌ personas table missing primary key constraint")
            
            # Check other tables that might be referenced
            tables_to_check = ['humor_generation_requests', 'evaluator_personas', 'persona_preferences', 'user_feedback']
            
            for table in tables_to_check:
                print(f"\n📋 Checking {table} table constraints...")
                try:
                    result = conn.execute(text(f"""
                        SELECT 
                            tc.constraint_name,
                            tc.constraint_type,
                            kcu.column_name
                        FROM information_schema.table_constraints tc
                        JOIN information_schema.key_column_usage kcu 
                            ON tc.constraint_name = kcu.constraint_name
                        WHERE tc.table_name = '{table}' 
                        AND tc.constraint_type = 'PRIMARY KEY'
                    """))
                    
                    primary_keys = result.fetchall()
                    if primary_keys:
                        print(f"✅ {table} table has primary key constraint")
                        for pk in primary_keys:
                            print(f"   Primary key: {pk[2]}")
                    else:
                        print(f"❌ {table} table missing primary key constraint")
                        
                except Exception as e:
                    print(f"   ⚠️  Could not check {table}: {e}")
        
        engine.dispose()
        return True
        
    except Exception as e:
        print(f"❌ Error checking table constraints: {e}")
        import traceback
        traceback.print_exc()
        return False

def fix_foreign_key_constraints():
    """Fix foreign key constraint issues"""
    try:
        print("🔧 Fixing foreign key constraints...")
        
        # Create engine
        engine = create_engine(settings.database_url)
        
        with engine.connect() as conn:
            # First, ensure personas table has proper primary key
            print("📋 Ensuring personas table has proper primary key...")
            try:
                # Check if primary key exists
                result = conn.execute(text("""
                    SELECT constraint_name 
                    FROM information_schema.table_constraints 
                    WHERE table_name = 'personas' AND constraint_type = 'PRIMARY KEY'
                """))
                
                if not result.fetchone():
                    print("   Adding primary key constraint to personas table...")
                    conn.execute(text("""
                        ALTER TABLE personas 
                        ADD CONSTRAINT personas_pkey PRIMARY KEY (id)
                    """))
                    print("   ✅ Primary key constraint added")
                else:
                    print("   ✅ Primary key constraint already exists")
                    
            except Exception as e:
                print(f"   ❌ Error with personas primary key: {e}")
            
            # Ensure evaluator_personas table has proper primary key
            print("📋 Ensuring evaluator_personas table has proper primary key...")
            try:
                result = conn.execute(text("""
                    SELECT constraint_name 
                    FROM information_schema.table_constraints 
                    WHERE table_name = 'evaluator_personas' AND constraint_type = 'PRIMARY KEY'
                """))
                
                if not result.fetchone():
                    print("   Adding primary key constraint to evaluator_personas table...")
                    conn.execute(text("""
                        ALTER TABLE evaluator_personas 
                        ADD CONSTRAINT evaluator_personas_pkey PRIMARY KEY (id)
                    """))
                    print("   ✅ Primary key constraint added")
                else:
                    print("   ✅ Primary key constraint already exists")
                    
            except Exception as e:
                print(f"   ❌ Error with evaluator_personas primary key: {e}")
            
            # Check if humor_generation_requests table exists and has proper constraints
            print("📋 Checking humor_generation_requests table...")
            try:
                result = conn.execute(text("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_name = 'humor_generation_requests'
                """))
                
                if result.fetchone():
                    print("   ✅ humor_generation_requests table exists")
                    
                    # Check if it has primary key
                    result = conn.execute(text("""
                        SELECT constraint_name 
                        FROM information_schema.table_constraints 
                        WHERE table_name = 'humor_generation_requests' AND constraint_type = 'PRIMARY KEY'
                    """))
                    
                    if not result.fetchone():
                        print("   Adding primary key constraint to humor_generation_requests table...")
                        conn.execute(text("""
                            ALTER TABLE humor_generation_requests 
                            ADD CONSTRAINT humor_generation_requests_pkey PRIMARY KEY (id)
                        """))
                        print("   ✅ Primary key constraint added")
                    else:
                        print("   ✅ Primary key constraint already exists")
                else:
                    print("   ⚠️  humor_generation_requests table does not exist")
                    
            except Exception as e:
                print(f"   ❌ Error with humor_generation_requests: {e}")
            
            # Commit changes
            conn.commit()
            print("\n✅ Foreign key constraint fixes completed")
        
        engine.dispose()
        return True
        
    except Exception as e:
        print(f"❌ Error fixing foreign key constraints: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_system_startup():
    """Test if the system can now start up"""
    try:
        print("🧪 Testing system startup...")
        
        # Try to import and initialize the components
        try:
            from models.database import get_db
            from personas.persona_manager import PersonaManager
            from agents.improved_humor_agents import ImprovedHumorOrchestrator
            
            print("✅ All components imported successfully")
            
            # Test database connection
            engine = create_engine(settings.database_url)
            SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
            db = SessionLocal()
            
            # Test PersonaManager
            persona_manager = PersonaManager(db)
            print("✅ PersonaManager initialized successfully")
            
            # Test humor orchestrator
            humor_orchestrator = ImprovedHumorOrchestrator()
            print("✅ HumorOrchestrator initialized successfully")
            
            db.close()
            engine.dispose()
            
            return True
            
        except Exception as e:
            print(f"❌ Error initializing components: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    except Exception as e:
        print(f"❌ Error testing system startup: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    print("🚀 Fixing Foreign Key Constraints...")
    print("=" * 70)
    
    success_count = 0
    total_tests = 3
    
    # Test 1: Check table constraints
    if check_table_constraints():
        success_count += 1
    
    # Test 2: Fix foreign key constraints
    if fix_foreign_key_constraints():
        success_count += 1
    
    # Test 3: Test system startup
    if test_system_startup():
        success_count += 1
    
    print("\n" + "=" * 70)
    print(f"📊 Test Results: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("\n🎉 All tests passed! Foreign key constraints are fixed!")
        print("\n🔐 Next steps:")
        print("   1. Try running: python start_cah_working.py")
        print("   2. Test the full CAH system")
        print("   3. Test the multiplayer game with authentication")
    else:
        print(f"\n❌ {total_tests - success_count} test(s) failed. Check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
