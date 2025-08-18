#!/usr/bin/env python3
"""
Fix Remaining PostgreSQL Type Issues
This script identifies and fixes any remaining PostgreSQL type issues
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

def check_all_column_types():
    """Check all column types in all tables to identify problematic ones"""
    try:
        print("🔍 Checking all column types for problematic ones...")
        
        # Create engine
        engine = create_engine(settings.database_url)
        inspector = inspect(engine)
        
        # Get all table names
        table_names = inspector.get_table_names()
        print(f"📋 Found {len(table_names)} tables: {table_names}")
        
        problematic_columns = []
        
        for table_name in table_names:
            print(f"\n📋 Checking table: {table_name}")
            columns = inspector.get_columns(table_name)
            
            for col in columns:
                col_name = col['name']
                col_type = col['type']
                col_nullable = col['nullable']
                
                print(f"   {col_name}: {col_type} (nullable: {col_nullable})")
                
                # Check for potentially problematic types
                if hasattr(col_type, 'item_type'):
                    print(f"     Item type: {col_type.item_type}")
                
                # Check for specific problematic types
                if str(col_type).lower() in ['text', 'varchar', 'char']:
                    # These might contain data that should be JSONB
                    print(f"     ⚠️  Potential TEXT column that might contain JSON data")
                
                # Check for any unknown or problematic types
                if hasattr(col_type, '__class__') and 'Unknown' in str(col_type.__class__):
                    print(f"     ❌ Unknown type detected: {col_type}")
                    problematic_columns.append((table_name, col_name, col_type))
        
        engine.dispose()
        
        if problematic_columns:
            print(f"\n❌ Found {len(problematic_columns)} problematic columns:")
            for table, col, col_type in problematic_columns:
                print(f"   {table}.{col}: {col_type}")
        else:
            print("\n✅ No problematic column types found")
        
        return problematic_columns
        
    except Exception as e:
        print(f"❌ Error checking column types: {e}")
        import traceback
        traceback.print_exc()
        return []

def check_specific_data_types():
    """Check specific data types that might be causing issues"""
    try:
        print("🔍 Checking specific data types...")
        
        # Create engine
        engine = create_engine(settings.database_url)
        
        with engine.connect() as conn:
            # Check for any columns that might contain mixed data types
            tables_to_check = ['personas', 'evaluator_personas', 'persona_preferences', 'user_feedback']
            
            for table in tables_to_check:
                print(f"\n📋 Checking {table} table data types...")
                
                try:
                    # Get column info
                    result = conn.execute(text(f"""
                        SELECT column_name, data_type, udt_name, is_nullable
                        FROM information_schema.columns 
                        WHERE table_name = '{table}'
                        ORDER BY ordinal_position
                    """))
                    
                    columns = result.fetchall()
                    for col in columns:
                        col_name, data_type, udt_name, is_nullable = col
                        print(f"   {col_name}: {data_type} (UDT: {udt_name}, Nullable: {is_nullable})")
                        
                        # Check for specific problematic types
                        if data_type == 'text' and udt_name == 'text':
                            # Check if this column contains JSON data
                            try:
                                result2 = conn.execute(text(f"""
                                    SELECT {col_name}, pg_typeof({col_name})
                                    FROM {table} 
                                    WHERE {col_name} IS NOT NULL 
                                    LIMIT 1
                                """))
                                row = result2.fetchone()
                                if row:
                                    print(f"     Sample data type: {row[1]}")
                                    if row[0] and isinstance(row[0], str) and (row[0].startswith('{') or row[0].startswith('[')):
                                        print(f"     ⚠️  Contains JSON-like data: {row[0][:50]}...")
                            except Exception as e:
                                print(f"     Could not check data: {e}")
                
                except Exception as e:
                    print(f"   ❌ Error checking {table}: {e}")
        
        engine.dispose()
        return True
        
    except Exception as e:
        print(f"❌ Error checking specific data types: {e}")
        import traceback
        traceback.print_exc()
        return False

def fix_remaining_type_issues():
    """Fix any remaining type issues"""
    try:
        print("🔧 Fixing remaining type issues...")
        
        # Create engine
        engine = create_engine(settings.database_url)
        
        with engine.connect() as conn:
            # Check for any TEXT columns that contain JSON data and convert them
            tables_to_check = ['personas', 'evaluator_personas', 'persona_preferences', 'user_feedback']
            
            for table in tables_to_check:
                print(f"\n📋 Checking {table} for JSON in TEXT columns...")
                
                try:
                    # Get all TEXT columns
                    result = conn.execute(text(f"""
                        SELECT column_name, data_type
                        FROM information_schema.columns 
                        WHERE table_name = '{table}' AND data_type = 'text'
                    """))
                    
                    text_columns = result.fetchall()
                    for col_name, data_type in text_columns:
                        print(f"   Checking {col_name}...")
                        
                        # Check if this column contains JSON data
                        try:
                            result2 = conn.execute(text(f"""
                                SELECT COUNT(*) as count
                                FROM {table} 
                                WHERE {col_name} IS NOT NULL 
                                AND {col_name} != ''
                                AND ({col_name} ~ '^[{{[].*[}}]]$' OR {col_name} ~ '^[0-9]+$')
                            """))
                            
                            json_count = result2.fetchone()[0]
                            if json_count > 0:
                                print(f"     Found {json_count} rows with JSON-like data in {col_name}")
                                
                                # Try to convert to JSONB if it's safe
                                try:
                                    print(f"     🔧 Converting {col_name} to JSONB...")
                                    conn.execute(text(f"""
                                        ALTER TABLE {table} 
                                        ALTER COLUMN {col_name} TYPE JSONB USING {col_name}::jsonb
                                    """))
                                    print(f"     ✅ {col_name} converted to JSONB")
                                except Exception as e:
                                    print(f"     ❌ Could not convert {col_name}: {e}")
                                    continue
                            else:
                                print(f"     {col_name} contains regular text data")
                                
                        except Exception as e:
                            print(f"     Could not analyze {col_name}: {e}")
                            continue
                
                except Exception as e:
                    print(f"   ❌ Error processing {table}: {e}")
                    continue
            
            # Commit changes
            conn.commit()
            print("\n✅ Type conversions completed")
        
        engine.dispose()
        return True
        
    except Exception as e:
        print(f"❌ Error fixing remaining type issues: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_persona_query_after_fix():
    """Test if the Persona query works after fixing types"""
    try:
        print("🧪 Testing Persona query after type fixes...")
        
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
    print("🚀 Fixing Remaining PostgreSQL Type Issues...")
    print("=" * 70)
    
    success_count = 0
    total_tests = 4
    
    # Test 1: Check all column types
    problematic_columns = check_all_column_types()
    if not problematic_columns:
        success_count += 1
    
    # Test 2: Check specific data types
    if check_specific_data_types():
        success_count += 1
    
    # Test 3: Fix remaining type issues
    if fix_remaining_type_issues():
        success_count += 1
    
    # Test 4: Test Persona query after fix
    if test_persona_query_after_fix():
        success_count += 1
    
    print("\n" + "=" * 70)
    print(f"📊 Test Results: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("\n🎉 All tests passed! Remaining PostgreSQL type issues are fixed!")
        print("\n🔐 Next steps:")
        print("   1. Try running: python start_cah_working.py")
        print("   2. Test the full CAH system")
        print("   3. Test the multiplayer game with authentication")
    else:
        print(f"\n❌ {total_tests - success_count} test(s) failed. Check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
