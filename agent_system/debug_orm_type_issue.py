#!/usr/bin/env python3
"""
Debug ORM Type Issue
This script debugs the specific SQLAlchemy ORM issue with "Unknown PG numeric type: 25"
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

def check_persona_table_structure():
    """Check the exact structure of the personas table"""
    try:
        print("🔍 Checking personas table structure in detail...")
        
        # Create engine
        engine = create_engine(settings.database_url)
        
        with engine.connect() as conn:
            # Get detailed column information
            result = conn.execute(text("""
                SELECT 
                    column_name,
                    data_type,
                    udt_name,
                    is_nullable,
                    column_default,
                    character_maximum_length,
                    numeric_precision,
                    numeric_scale,
                    datetime_precision
                FROM information_schema.columns 
                WHERE table_name = 'personas'
                ORDER BY ordinal_position
            """))
            
            columns = result.fetchall()
            print(f"📋 Found {len(columns)} columns in personas table:")
            
            for col in columns:
                col_name, data_type, udt_name, is_nullable, col_default, char_max_len, num_precision, num_scale, dt_precision = col
                print(f"   {col_name}:")
                print(f"     Data Type: {data_type}")
                print(f"     UDT Name: {udt_name}")
                print(f"     Nullable: {is_nullable}")
                print(f"     Default: {col_default}")
                print(f"     Char Max Length: {char_max_len}")
                print(f"     Numeric Precision: {num_precision}")
                print(f"     Numeric Scale: {num_scale}")
                print(f"     DateTime Precision: {dt_precision}")
                print()
        
        engine.dispose()
        return True
        
    except Exception as e:
        print(f"❌ Error checking table structure: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_persona_data_sample():
    """Check a sample of actual data from the personas table"""
    try:
        print("🔍 Checking sample data from personas table...")
        
        # Create engine
        engine = create_engine(settings.database_url)
        
        with engine.connect() as conn:
            # Get a sample row with all columns
            result = conn.execute(text("""
                SELECT * FROM personas LIMIT 1
            """))
            
            row = result.fetchone()
            if row:
                print("📋 Sample row data:")
                for i, value in enumerate(row):
                    print(f"   Column {i}: {type(value)} = {repr(value)}")
            else:
                print("⚠️  No data found in personas table")
        
        engine.dispose()
        return True
        
    except Exception as e:
        print(f"❌ Error checking sample data: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_persona_model_definition():
    """Check how the Persona model is defined"""
    try:
        print("🔍 Checking Persona model definition...")
        
        # Try to import the model
        try:
            from models.database import Persona
            print("✅ Persona model imported successfully")
            
            # Check the model's table info
            print(f"📋 Table name: {Persona.__tablename__}")
            print(f"📋 Columns:")
            
            for column in Persona.__table__.columns:
                print(f"   {column.name}: {column.type} (nullable: {column.nullable})")
                
                # Check for specific type issues
                if hasattr(column.type, 'item_type'):
                    print(f"     Item type: {column.type.item_type}")
                
                # Check if it's a JSON/JSONB type
                if 'JSON' in str(column.type):
                    print(f"     JSON type detected: {column.type}")
            
        except Exception as e:
            print(f"❌ Error importing Persona model: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking model definition: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_individual_column_queries():
    """Test querying individual columns to isolate the problematic one"""
    try:
        print("🧪 Testing individual column queries...")
        
        # Create engine
        engine = create_engine(settings.database_url)
        
        with engine.connect() as conn:
            # Test each column individually
            columns_to_test = [
                'id', 'name', 'description', 'demographics', 'personality_traits',
                'expertise_areas', 'prompt_template', 'is_active', 'created_at', 'avg_rating', 'total_generations'
            ]
            
            for col in columns_to_test:
                try:
                    print(f"   Testing column: {col}")
                    result = conn.execute(text(f"SELECT {col} FROM personas LIMIT 1"))
                    row = result.fetchone()
                    if row:
                        value = row[0]
                        print(f"     ✅ {col}: {type(value)} = {repr(value)[:50]}")
                    else:
                        print(f"     ⚠️  {col}: No data")
                except Exception as e:
                    print(f"     ❌ {col}: Error - {e}")
        
        engine.dispose()
        return True
        
    except Exception as e:
        print(f"❌ Error testing individual columns: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_sqlalchemy_orm_with_limited_columns():
    """Test SQLAlchemy ORM with limited columns to isolate the issue"""
    try:
        print("🧪 Testing SQLAlchemy ORM with limited columns...")
        
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
        
        # Test with just basic columns first
        try:
            print("📋 Testing with basic columns only...")
            result = db.query(Persona.id, Persona.name).all()
            print(f"✅ Basic query successful: Found {len(result)} personas")
        except Exception as e:
            print(f"❌ Basic query failed: {e}")
            return False
        
        # Test with JSONB columns
        try:
            print("📋 Testing with JSONB columns...")
            result = db.query(Persona.id, Persona.name, Persona.demographics).all()
            print(f"✅ JSONB query successful: Found {len(result)} personas")
        except Exception as e:
            print(f"❌ JSONB query failed: {e}")
            return False
        
        # Test with all columns
        try:
            print("📋 Testing with all columns...")
            result = db.query(Persona).all()
            print(f"✅ Full query successful: Found {len(result)} personas")
        except Exception as e:
            print(f"❌ Full query failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        db.close()
        engine.dispose()
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing ORM with limited columns: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    print("🚀 Debugging ORM Type Issue...")
    print("=" * 70)
    
    success_count = 0
    total_tests = 5
    
    # Test 1: Check table structure
    if check_persona_table_structure():
        success_count += 1
    
    # Test 2: Check sample data
    if check_persona_data_sample():
        success_count += 1
    
    # Test 3: Check model definition
    if check_persona_model_definition():
        success_count += 1
    
    # Test 4: Test individual columns
    if test_individual_column_queries():
        success_count += 1
    
    # Test 5: Test ORM with limited columns
    if test_sqlalchemy_orm_with_limited_columns():
        success_count += 1
    
    print("\n" + "=" * 70)
    print(f"📊 Test Results: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("\n🎉 All tests passed! ORM issue is identified and fixed!")
        print("\n🔐 Next steps:")
        print("   1. Try running: python start_cah_working.py")
        print("   2. Test the full CAH system")
        print("   3. Test the multiplayer game with authentication")
    else:
        print(f"\n❌ {total_tests - success_count} test(s) failed. Check the error messages above.")
        print("\n🔍 This should help us identify exactly which column is causing the ORM issue.")

if __name__ == "__main__":
    main()
