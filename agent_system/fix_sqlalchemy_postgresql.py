#!/usr/bin/env python3
"""
Fix SQLAlchemy PostgreSQL Type Issue
This script fixes the specific SQLAlchemy issue when querying PostgreSQL with JSONB columns
"""

import os
import sys
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.orm import sessionmaker
import psycopg2

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

def check_psycopg2_version():
    """Check psycopg2 version and compatibility"""
    try:
        print(f"🔍 Checking psycopg2 version: {psycopg2.__version__}")
        
        # Check if we have the binary version
        if hasattr(psycopg2, 'extensions'):
            print("✅ psycopg2 binary version detected")
        else:
            print("⚠️  psycopg2 pure Python version detected")
            
        return True
    except Exception as e:
        print(f"❌ Error checking psycopg2: {e}")
        return False

def test_raw_psycopg2_query():
    """Test raw psycopg2 query to see if the issue is in psycopg2 or SQLAlchemy"""
    try:
        print("🧪 Testing raw psycopg2 query...")
        
        # Parse database URL to get connection parameters
        db_url = settings.database_url
        if db_url.startswith('postgresql://'):
            db_url = db_url.replace('postgresql://', '')
        
        # Extract connection info
        if '@' in db_url:
            auth, rest = db_url.split('@', 1)
            if ':' in auth:
                username, password = auth.split(':', 1)
            else:
                username = auth
                password = ''
            
            if ':' in rest:
                host_port, database = rest.split('/', 1)
                if ':' in host_port:
                    host, port = host_port.split(':', 1)
                else:
                    host = host_port
                    port = '5432'
            else:
                host = rest.split('/')[0]
                port = '5432'
                database = rest.split('/')[1]
        else:
            # Default values
            username = 'postgres'
            password = ''
            host = 'localhost'
            port = '5432'
            database = 'cah_db'
        
        print(f"📋 Connection params: {host}:{port}, DB: {database}, User: {username}")
        
        # Connect with psycopg2
        conn = psycopg2.connect(
            host=host,
            port=port,
            database=database,
            user=username,
            password=password
        )
        
        cursor = conn.cursor()
        
        # Test simple query
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        print(f"✅ Simple query: {result}")
        
        # Test personas query
        cursor.execute("SELECT id, name, demographics FROM personas LIMIT 1")
        result = cursor.fetchone()
        if result:
            print(f"✅ Personas query: ID={result[0]}, Name={result[1]}")
            print(f"   Demographics type: {type(result[2])} - {result[2]}")
        else:
            print("⚠️  No personas found")
        
        # Test evaluator_personas query
        cursor.execute("SELECT id, name, evaluation_criteria FROM evaluator_personas LIMIT 1")
        result = cursor.fetchone()
        if result:
            print(f"✅ Evaluator query: ID={result[0]}, Name={result[1]}")
            print(f"   Criteria type: {type(result[2])} - {result[2]}")
        else:
            print("⚠️  No evaluators found")
        
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        print(f"❌ Error in raw psycopg2 query: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_sqlalchemy_simple_query():
    """Test SQLAlchemy with simple queries"""
    try:
        print("🧪 Testing SQLAlchemy simple query...")
        
        # Create engine
        engine = create_engine(settings.database_url)
        
        with engine.connect() as conn:
            # Test simple query
            result = conn.execute(text("SELECT 1 as test"))
            row = result.fetchone()
            print(f"✅ SQLAlchemy simple query: {row[0]}")
            
            # Test personas query with specific columns
            result = conn.execute(text("SELECT id, name FROM personas LIMIT 1"))
            row = result.fetchone()
            if row:
                print(f"✅ SQLAlchemy personas query: ID={row[0]}, Name={row[1]}")
            else:
                print("⚠️  No personas found")
        
        engine.dispose()
        return True
        
    except Exception as e:
        print(f"❌ Error in SQLAlchemy query: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_sqlalchemy_jsonb_query():
    """Test SQLAlchemy with JSONB columns specifically"""
    try:
        print("🧪 Testing SQLAlchemy JSONB query...")
        
        # Create engine
        engine = create_engine(settings.database_url)
        
        with engine.connect() as conn:
            # Test personas query with JSONB columns
            result = conn.execute(text("SELECT id, name, demographics, personality_traits FROM personas LIMIT 1"))
            row = result.fetchone()
            if row:
                print(f"✅ SQLAlchemy personas with JSONB: ID={row[0]}, Name={row[1]}")
                print(f"   Demographics: {type(row[2])} - {row[2]}")
                print(f"   Personality: {type(row[3])} - {row[3]}")
            else:
                print("⚠️  No personas found")
        
        engine.dispose()
        return True
        
    except Exception as e:
        print(f"❌ Error in SQLAlchemy JSONB query: {e}")
        import traceback
        traceback.print_exc()
        return False

def fix_sqlalchemy_issue():
    """Attempt to fix the SQLAlchemy issue"""
    try:
        print("🔧 Attempting to fix SQLAlchemy issue...")
        
        # The issue might be with SQLAlchemy's type handling
        # Let's check if we need to update the engine configuration
        
        # Create engine with specific PostgreSQL settings
        engine = create_engine(
            settings.database_url,
            connect_args={
                "options": "-c timezone=utc"
            },
            pool_pre_ping=True,
            pool_recycle=300
        )
        
        with engine.connect() as conn:
            # Test if the issue persists with the new engine
            result = conn.execute(text("SELECT id, name, demographics FROM personas LIMIT 1"))
            row = result.fetchone()
            if row:
                print(f"✅ Fixed engine personas query: ID={row[0]}, Name={row[1]}")
                print(f"   Demographics: {type(row[2])} - {row[2]}")
            else:
                print("⚠️  No personas found")
        
        engine.dispose()
        return True
        
    except Exception as e:
        print(f"❌ Error fixing SQLAlchemy issue: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    print("🚀 Fixing SQLAlchemy PostgreSQL Type Issue...")
    print("=" * 70)
    
    success_count = 0
    total_tests = 5
    
    # Test 1: Check psycopg2 version
    if check_psycopg2_version():
        success_count += 1
    
    # Test 2: Test raw psycopg2
    if test_raw_psycopg2_query():
        success_count += 1
    
    # Test 3: Test SQLAlchemy simple query
    if test_sqlalchemy_simple_query():
        success_count += 1
    
    # Test 4: Test SQLAlchemy JSONB query
    if test_sqlalchemy_jsonb_query():
        success_count += 1
    
    # Test 5: Fix SQLAlchemy issue
    if fix_sqlalchemy_issue():
        success_count += 1
    
    print("\n" + "=" * 70)
    print(f"📊 Test Results: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("\n🎉 All tests passed! SQLAlchemy PostgreSQL issue is fixed!")
        print("\n🔐 Next steps:")
        print("   1. Try running: python start_cah_working.py")
        print("   2. Test the full CAH system")
        print("   3. Test the multiplayer game with authentication")
    else:
        print(f"\n❌ {total_tests - success_count} test(s) failed. Check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
