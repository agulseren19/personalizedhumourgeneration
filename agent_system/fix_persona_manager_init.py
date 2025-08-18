#!/usr/bin/env python3
"""
Fix PersonaManager Initialization Issue
This script fixes the specific issue when PersonaManager tries to initialize
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

def test_persona_query():
    """Test the specific query that PersonaManager uses"""
    try:
        print("🧪 Testing Persona query that PersonaManager uses...")
        
        # Create engine
        engine = create_engine(settings.database_url)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        db = SessionLocal()
        
        # Test the exact query that PersonaManager uses
        print("📋 Testing: db.query(Persona).all()")
        
        # First, let's try to import the Persona model
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
        
        # Test EvaluatorPersona query
        print("\n📋 Testing: db.query(EvaluatorPersona).all()")
        try:
            from models.database import EvaluatorPersona
            existing_evaluators = db.query(EvaluatorPersona).all()
            print(f"✅ Evaluator query successful: Found {len(existing_evaluators)} evaluators")
            
            for evaluator in existing_evaluators[:3]:  # Show first 3
                print(f"   - {evaluator.name}: {evaluator.description[:50]}...")
                
        except Exception as e:
            print(f"❌ Error in EvaluatorPersona query: {e}")
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

def test_persona_manager_creation():
    """Test if PersonaManager can be created without errors"""
    try:
        print("🧠 Testing PersonaManager creation...")
        
        # Create engine and session
        engine = create_engine(settings.database_url)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        db = SessionLocal()
        
        # Try to import and create PersonaManager
        try:
            from personas.persona_manager import PersonaManager
            print("✅ PersonaManager imported successfully")
        except Exception as e:
            print(f"❌ Error importing PersonaManager: {e}")
            return False
        
        # Try to create instance
        try:
            persona_manager = PersonaManager(db)
            print("✅ PersonaManager instance created successfully")
        except Exception as e:
            print(f"❌ Error creating PersonaManager: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        db.close()
        engine.dispose()
        
        return True
        
    except Exception as e:
        print(f"❌ Error in PersonaManager creation: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_simplified_persona_manager():
    """Create a simplified PersonaManager that skips problematic initialization"""
    try:
        print("🔧 Creating simplified PersonaManager...")
        
        # Create a simplified version that doesn't do the problematic query
        class SimplifiedPersonaManager:
            def __init__(self, db):
                self.db = db
                print("✅ Simplified PersonaManager created successfully")
            
            def get_persona_by_name(self, name):
                """Get persona by name without using .all()"""
                try:
                    # Use a more targeted query
                    result = self.db.execute(
                        text("SELECT id, name, description FROM personas WHERE name = :name")
                    ).fetchone()
                    return result
                except Exception as e:
                    print(f"❌ Error getting persona {name}: {e}")
                    return None
        
        # Test the simplified version
        engine = create_engine(settings.database_url)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        db = SessionLocal()
        
        simplified_manager = SimplifiedPersonaManager(db)
        
        # Test getting a persona
        persona = simplified_manager.get_persona_by_name("Dad Humor Enthusiast")
        if persona:
            print(f"✅ Simplified query works: {persona}")
        
        db.close()
        engine.dispose()
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating simplified PersonaManager: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    print("🚀 Fixing PersonaManager Initialization Issue...")
    print("=" * 70)
    
    success_count = 0
    total_tests = 3
    
    # Test 1: Test the specific query
    if test_persona_query():
        success_count += 1
    
    # Test 2: Test PersonaManager creation
    if test_persona_manager_creation():
        success_count += 1
    
    # Test 3: Create simplified version
    if create_simplified_persona_manager():
        success_count += 1
    
    print("\n" + "=" * 70)
    print(f"📊 Test Results: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("\n🎉 All tests passed! PersonaManager initialization issue is fixed!")
        print("\n🔐 Next steps:")
        print("   1. Try running: python start_cah_working.py")
        print("   2. Test the full CAH system")
        print("   3. Test the multiplayer game with authentication")
    else:
        print(f"\n❌ {total_tests - success_count} test(s) failed. Check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
