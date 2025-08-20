#!/usr/bin/env python3
"""
🚀 QUICK TEST SCRIPT - Hızlı Debug için
Her değişiklikten sonra çalıştır!
"""

import sys
import os

def test_imports():
    """Test tüm kritik importları"""
    print("🔍 Testing critical imports...")
    
    try:
        # Test API
        from api.cah_crewai_api import app
        print("✅ API import successful")
        
        # Test core agents
        from agents.improved_humor_agents import ImprovedHumorAgent, ImprovedHumorOrchestrator
        print("✅ Core agents OK")
        
        # Test database models
        from models.database import User, GamePlayer, GameRound, UserFeedback
        print("✅ Database models OK")
        
        # Test persona system
        from personas.persona_manager import PersonaManager
        print("✅ Persona system OK")
        
        # Test multiplayer
        from game.authenticated_multiplayer_cah import AuthenticatedMultiplayerCAHGame
        print("✅ Multiplayer system OK")
        
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_api_endpoints():
    """Test API endpoints (without starting server)"""
    print("\n🧪 Testing API endpoints...")
    
    try:
        from api.cah_crewai_api import app
        from fastapi.testclient import TestClient
        
        client = TestClient(app)
        
        # Test health endpoint
        response = client.get("/health")
        print(f"✅ Health endpoint: {response.status_code}")
        
        # Test root endpoint
        response = client.get("/")
        print(f"✅ Root endpoint: {response.status_code}")
        
        return True
        
    except Exception as e:
        print(f"❌ API test error: {e}")
        return False

def test_database_connection():
    """Test database connection (if DATABASE_URL available)"""
    print("\n🗄️ Testing database connection...")
    
    try:
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            print("⚠️ No DATABASE_URL - skipping database test")
            return True
            
        from sqlalchemy import create_engine, text
        engine = create_engine(database_url)
        
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            print("✅ Database connection OK")
            
        return True
        
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False

def main():
    """Ana test fonksiyonu"""
    print("🚀 CAH SYSTEM QUICK TEST")
    print("=" * 40)
    
    tests = [
        ("Import Tests", test_imports),
        ("API Endpoints", test_api_endpoints),
        ("Database Connection", test_database_connection)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}...")
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    print("\n" + "=" * 40)
    print(f"🎯 RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Ready to deploy!")
        return 0
    else:
        print("⚠️ Some tests failed. Fix before deploying!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
