#!/usr/bin/env python3
"""
Simple test script for multiplayer functionality (without database)
"""

import asyncio
import os

async def test_multiplayer_imports():
    """Test multiplayer imports"""
    print("🧪 Testing Multiplayer Imports...")
    
    try:
        # Test core multiplayer imports
        print("  🔍 Testing multiplayer game import...")
        from agent_system.game.multiplayer_cah import MultiplayerCAHGame
        print("  ✅ MultiplayerCAHGame imported successfully")
        
        print("  🔍 Testing authenticated multiplayer import...")
        from agent_system.game.authenticated_multiplayer_cah import AuthenticatedMultiplayerCAHGame
        print("  ✅ AuthenticatedMultiplayerCAHGame imported successfully")
        
        print("  🔍 Testing API routes import...")
        from agent_system.api.multiplayer_routes import router as multiplayer_router
        print("  ✅ Multiplayer routes imported successfully")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Import failed: {e}")
        return False

async def test_multiplayer_basic():
    """Test basic multiplayer functionality"""
    print("🧪 Testing Basic Multiplayer Game...")
    
    try:
        from agent_system.game.multiplayer_cah import MultiplayerCAHGame
        from agent_system.personas.persona_manager import PersonaManager
        
        # Create persona manager
        persona_manager = PersonaManager()
        
        # Create a simple game with persona manager
        game = MultiplayerCAHGame("test_game_id", persona_manager)
        
        print(f"  ✅ Game created with ID: {game.game_id}")
        print(f"  ✅ Game state: {game.state}")
        print(f"  ✅ Players: {len(game.players)}")
        print(f"  ✅ Persona manager: {type(persona_manager).__name__}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Multiplayer test failed: {e}")
        return False

if __name__ == "__main__":
    # Set API key
    os.environ["OPENAI_API_KEY"] = "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"
    
    async def run_tests():
        print("🎮 MULTIPLAYER TEST SUITE")
        print("=" * 40)
        
        # Test imports
        import_success = await test_multiplayer_imports()
        
        if import_success:
            # Test basic functionality
            basic_success = await test_multiplayer_basic()
            
            if basic_success:
                print("\n🎉 SUCCESS: All multiplayer tests passed!")
                return True
        
        print("\n❌ FAILED: Some multiplayer tests failed!")
        return False
    
    # Run tests
    result = asyncio.run(run_tests())
