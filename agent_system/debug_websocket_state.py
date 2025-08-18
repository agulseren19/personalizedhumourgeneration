#!/usr/bin/env python3
"""
Debug script to check WebSocket state and identify synchronization issues
"""

import asyncio
import logging
import json
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def debug_websocket_state():
    """Debug the current WebSocket state"""
    try:
        logger.info("🔍 Starting WebSocket state debugging...")
        
        # Check if we can import the main module
        try:
            from api.main import active_websocket_connections
            logger.info(f"✅ Successfully imported main.py WebSocket connections")
            logger.info(f"📊 Main.py has {len(active_websocket_connections)} games with WebSocket connections")
            
            for game_id, connections in active_websocket_connections.items():
                logger.info(f"🎮 Game {game_id}: {len(connections)} WebSocket connections")
                for user_id, websocket in connections.items():
                    logger.info(f"   - User {user_id}: {type(websocket)}")
                    
        except ImportError as e:
            logger.warning(f"⚠️ Could not import main.py: {e}")
        
        # Check if we can import the game manager
        try:
            from api.multiplayer_routes import get_game_manager
            
            # Create a mock database session
            class MockDB:
                def query(self, model):
                    return MockQuery()
                def add(self, item):
                    pass
                def commit(self):
                    pass
                def rollback(self):
                    pass
            
            class MockQuery:
                def filter(self, condition):
                    return self
                def first(self):
                    return None
                def all(self):
                    return []
            
            db = MockDB()
            game_manager = get_game_manager(db)
            
            logger.info(f"✅ Successfully imported game manager")
            logger.info(f"📊 Game manager has {len(game_manager.games)} games")
            
            # Check WebSocket connections in game manager
            for game_id, game_state in game_manager.games.items():
                connections = game_manager.get_websocket_connections(game_id)
                logger.info(f"🎮 Game {game_id}: {len(connections)} WebSocket connections in game manager")
                for user_id, websocket in connections.items():
                    logger.info(f"   - User {user_id}: {type(websocket)}")
                    
        except ImportError as e:
            logger.warning(f"⚠️ Could not import game manager: {e}")
        
        # Test WebSocket synchronization
        logger.info("🧪 Testing WebSocket synchronization...")
        
        # Create a test game
        from game.authenticated_multiplayer_cah import AuthenticatedMultiplayerCAHGame, GameStatus, RoundPhase
        
        # Create mock components
        class MockHumorOrchestrator:
            async def generate_and_evaluate_humor(self, request):
                return {'success': True, 'results': [{'generation': {'text': 'Mock card'}}]}
        
        class MockPersonaManager:
            pass
        
        # Create game manager
        test_game_manager = AuthenticatedMultiplayerCAHGame(
            humor_orchestrator=MockHumorOrchestrator(),
            persona_manager=MockPersonaManager()
        )
        
        # Create mock game
        from game.authenticated_multiplayer_cah import AuthenticatedPlayer, GameState, GameRound
        
        player1 = AuthenticatedPlayer(
            user_id=1,
            email="test1@test.com",
            username="TestPlayer1",
            is_host=True,
            is_judge=False,
            hand=["Test Card 1"],
            score=0
        )
        
        test_game = GameState(
            game_id="test_debug_001",
            players={1: player1},
            status=GameStatus.WAITING,
            current_round=None,
            round_history=[],
            settings={},
            created_at=datetime.now()
        )
        
        test_game_manager.games["test_debug_001"] = test_game
        
        # Test WebSocket connection
        class MockWebSocket:
            def __init__(self, user_id):
                self.user_id = user_id
                self.messages = []
            
            async def send_text(self, message):
                self.messages.append(message)
                logger.info(f"📨 Mock WebSocket {self.user_id} received: {message[:100]}...")
        
        ws1 = MockWebSocket(1)
        test_game_manager.add_websocket_connection("test_debug_001", 1, ws1)
        
        logger.info(f"✅ Test game created with WebSocket connection")
        
        # Test broadcasting
        test_message = {"type": "test", "message": "Test broadcast"}
        await test_game_manager.broadcast_to_game("test_debug_001", test_message)
        
        logger.info(f"✅ Broadcasting test completed. Messages sent: {len(ws1.messages)}")
        
        logger.info("🎉 WebSocket debugging completed!")
        
    except Exception as e:
        logger.error(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    asyncio.run(debug_websocket_state())
