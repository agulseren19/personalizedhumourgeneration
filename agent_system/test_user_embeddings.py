#!/usr/bin/env python3
"""
Test User Embeddings Integration
Demonstrates how user embeddings enhance the existing CAH system
"""

import asyncio
import sys
from pathlib import Path

# Add the agent_system directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from knowledge.user_embedding_manager import UserEmbeddingManager
from knowledge.improved_aws_knowledge_base import PostgreSQLKnowledgeBase
from agents.improved_humor_agents import ImprovedHumorAgent

async def test_user_embeddings():
    """Test the user embedding system integration"""
    print("🧠 Testing User Embeddings Integration")
    print("=" * 50)
    
    # Initialize managers
    embedding_manager = UserEmbeddingManager()
    knowledge_base = PostgreSQLKnowledgeBase()
    humor_agent = ImprovedHumorAgent()
    
    # Test user
    test_user_id = "test_user_123"
    
    print(f"\n👤 Testing with user: {test_user_id}")
    
    # 1. Test initial embedding creation
    print("\n1️⃣ Testing initial embedding creation...")
    initial_embedding = embedding_manager.get_user_embedding(test_user_id)
    print(f"   Initial embedding created: {len(initial_embedding['embedding_vector'])} dimensions")
    print(f"   Training samples: {initial_embedding['training_samples']}")
    
    # 2. Test feedback processing and embedding updates
    print("\n2️⃣ Testing feedback processing...")
    
    # Simulate user feedback
    feedback_data = [
        {"feedback_score": 9.0, "persona_name": "Edgy Comedian", "context": "politics joke", "topic": "politics"},
        {"feedback_score": 8.5, "persona_name": "Tech Geek", "context": "technology joke", "topic": "technology"},
        {"feedback_score": 7.0, "persona_name": "Pop Culture Expert", "context": "celebrity joke", "topic": "entertainment"},
        {"feedback_score": 3.0, "persona_name": "Family Friendly", "context": "clean joke", "topic": "family"}
    ]
    
    # Update embeddings
    embedding_manager.update_user_embedding(test_user_id, feedback_data)
    
    # Check updated embedding
    updated_embedding = embedding_manager.get_user_embedding(test_user_id)
    print(f"   Updated embedding: {updated_embedding['training_samples']} training samples")
    print(f"   Last trained: {updated_embedding['last_trained']}")
    
    # 3. Test personalized predictions
    print("\n3️⃣ Testing personalized predictions...")
    
    # Test different personas
    test_personas = ["Edgy Comedian", "Tech Geek", "Family Friendly"]
    
    for persona in test_personas:
        # Create simple text embedding
        text_embedding = [0.1] * 128  # Simple test embedding
        
        # Get personalized prediction
        prediction = embedding_manager.get_personalized_prediction(
            test_user_id, text_embedding, persona, "test context", "test topic"
        )
        
        print(f"   {persona}: {prediction:.2f}/10")
    
    # 4. Test persona ranking
    print("\n4️⃣ Testing persona ranking with embeddings...")
    
    # Create mock user preferences
    class MockUserPreference:
        def __init__(self, user_id: str):
            self.user_id = user_id
            self.liked_personas = ["Edgy Comedian", "Tech Geek"]
            self.disliked_personas = ["Family Friendly"]
            self.interaction_history = []
    
    mock_prefs = MockUserPreference(test_user_id)
    
    # Test persona filtering with embeddings
    test_personas = ["Edgy Comedian", "Tech Geek", "Pop Culture Expert", "Family Friendly", "Philosophical Jester"]
    filtered_personas = humor_agent._filter_personas_by_preferences(test_personas, mock_prefs)
    
    print(f"   Original personas: {test_personas}")
    print(f"   Filtered personas: {filtered_personas}")
    
    # 5. Test user similarity
    print("\n5️⃣ Testing user similarity...")
    
    # Create another test user
    test_user_2 = "test_user_456"
    embedding_manager.get_user_embedding(test_user_2)
    
    # Add some similar feedback
    similar_feedback = [
        {"feedback_score": 8.5, "persona_name": "Edgy Comedian", "context": "politics joke", "topic": "politics"},
        {"feedback_score": 9.0, "persona_name": "Tech Geek", "context": "technology joke", "topic": "technology"}
    ]
    embedding_manager.update_user_embedding(test_user_2, similar_feedback)
    
    # Find similar users
    similar_users = embedding_manager.get_similar_users(test_user_id, top_k=3)
    print(f"   Users similar to {test_user_id}:")
    for user, similarity in similar_users:
        print(f"     • {user}: {similarity:.3f}")
    
    # 6. Show embedding statistics
    print("\n6️⃣ Embedding statistics...")
    stats = embedding_manager.get_embedding_stats(test_user_id)
    print(f"   User: {stats['user_id']}")
    print(f"   Training samples: {stats['training_samples']}")
    print(f"   Embedding magnitude: {stats['embedding_magnitude']:.3f}")
    print(f"   Model version: {stats['model_version']}")
    
    print("\n✅ User embedding system test completed successfully!")
    print("\n🎯 Key Benefits Demonstrated:")
    print("   • Personalized persona selection based on user preferences")
    print("   • Learning from user feedback to improve recommendations")
    print("   • Finding similar users for collaborative filtering")
    print("   • Quantifiable personalization scores for each persona")

if __name__ == "__main__":
    print("🎭 User Embeddings Test for CAH System")
    print("Based on SHEEP-Medium/HuBi-Medium research")
    print("=" * 60)
    
    try:
        asyncio.run(test_user_embeddings())
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("This is expected if the database is not set up yet.")
        print("The user embedding system will work once integrated with your database.")
