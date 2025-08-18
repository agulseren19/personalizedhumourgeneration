#!/usr/bin/env python3
"""
Simple Feedback Test
Test the database feedback system
"""

import os
import sys
sys.path.append('/Users/aslihangulseren/Desktop/CAH')

from agent_system.models.database import UserFeedback, create_database, get_session_local
from datetime import datetime

def test_simple_feedback():
    """Test simple feedback storage and retrieval"""
    print("🧪 Testing Simple Feedback System")
    print("=" * 50)
    
    # Create test database
    test_db_url = "sqlite:///test_feedback.db"
    engine = create_database(test_db_url)
    SessionLocal = get_session_local(test_db_url)
    db = SessionLocal()
    
    try:
        # Test storing feedback
        test_feedback = UserFeedback(
            user_id="test_user_123",
            persona_name="Dark Humor Connoisseur",
            feedback_score=8.5,
            context="Test context",
            response_text="Test response",
            topic="general",
            audience="friends",
            created_at=datetime.now()
        )
        
        db.add(test_feedback)
        db.commit()
        
        print("✅ Feedback storage: PASSED")
        print(f"   - Stored user_id: {test_feedback.user_id}")
        print(f"   - Stored persona_name: {test_feedback.persona_name}")
        print(f"   - Stored score: {test_feedback.feedback_score}")
        
        # Test retrieval
        retrieved = db.query(UserFeedback).filter(
            UserFeedback.user_id == "test_user_123"
        ).first()
        
        if retrieved:
            print("✅ Feedback retrieval: PASSED")
            print(f"   - Retrieved user_id: {retrieved.user_id}")
            print(f"   - Retrieved persona_name: {retrieved.persona_name}")
            print(f"   - Retrieved score: {retrieved.feedback_score}")
        else:
            print("❌ Feedback retrieval: FAILED")
        
        # Test multiple feedback records
        additional_feedback = [
            UserFeedback(
                user_id="test_user_123",
                persona_name="Millennial Memer",
                feedback_score=6.0,
                context="Test context",
                response_text="Test response",
                topic="general",
                audience="friends",
                created_at=datetime.now()
            ),
            UserFeedback(
                user_id="test_user_123",
                persona_name="Corporate Humor Specialist",
                feedback_score=4.0,
                context="Test context",
                response_text="Test response",
                topic="general",
                audience="friends",
                created_at=datetime.now()
            )
        ]
        
        for feedback in additional_feedback:
            db.add(feedback)
        db.commit()
        
        # Test retrieving all feedback for user
        all_feedback = db.query(UserFeedback).filter(
            UserFeedback.user_id == "test_user_123"
        ).order_by(UserFeedback.created_at.desc()).all()
        
        print(f"✅ Multiple feedback retrieval: PASSED")
        print(f"   - Found {len(all_feedback)} feedback records")
        
        # Analyze preferences
        persona_scores = {}
        for feedback in all_feedback:
            persona = feedback.persona_name
            if persona not in persona_scores:
                persona_scores[persona] = []
            persona_scores[persona].append(feedback.feedback_score)
        
        print(f"   - Persona scores: {persona_scores}")
        
        # Find preferred personas (score > 6)
        preferred_personas = []
        for persona, scores in persona_scores.items():
            avg_score = sum(scores) / len(scores)
            if avg_score > 6:
                preferred_personas.append(persona)
        
        print(f"   - Preferred personas: {preferred_personas}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        db.close()
        # Clean up test database
        if os.path.exists("test_feedback.db"):
            os.remove("test_feedback.db")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    test_simple_feedback() 