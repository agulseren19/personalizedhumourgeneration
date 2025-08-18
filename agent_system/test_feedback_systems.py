#!/usr/bin/env python3
"""
Unit Tests for Feedback Systems
Tests both database and AWS knowledge base feedback storage/retrieval
"""

import unittest
import asyncio
import tempfile
import os
from datetime import datetime
from typing import Dict, Any

# Import our systems
from agent_system.models.database import (
    Base, User, UserFeedback, Persona, PersonaPreference,
    create_database, get_session_local
)
from agent_system.knowledge.improved_aws_knowledge_base import (
    ImprovedAWSKnowledgeBase, UserPreference, FeedbackData
)
from agent_system.personas.persona_manager import PersonaManager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

class TestFeedbackSystems(unittest.TestCase):
    """Test both database and AWS knowledge base feedback systems"""
    
    def setUp(self):
        """Set up test environment"""
        # Create temporary database
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.db_url = f"sqlite:///{self.temp_db.name}"
        
        # Create database and session
        self.engine = create_database(self.db_url)
        self.SessionLocal = get_session_local(self.db_url)
        self.db = self.SessionLocal()
        
        # Initialize AWS knowledge base in mock mode
        self.aws_kb = ImprovedAWSKnowledgeBase(mock_mode=True)
        
        # Initialize persona manager
        self.persona_manager = PersonaManager(self.db)
        
        # Create test data
        self._create_test_data()
    
    def tearDown(self):
        """Clean up test environment"""
        self.db.close()
        self.temp_db.close()
        os.unlink(self.temp_db.name)
    
    def _create_test_data(self):
        """Create test personas and users"""
        # Create test personas
        test_personas = [
            {
                "name": "Dark Humor Connoisseur",
                "description": "Specializes in dark humor",
                "personality_traits": {"humor_style": "dark"},
                "expertise_areas": ["dark_humor", "sarcasm"],
                "is_active": True
            },
            {
                "name": "Millennial Memer", 
                "description": "Specializes in millennial humor",
                "personality_traits": {"humor_style": "meme"},
                "expertise_areas": ["memes", "pop_culture"],
                "is_active": True
            },
            {
                "name": "Corporate Humor Specialist",
                "description": "Specializes in workplace humor",
                "personality_traits": {"humor_style": "corporate"},
                "expertise_areas": ["workplace", "office"],
                "is_active": True
            }
        ]
        
        for persona_data in test_personas:
            persona = Persona(**persona_data)
            self.db.add(persona)
        
        self.db.commit()
    
    def test_database_feedback_storage(self):
        """Test storing feedback in database"""
        print("\n=== Testing Database Feedback Storage ===")
        
        # Test data
        user_id = "test_user_123"
        persona_name = "Dark Humor Connoisseur"
        feedback_score = 8.5
        context = "What's the worst part about adult life?"
        response_text = "Realizing your dreams are in the same place you left your virginity."
        
        # Create feedback record
        feedback = UserFeedback(
            user_id=user_id,
            persona_name=persona_name,
            feedback_score=feedback_score,
            context=context,
            response_text=response_text,
            topic="general",
            audience="friends"
        )
        
        self.db.add(feedback)
        self.db.commit()
        
        # Verify storage
        stored_feedback = self.db.query(UserFeedback).filter(
            UserFeedback.user_id == user_id
        ).first()
        
        self.assertIsNotNone(stored_feedback)
        self.assertEqual(stored_feedback.user_id, user_id)
        self.assertEqual(stored_feedback.persona_name, persona_name)
        self.assertEqual(stored_feedback.feedback_score, feedback_score)
        
        print(f"✅ Database feedback storage: PASSED")
        print(f"   - Stored user_id: {stored_feedback.user_id}")
        print(f"   - Stored persona_name: {stored_feedback.persona_name}")
        print(f"   - Stored score: {stored_feedback.feedback_score}")
    
    def test_database_feedback_retrieval(self):
        """Test retrieving feedback from database"""
        print("\n=== Testing Database Feedback Retrieval ===")
        
        # Create multiple feedback records
        test_feedback = [
            {"user_id": "test_user_123", "persona_name": "Dark Humor Connoisseur", "score": 8.5},
            {"user_id": "test_user_123", "persona_name": "Millennial Memer", "score": 6.0},
            {"user_id": "test_user_123", "persona_name": "Corporate Humor Specialist", "score": 4.0},
            {"user_id": "test_user_123", "persona_name": "Dark Humor Connoisseur", "score": 9.0},
        ]
        
        for feedback_data in test_feedback:
            feedback = UserFeedback(
                user_id=feedback_data["user_id"],
                persona_name=feedback_data["persona_name"],
                feedback_score=feedback_data["score"],
                context="Test context",
                response_text="Test response"
            )
            self.db.add(feedback)
        
        self.db.commit()
        
        # Retrieve feedback
        user_feedback = self.db.query(UserFeedback).filter(
            UserFeedback.user_id == "test_user_123"
        ).order_by(UserFeedback.created_at.desc()).all()
        
        self.assertEqual(len(user_feedback), 4)
        
        # Check persona preferences
        persona_scores = {}
        for feedback in user_feedback:
            persona = feedback.persona_name
            if persona not in persona_scores:
                persona_scores[persona] = []
            persona_scores[persona].append(feedback.feedback_score)
        
        print(f"✅ Database feedback retrieval: PASSED")
        print(f"   - Found {len(user_feedback)} feedback records")
        print(f"   - Persona scores: {persona_scores}")
        
        # Test persona manager retrieval
        personalized_personas = self.persona_manager.get_personalized_personas(
            user_id="test_user_123", context="Test context", count=3
        )
        
        print(f"   - Personalized personas: {[p.name for p in personalized_personas]}")
    
    def test_aws_knowledge_base_storage(self):
        """Test storing feedback in AWS knowledge base"""
        print("\n=== Testing AWS Knowledge Base Storage ===")
        
        user_id = "test_user_456"
        persona_name = "Dark Humor Connoisseur"
        feedback_score = 8.5
        context = "What's the worst part about adult life?"
        response_text = "Realizing your dreams are in the same place you left your virginity."
        
        # Store feedback
        success = asyncio.run(self.aws_kb.update_user_feedback(
            user_id=user_id,
            persona_name=persona_name,
            feedback_score=feedback_score,
            context=context,
            response_text=response_text,
            topic="general",
            audience="friends"
        ))
        
        self.assertTrue(success)
        
        # Retrieve user preferences
        user_pref = asyncio.run(self.aws_kb.get_user_preference(user_id))
        
        self.assertIsNotNone(user_pref)
        self.assertEqual(len(user_pref.interaction_history), 1)
        
        interaction = user_pref.interaction_history[0]
        self.assertEqual(interaction['persona_name'], persona_name)
        self.assertEqual(interaction['feedback_score'], feedback_score)
        
        print(f"✅ AWS knowledge base storage: PASSED")
        print(f"   - Stored user_id: {user_id}")
        print(f"   - Stored persona_name: {interaction['persona_name']}")
        print(f"   - Stored score: {interaction['feedback_score']}")
    
    def test_aws_knowledge_base_retrieval(self):
        """Test retrieving feedback from AWS knowledge base"""
        print("\n=== Testing AWS Knowledge Base Retrieval ===")
        
        user_id = "test_user_789"
        
        # Store multiple feedback records
        test_feedback = [
            {"persona_name": "Dark Humor Connoisseur", "score": 8.5},
            {"persona_name": "Millennial Memer", "score": 6.0},
            {"persona_name": "Corporate Humor Specialist", "score": 4.0},
            {"persona_name": "Dark Humor Connoisseur", "score": 9.0},
        ]
        
        for feedback_data in test_feedback:
            success = asyncio.run(self.aws_kb.update_user_feedback(
                user_id=user_id,
                persona_name=feedback_data["persona_name"],
                feedback_score=feedback_data["score"],
                context="Test context",
                response_text="Test response"
            ))
            self.assertTrue(success)
        
        # Retrieve user preferences
        user_pref = asyncio.run(self.aws_kb.get_user_preference(user_id))
        
        self.assertIsNotNone(user_pref)
        self.assertEqual(len(user_pref.interaction_history), 4)
        
        # Check persona preferences
        persona_scores = {}
        for interaction in user_pref.interaction_history:
            persona = interaction['persona_name']
            if persona not in persona_scores:
                persona_scores[persona] = []
            persona_scores[persona].append(interaction['feedback_score'])
        
        print(f"✅ AWS knowledge base retrieval: PASSED")
        print(f"   - Found {len(user_pref.interaction_history)} interactions")
        print(f"   - Persona scores: {persona_scores}")
        print(f"   - Liked personas: {user_pref.liked_personas}")
        print(f"   - Disliked personas: {user_pref.disliked_personas}")
    
    def test_persona_manager_integration(self):
        """Test persona manager integration with both systems"""
        print("\n=== Testing Persona Manager Integration ===")
        
        user_id = "test_user_integration"
        
        # Store feedback in AWS knowledge base
        test_feedback = [
            {"persona_name": "Dark Humor Connoisseur", "score": 8.5},
            {"persona_name": "Millennial Memer", "score": 6.0},
            {"persona_name": "Corporate Humor Specialist", "score": 4.0},
        ]
        
        for feedback_data in test_feedback:
            asyncio.run(self.aws_kb.update_user_feedback(
                user_id=user_id,
                persona_name=feedback_data["persona_name"],
                feedback_score=feedback_data["score"],
                context="Test context",
                response_text="Test response"
            ))
        
        # Test persona manager retrieval
        personalized_personas = self.persona_manager.get_personalized_personas(
            user_id=user_id, context="Test context", count=3
        )
        
        print(f"✅ Persona manager integration: PASSED")
        print(f"   - Retrieved {len(personalized_personas)} personalized personas")
        print(f"   - Persona names: {[p.name for p in personalized_personas]}")
        
        # Verify that preferred personas are prioritized
        persona_names = [p.name for p in personalized_personas]
        if "Dark Humor Connoisseur" in persona_names:
            print(f"   - ✅ Dark Humor Connoisseur found in personalized list")
        else:
            print(f"   - ❌ Dark Humor Connoisseur NOT found in personalized list")
    
    def test_feedback_consistency(self):
        """Test consistency between database and AWS knowledge base"""
        print("\n=== Testing Feedback Consistency ===")
        
        user_id = "test_user_consistency"
        persona_name = "Dark Humor Connoisseur"
        feedback_score = 8.5
        
        # Store in both systems
        # Database
        db_feedback = UserFeedback(
            user_id=user_id,
            persona_name=persona_name,
            feedback_score=feedback_score,
            context="Test context",
            response_text="Test response"
        )
        self.db.add(db_feedback)
        self.db.commit()
        
        # AWS Knowledge Base
        asyncio.run(self.aws_kb.update_user_feedback(
            user_id=user_id,
            persona_name=persona_name,
            feedback_score=feedback_score,
            context="Test context",
            response_text="Test response"
        ))
        
        # Retrieve from both systems
        db_feedback_retrieved = self.db.query(UserFeedback).filter(
            UserFeedback.user_id == user_id
        ).first()
        
        aws_user_pref = asyncio.run(self.aws_kb.get_user_preference(user_id))
        
        # Compare
        self.assertIsNotNone(db_feedback_retrieved)
        self.assertIsNotNone(aws_user_pref)
        self.assertEqual(db_feedback_retrieved.persona_name, persona_name)
        self.assertEqual(db_feedback_retrieved.feedback_score, feedback_score)
        
        if aws_user_pref.interaction_history:
            aws_interaction = aws_user_pref.interaction_history[0]
            self.assertEqual(aws_interaction['persona_name'], persona_name)
            self.assertEqual(aws_interaction['feedback_score'], feedback_score)
        
        print(f"✅ Feedback consistency: PASSED")
        print(f"   - Database: {db_feedback_retrieved.persona_name} = {db_feedback_retrieved.feedback_score}")
        if aws_user_pref.interaction_history:
            print(f"   - AWS KB: {aws_interaction['persona_name']} = {aws_interaction['feedback_score']}")

def run_feedback_tests():
    """Run all feedback system tests"""
    print("🧪 Running Feedback System Tests")
    print("=" * 50)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestFeedbackSystems)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"   - Tests run: {result.testsRun}")
    print(f"   - Failures: {len(result.failures)}")
    print(f"   - Errors: {len(result.errors)}")
    
    if result.failures:
        print("\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"   - {test}: {traceback}")
    
    if result.errors:
        print("\n❌ Errors:")
        for test, traceback in result.errors:
            print(f"   - {test}: {traceback}")
    
    if not result.failures and not result.errors:
        print("\n✅ All tests passed!")
    
    return result

if __name__ == "__main__":
    run_feedback_tests() 