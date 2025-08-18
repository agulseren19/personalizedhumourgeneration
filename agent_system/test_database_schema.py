#!/usr/bin/env python3
"""
Database Schema Test
Check current database schema and identify issues
"""

import sqlite3
import os
from agent_system.models.database import create_database, get_session_local, UserFeedback, Persona
from sqlalchemy.orm import Session

def check_database_schema():
    """Check the current database schema"""
    print("🔍 Checking Database Schema")
    print("=" * 50)
    
    # Check if database exists
    db_path = "agent_system/agent_humor.db"
    if os.path.exists(db_path):
        print(f"✅ Database exists: {db_path}")
        
        # Connect to SQLite directly to check schema
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get table info
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"📋 Tables found: {[table[0] for table in tables]}")
        
        # Check user_feedback table structure
        if ('user_feedback',) in tables:
            cursor.execute("PRAGMA table_info(user_feedback);")
            columns = cursor.fetchall()
            print("\n📊 user_feedback table structure:")
            for col in columns:
                print(f"   - {col[1]} ({col[2]}) - {'NOT NULL' if col[3] else 'NULL'}")
        
        conn.close()
    else:
        print(f"❌ Database not found: {db_path}")
    
    print("\n" + "=" * 50)

def test_user_feedback_model():
    """Test the UserFeedback model"""
    print("🧪 Testing UserFeedback Model")
    print("=" * 50)
    
    # Create test database
    test_db_url = "sqlite:///test_schema.db"
    engine = create_database(test_db_url)
    SessionLocal = get_session_local(test_db_url)
    db = SessionLocal()
    
    try:
        # Test creating a UserFeedback record
        test_feedback = UserFeedback(
            user_id="test_user_123",
            persona_name="Dark Humor Connoisseur",
            feedback_score=8.5,
            context="Test context",
            response_text="Test response",
            topic="general",
            audience="friends"
        )
        
        db.add(test_feedback)
        db.commit()
        
        print("✅ UserFeedback model test: PASSED")
        print(f"   - user_id: {test_feedback.user_id}")
        print(f"   - persona_name: {test_feedback.persona_name}")
        print(f"   - feedback_score: {test_feedback.feedback_score}")
        
        # Test retrieval
        retrieved = db.query(UserFeedback).filter(
            UserFeedback.user_id == "test_user_123"
        ).first()
        
        if retrieved:
            print("✅ UserFeedback retrieval: PASSED")
            print(f"   - Retrieved user_id: {retrieved.user_id}")
            print(f"   - Retrieved persona_name: {retrieved.persona_name}")
        else:
            print("❌ UserFeedback retrieval: FAILED")
        
    except Exception as e:
        print(f"❌ UserFeedback model test: FAILED")
        print(f"   - Error: {e}")
    
    finally:
        db.close()
        # Clean up test database
        if os.path.exists("test_schema.db"):
            os.remove("test_schema.db")
    
    print("\n" + "=" * 50)

def check_existing_data():
    """Check existing data in the database"""
    print("📊 Checking Existing Data")
    print("=" * 50)
    
    db_path = "agent_system/agent_humor.db"
    if not os.path.exists(db_path):
        print("❌ Database not found")
        return
    
    # Connect to database
    db_url = f"sqlite:///{db_path}"
    SessionLocal = get_session_local(db_url)
    db = SessionLocal()
    
    try:
        # Check personas
        personas = db.query(Persona).all()
        print(f"👥 Personas found: {len(personas)}")
        for persona in personas:
            print(f"   - {persona.name} (ID: {persona.id})")
        
        # Check user feedback
        feedback_count = db.query(UserFeedback).count()
        print(f"\n📝 User feedback records: {feedback_count}")
        
        if feedback_count > 0:
            recent_feedback = db.query(UserFeedback).order_by(
                UserFeedback.created_at.desc()
            ).limit(5).all()
            
            print("📋 Recent feedback:")
            for feedback in recent_feedback:
                print(f"   - User: {feedback.user_id}, Persona: {feedback.persona_name}, Score: {feedback.feedback_score}")
        
    except Exception as e:
        print(f"❌ Error checking data: {e}")
    
    finally:
        db.close()
    
    print("\n" + "=" * 50)

def main():
    """Run all database checks"""
    print("🔍 Database Schema and Data Analysis")
    print("=" * 60)
    
    check_database_schema()
    test_user_feedback_model()
    check_existing_data()
    
    print("✅ Database analysis complete!")

if __name__ == "__main__":
    main() 