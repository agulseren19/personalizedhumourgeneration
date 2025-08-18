from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.sql import func
from datetime import datetime
from typing import Dict, Any, Optional

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=True)
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)  # For authentication
    created_at = Column(DateTime, default=func.now())
    
    # User preferences and demographics
    age_range = Column(String, nullable=True)  # "18-25", "26-35", etc.
    occupation = Column(String, nullable=True)
    education_level = Column(String, nullable=True)
    interests = Column(JSONB, nullable=True)  # List of interests
    humor_preferences = Column(JSONB, nullable=True)  # Learned preferences
    
    # Relationships
    # generation_requests removed since using string user IDs

# Multiplayer Game Models
class Game(Base):
    __tablename__ = "games"
    
    id = Column(String, primary_key=True, index=True)  # Game ID (UUID)
    status = Column(String, default="waiting")  # waiting, active, completed
    created_at = Column(DateTime, default=func.now())
    started_at = Column(DateTime, nullable=True)
    ended_at = Column(DateTime, nullable=True)
    settings = Column(JSONB, nullable=True)  # Game settings
    
    # Relationships
    players = relationship("GamePlayer", back_populates="game")
    rounds = relationship("GameRound", back_populates="game")

class GamePlayer(Base):
    __tablename__ = "game_players"
    
    id = Column(Integer, primary_key=True, index=True)
    game_id = Column(String, ForeignKey("games.id"))
    user_id = Column(Integer, ForeignKey("users.id"))
    username = Column(String, nullable=False)
    is_host = Column(Boolean, default=False)
    is_judge = Column(Boolean, default=False)
    score = Column(Integer, default=0)
    joined_at = Column(DateTime, default=func.now())
    left_at = Column(DateTime, nullable=True)
    
    # Relationships
    game = relationship("Game", back_populates="players")
    user = relationship("User")
    submitted_cards = relationship("SubmittedCard", back_populates="player")

class GameRound(Base):
    __tablename__ = "game_rounds"
    
    id = Column(Integer, primary_key=True, index=True)
    game_id = Column(String, ForeignKey("games.id"))
    round_number = Column(Integer, default=1)
    black_card = Column(Text, nullable=False)
    judge_user_id = Column(Integer, ForeignKey("users.id"))
    status = Column(String, default="collecting")  # collecting, judging, completed
    created_at = Column(DateTime, default=func.now())
    completed_at = Column(DateTime, nullable=True)
    winning_card_id = Column(Integer, ForeignKey("submitted_cards.id"), nullable=True)
    
    # Relationships
    game = relationship("Game", back_populates="rounds")
    judge = relationship("User")
    winning_card = relationship("SubmittedCard", foreign_keys=[winning_card_id])

class SubmittedCard(Base):
    __tablename__ = "submitted_cards"
    
    id = Column(Integer, primary_key=True, index=True)
    round_id = Column(Integer, ForeignKey("game_rounds.id"))
    player_id = Column(Integer, ForeignKey("game_players.id"))
    white_card = Column(Text, nullable=False)
    submitted_at = Column(DateTime, default=func.now())
    
    # Relationships
    round = relationship("GameRound", foreign_keys=[round_id])
    player = relationship("GamePlayer", back_populates="submitted_cards")

class Persona(Base):
    __tablename__ = "personas"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(Text)
    demographics = Column(JSONB)  # Age, occupation, education, etc.
    personality_traits = Column(JSONB)  # Humor style, wit level, etc.
    expertise_areas = Column(JSONB)  # Topics they're good at
    prompt_template = Column(Text)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.now())
    
    # Performance metrics
    avg_rating = Column(Float, default=0.0)
    total_generations = Column(Integer, default=0)
    
    # Relationships
    generations = relationship("HumorGeneration", back_populates="persona")
    preferences = relationship("PersonaPreference", back_populates="persona")

class EvaluatorPersona(Base):
    __tablename__ = "evaluator_personas"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(Text)
    evaluation_criteria = Column(JSONB)  # What they focus on
    personality_traits = Column(JSONB)
    prompt_template = Column(Text)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    evaluations = relationship("HumorEvaluation", back_populates="evaluator_persona")

class HumorGenerationRequest(Base):
    __tablename__ = "humor_generation_requests"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)  # String-based session user IDs
    context = Column(Text)  # The context/topic for humor
    target_audience = Column(String)  # Who is the audience
    humor_type = Column(String)  # "witty", "sarcastic", "pun", etc.
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    # user relationship removed since using string user IDs
    generations = relationship("HumorGeneration", back_populates="request")

class HumorGeneration(Base):
    __tablename__ = "humor_generations"
    
    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(Integer, ForeignKey("humor_generation_requests.id"))
    persona_id = Column(Integer, ForeignKey("personas.id"))
    generated_text = Column(Text)
    model_used = Column(String)  # "gpt-4", "claude-3-sonnet", etc.
    generation_time = Column(Float)  # Time taken to generate
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    request = relationship("HumorGenerationRequest", back_populates="generations")
    persona = relationship("Persona", back_populates="generations")
    evaluations = relationship("HumorEvaluation", back_populates="generation")
    user_feedback = relationship("UserFeedback", back_populates="generation")

class HumorEvaluation(Base):
    __tablename__ = "humor_evaluations"
    
    id = Column(Integer, primary_key=True, index=True)
    generation_id = Column(Integer, ForeignKey("humor_generations.id"))
    evaluator_persona_id = Column(Integer, ForeignKey("evaluator_personas.id"))
    
    # Evaluation scores
    humor_score = Column(Float)  # 0-10 scale
    creativity_score = Column(Float)
    appropriateness_score = Column(Float)
    context_relevance_score = Column(Float)
    overall_score = Column(Float)
    
    # Detailed feedback
    evaluation_reasoning = Column(Text)
    model_used = Column(String)
    evaluation_time = Column(Float)
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    generation = relationship("HumorGeneration", back_populates="evaluations")
    evaluator_persona = relationship("EvaluatorPersona", back_populates="evaluations")

class UserFeedback(Base):
    __tablename__ = "user_feedback"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)  # Changed to String to support frontend user IDs
    generation_id = Column(Integer, ForeignKey("humor_generations.id"), nullable=True)
    
    # Feedback data
    persona_name = Column(String)  # Name of the persona that generated the content
    feedback_score = Column(Float)  # User's rating (1-10 scale)
    context = Column(Text)  # Context of the generation
    response_text = Column(Text)  # The generated text
    topic = Column(String)  # Topic of the generation
    audience = Column(String)  # Target audience
    
    # User ratings (legacy fields)
    liked = Column(Boolean, nullable=True)  # True for like, False for dislike
    humor_rating = Column(Integer, nullable=True)  # 1-5 scale
    feedback_text_legacy = Column(Text, nullable=True)  # Optional text feedback
    
    created_at = Column(DateTime, default=func.now())
    
    # Relationships (optional since we're using string user_id)
    generation = relationship("HumorGeneration", back_populates="user_feedback", foreign_keys=[generation_id])

class PersonaPreference(Base):
    __tablename__ = "persona_preferences"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)  # String-based session user IDs
    persona_id = Column(Integer, ForeignKey("personas.id"))
    
    # Learned preferences
    preference_score = Column(Float)  # How much user likes this persona
    interaction_count = Column(Integer, default=0)
    last_interaction = Column(DateTime, default=func.now())
    
    # Context-based preferences
    context_preferences = Column(JSONB)  # What contexts this persona works well for this user
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    persona = relationship("Persona", back_populates="preferences")

# Database initialization
def create_database(database_url: str):
    """Create database tables"""
    try:
        engine = create_engine(database_url)
        Base.metadata.create_all(bind=engine)
        print(f"✅ Database tables created successfully using: {database_url}")
        return engine
    except Exception as e:
        print(f"❌ Failed to create database tables: {e}")
        raise

def get_session_local(database_url: str):
    """Create database session factory"""
    try:
        engine = create_engine(database_url)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        print(f"✅ Database session factory created using: {database_url}")
        return SessionLocal
    except Exception as e:
        print(f"❌ Failed to create database session factory: {e}")
        raise

def get_db():
    """Dependency function for FastAPI to get database session"""
    try:
        # Try to import settings from config
        import sys
        from pathlib import Path
        
        # Add the parent directory to Python path
        current_dir = Path(__file__).parent
        parent_dir = current_dir.parent
        if str(parent_dir) not in sys.path:
            sys.path.insert(0, str(parent_dir))
        
        from config.settings import settings
        database_url = settings.database_url
    except ImportError:
        # Fallback to environment variable or default
        import os
        database_url = os.getenv("DATABASE_URL", "postgresql://aslihangulseren@localhost:5432/cah_db")
    
    SessionLocal = get_session_local(database_url)
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close() 