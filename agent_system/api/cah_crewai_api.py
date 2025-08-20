#!/usr/bin/env python3

"""
Cards Against Humanity API Server with CrewAI
Exposes the CrewAI-based humor system as REST APIs
"""

import asyncio
import json
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import time
import sys
import os
from fastapi.responses import JSONResponse

# Fix import paths for Render deployment
current_dir = os.path.dirname(os.path.abspath(__file__))
agent_system_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(agent_system_dir)

# Add paths to Python path
sys.path.insert(0, project_root)
sys.path.insert(0, agent_system_dir)

# Try to import required modules with better error handling
try:
    from agent_system.agents.improved_humor_agents import (
        ImprovedHumorOrchestrator, 
        HumorRequest
    )
    from agent_system.agents.improved_humor_agents import ContentFilter
    print("✅ Core humor agents imported successfully")
except ImportError as e:
    print(f"❌ Core humor agents import failed: {e}")
    deployment_errors.append(f"Core humor agents: {str(e)}")
    # Create fallback classes
    class ImprovedHumorOrchestrator:
        def __init__(self):
            pass
        async def generate_humor(self, *args, **kwargs):
            return {"error": "Humor system not available"}
    
    class ContentFilter:
        def __init__(self):
            pass
        def filter_content(self, *args, **kwargs):
            return True

try:
    from agent_system.knowledge.improved_aws_knowledge_base import improved_aws_knowledge_base
    print("✅ PostgreSQL knowledge base imported successfully")
except ImportError as e:
    print(f"⚠️  PostgreSQL knowledge base import failed: {e}")
    deployment_errors.append(f"PostgreSQL knowledge base: {str(e)}")
    print("⚠️  Using fallback knowledge base functionality")
    improved_aws_knowledge_base = None

try:
    from agent_system.personas.enhanced_persona_templates import get_all_personas
    print("✅ Persona templates imported successfully")
except ImportError as e:
    print(f"❌ Persona templates import failed: {e}")
    get_all_personas = lambda: []

try:
    from agent_system.personas.dynamic_persona_generator import dynamic_persona_generator
    print("✅ Dynamic persona generator imported successfully")
except ImportError as e:
    print(f"❌ Dynamic persona generator import failed: {e}")
    dynamic_persona_generator = None

try:
    from agent_system.agents.bws_evaluation import bws_evaluator, BWS_Item, BWS_Comparison
    print("✅ BWS evaluation imported successfully")
except ImportError as e:
    print(f"❌ BWS evaluation import failed: {e}")
    bws_evaluator = None
    BWS_Item = None
    BWS_Comparison = None

# Import multiplayer game functionality
try:
    from agent_system.api.multiplayer_routes import get_game_manager
    from agent_system.game.authenticated_multiplayer_cah import AuthenticatedMultiplayerCAHGame
    from agent_system.models.database import get_db
    print("✅ Authenticated multiplayer game imported successfully")
except ImportError as e:
    print(f"❌ Authenticated multiplayer game import failed: {e}")
    AuthenticatedMultiplayerCAHGame = None
    get_game_manager = None

# Import authentication routes
try:
    from agent_system.api.auth_routes import router as auth_router
    print("✅ Authentication routes imported successfully")
except ImportError as e:
    print(f"❌ Authentication routes import failed: {e}")
    auth_router = None

# Import Google OAuth routes
try:
    from agent_system.api.google_oauth_routes import router as google_oauth_router
    print("✅ Google OAuth routes imported successfully")
except ImportError as e:
    print(f"❌ Google OAuth routes import failed: {e}")
    google_oauth_router = None

# Import multiplayer routes
try:
    from agent_system.api.multiplayer_routes import router as multiplayer_router
    print("✅ Multiplayer routes imported successfully")
except ImportError as e:
    print(f"❌ Multiplayer routes import failed: {e}")
    multiplayer_router = None

# Helper function to convert numpy types to Python native types
def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

# Pydantic models for API requests/responses
class GenerateHumorRequest(BaseModel):
    context: str
    audience: str = "friends"
    topic: str = "general"
    user_id: str
    card_type: str = "white"  # "white" or "black"

class FeedbackRequest(BaseModel):
    user_id: str
    persona_name: str
    feedback_score: float
    context: str
    response_text: str = ""
    topic: str = ""
    audience: str = ""

class HumorResponse(BaseModel):
    success: bool
    generations: List[Dict[str, Any]]
    recommended_personas: List[str]
    generation_time: float
    error: Optional[str] = None

class UserAnalytics(BaseModel):
    user_id: str
    total_interactions: int
    average_score: float
    liked_personas: List[str]
    disliked_personas: List[str]
    persona_performance: Dict[str, Dict[str, Any]]

class ContentFilterResponse(BaseModel):
    is_safe: bool
    toxicity_score: float
    sanitized_content: str

class BWS_EvaluationRequest(BaseModel):
    """Request to start BWS evaluation for generated items"""
    generation_ids: List[str]
    user_id: str

class BWS_JudgmentRequest(BaseModel):
    """Request to record BWS judgment"""
    comparison_id: str
    best_item_id: str
    worst_item_id: str
    user_id: str

class BWS_ComparisonResponse(BaseModel):
    """BWS comparison for user evaluation"""
    comparison_id: str
    items: List[Dict[str, Any]]
    instruction: str

# Initialize FastAPI app
app = FastAPI(
    title="Cards Against Humanity API (CrewAI)",
    description="AI-powered humor generation with CrewAI multi-agent system",
    version="1.0.0"
)

# Add CORS middleware FIRST - before any routes
print("🔧 Adding CORS middleware...")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow ALL origins temporarily
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=86400,  # Cache preflight for 24 hours
)
print("✅ CORS middleware added successfully")
print("   Allowed origins: * (ALL origins)")
print("   Allow credentials: True")
print("   Allow methods: *")
print("   Allow headers: *")

# Add authentication routes if available
if auth_router:
    app.include_router(auth_router)
    print("✅ Authentication routes added to API")
else:
    print("⚠️  Authentication routes not available")

# Add Google OAuth routes if available
if google_oauth_router:
    app.include_router(google_oauth_router)
    print("✅ Google OAuth routes added to API")
else:
    print("⚠️  Google OAuth routes not available")

# Add multiplayer routes if available
if multiplayer_router:
    app.include_router(multiplayer_router)
    print("✅ Multiplayer routes added to API")
else:
    print("⚠️  Multiplayer routes not available")

# WebSocket connection management - defined at module level for accessibility
active_connections: Dict[str, Dict[str, WebSocket]] = {}  # game_id -> {user_id: websocket}

# Global broadcast function reference
broadcast_function = None

# Initialize CrewAI humor system
humor_system = None
content_filter = None
persona_manager = None

# Add global error tracking
deployment_errors = []

async def initialize_humor_system():
    """Initialize the humor system"""
    global humor_system, content_filter, persona_manager
    
    if humor_system is not None:
        return  # Already initialized
    
    print("🚀 Initializing CAH CrewAI humor system...")
    
    try:
        # Initialize database session
        from agent_system.models.database import get_session_local, create_database
        from agent_system.config.settings import settings
        
        # Ensure database exists
        create_database(settings.database_url)
        
        SessionLocal = get_session_local(settings.database_url)
        db = SessionLocal()
        
        # Initialize CrewAI orchestrator
        from agent_system.personas.persona_manager import PersonaManager
        persona_manager = PersonaManager(db)
        humor_system = ImprovedHumorOrchestrator()
        content_filter = ContentFilter()
        
        # Ensure global variables are properly set
        globals()['persona_manager'] = persona_manager
        
        print("✅ CAH CrewAI humor system ready!")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize humor system: {e}")
        import traceback
        traceback.print_exc()
        return False

@app.on_event("startup")
async def startup_event():
    """Initialize the CrewAI humor system on startup"""
    print("🚀 Starting CAH CrewAI API Server...")
    await initialize_humor_system()
    print("✅ CAH CrewAI API Server ready!")

async def create_sample_dynamic_personas():
    """Create sample dynamic personas for immediate availability"""
    try:
        # Create simple pre-defined personas instead of using LLM generation
        sample_personas = {
            "gaming_enthusiast": {
                "name": "Gaming Enthusiast",
                "description": "A humor expert who specializes in gaming culture and digital life",
                "humor_style": "witty and tech-savvy",
                "expertise_areas": ["gaming", "technology", "digital culture"],
                "demographic_hints": {"age_range": "18-35", "tech_savvy": True},
                "prompt_style": "Generate clever humor that references gaming culture and digital life",
                "examples": [
                    "Accidentally speedrunning my morning routine",
                    "Lag in real life",
                    "Achievement unlocked: Adulting"
                ]
            },
            "office_humorist": {
                "name": "Office Humorist", 
                "description": "A workplace humor specialist who finds comedy in corporate life",
                "humor_style": "observational and relatable",
                "expertise_areas": ["workplace", "corporate", "office culture"],
                "demographic_hints": {"age_range": "25-55", "professional": True},
                "prompt_style": "Generate relatable workplace humor that captures office life",
                "examples": [
                    "The printer's existential crisis",
                    "Accidentally sending 'love you' to the entire company",
                    "The coffee machine's passive-aggressive messages"
                ]
            },
            "family_comedian": {
                "name": "Family Comedian",
                "description": "A family-friendly humor expert who specializes in parenting and domestic life",
                "humor_style": "wholesome and relatable",
                "expertise_areas": ["family", "parenting", "domestic life"],
                "demographic_hints": {"age_range": "25-45", "family_oriented": True},
                "prompt_style": "Generate family-friendly humor that parents and kids can enjoy",
                "examples": [
                    "My child's elaborate excuses for bedtime",
                    "The toy that only works when I'm not looking", 
                    "My kid's negotiation skills are better than mine"
                ]
            }
        }
        
        # Add personas to the dynamic generator
        for key, persona_data in sample_personas.items():
            from agent_system.personas.enhanced_persona_templates import PersonaTemplate
            persona = PersonaTemplate(
                name=persona_data["name"],
                description=persona_data["description"],
                humor_style=persona_data["humor_style"],
                expertise_areas=persona_data["expertise_areas"],
                demographic_hints=persona_data["demographic_hints"],
                prompt_style=persona_data["prompt_style"],
                examples=persona_data["examples"]
            )
            dynamic_persona_generator.generated_personas[key] = persona
        
        print(f"✅ Created {len(sample_personas)} sample dynamic personas")
        
    except Exception as e:
        print(f"⚠️ Warning: Could not create sample dynamic personas: {e}")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Cards Against Humanity CrewAI API is running!", 
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

@app.get("/deployment-status")
async def deployment_status():
    """Get detailed deployment status for debugging"""
    return {
        "status": "deployed",
        "timestamp": datetime.now().isoformat(),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "imports": {
            "humor_system": humor_system is not None,
            "content_filter": content_filter is not None,
            "knowledge_base": improved_aws_knowledge_base is not None,
            "persona_templates": "get_all_personas" in globals(),
            "dynamic_persona_generator": dynamic_persona_generator is not None,
            "bws_evaluation": bws_evaluator is not None,
            "multiplayer_game": AuthenticatedMultiplayerCAHGame is not None,
            "auth_routes": auth_router is not None,
            "google_oauth_routes": google_oauth_router is not None,
            "multiplayer_routes": multiplayer_router is not None
        },
        "deployment_errors": deployment_errors,
        "cors_enabled": True
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for Render"""
    try:
        # Basic health check - don't fail on external service issues
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "api_server": "running",
            "cors_middleware": "enabled"
        }
        
        # Try database connection (but don't fail if it's down)
        try:
            from agent_system.models.database import get_session_local
            from agent_system.config.settings import settings
            
            SessionLocal = get_session_local(settings.database_url)
            db = SessionLocal()
            
            try:
                from sqlalchemy import text
                result = db.execute(text("SELECT 1"))
                health_status["database"] = "healthy"
            except Exception as e:
                health_status["database"] = f"unhealthy: {str(e)[:100]}"
            finally:
                db.close()
        except Exception as e:
            health_status["database"] = f"unavailable: {str(e)[:100]}"
        
        # Test knowledge base (but don't fail if it's down)
        try:
            health_status["knowledge_base"] = "healthy" if improved_aws_knowledge_base else "unavailable"
        except Exception as e:
            health_status["knowledge_base"] = f"unavailable: {str(e)[:100]}"
        
        # Test humor system (but don't fail if it's down)
        try:
            health_status["humor_system"] = "healthy" if humor_system else "unavailable"
        except Exception as e:
            health_status["humor_system"] = f"unavailable: {str(e)[:100]}"
        
        return health_status
        
    except Exception as e:
        # Return basic health status even if detailed check fails
        return {
            "status": "degraded",
            "timestamp": datetime.now().isoformat(),
            "api_server": "running",
            "error": str(e)[:200]
        }

@app.get("/test-humor")
async def test_humor():
    """Test CrewAI humor generation system"""
    try:
        if humor_system is None:
            return {"error": "CrewAI humor system not initialized"}
        
        # Try a simple generation
        request = HumorRequest(
            context="Test context _____",
            audience="test",
            topic="test",
            user_id="test"
        )
        
        result = await humor_system.generate_and_evaluate_humor(request)
        
        return {
            "success": True,
            "humor_system_type": "CrewAI Multi-Agent System",
            "result_keys": list(result.keys()),
            "result_success": result.get('success'),
            "num_results": len(result.get('top_results', []))
        }
        
    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }

@app.get("/personas")
async def get_personas():
    """Get all available personas from database and dynamic ones"""
    try:
        from agent_system.models.database import get_session_local, Persona
        from agent_system.config.settings import settings
        
        # Get database personas (includes static and saved dynamic ones)
        SessionLocal = get_session_local(settings.database_url)
        db = SessionLocal()
        
        try:
            db_personas = db.query(Persona).filter(Persona.is_active == True).all()
            
            # Get dynamic personas from memory
            dynamic_personas = dynamic_persona_generator.get_all_personas()
            
            print(f"DEBUG: Database personas count: {len(db_personas)}")
            print(f"DEBUG: Dynamic personas in memory: {len(dynamic_personas)}")
            print(f"DEBUG: Dynamic persona names: {list(dynamic_personas.keys())}")
            
            personas_list = []
            
            # Add database personas
            for persona in db_personas:
                personas_list.append({
                    "id": persona.id,
                    "name": persona.name,
                    "description": persona.description,
                    "expertise": persona.expertise_areas or [],
                    "humor_style": persona.personality_traits.get("humor_style", "unknown") if persona.personality_traits else "unknown",
                    "source": "database",
                    "is_ai_comedian": persona.personality_traits.get("is_ai_comedian", False) if persona.personality_traits else False,
                    "is_dynamic": persona.personality_traits.get("is_dynamic", False) if persona.personality_traits else False
                })
            
            # Add any dynamic personas not yet in database
            for key, persona in dynamic_personas.items():
                # Check if already in database
                if not any(db_p.name == persona.name for db_p in db_personas):
                    personas_list.append({
                        "id": key,
                        "name": persona.name,
                        "description": persona.description,
                        "expertise": persona.expertise_areas,
                        "humor_style": persona.humor_style,
                        "source": "dynamic_memory",
                        "is_ai_comedian": True,
                        "is_dynamic": True
                    })
            
            return {
                "success": True,
                "personas": personas_list,
                "total_count": len(personas_list),
                "static_count": len([p for p in personas_list if not p.get("is_dynamic", False)]),
                "dynamic_count": len([p for p in personas_list if p.get("is_dynamic", False)])
            }
        finally:
            db.close()
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting personas: {str(e)}")

@app.post("/generate")
async def generate_humor(request: HumorRequest):
    """Generate humor with improved error handling and database testing"""
    try:
        print(f"🎭 Generating humor for user {request.user_id}")
        print(f"   Context: {request.context}")
        print(f"   Audience: {request.audience}")
        print(f"   Topic: {request.topic}")
        
        # Test database connection first
        try:
            from agent_system.models.database import get_session_local
            from agent_system.config.settings import settings
            
            SessionLocal = get_session_local(settings.database_url)
            db = SessionLocal()
            
            try:
                # Simple database test
                from sqlalchemy import text
                result = db.execute(text("SELECT 1"))
                print("✅ Database connection test successful")
            except Exception as db_error:
                print(f"❌ Database connection test failed: {db_error}")
                return JSONResponse(
                    status_code=500,
                    content={"detail": f"Database connection failed: {db_error}"}
                )
            finally:
                db.close()
                
        except Exception as e:
            print(f"❌ Database setup failed: {e}")
            return JSONResponse(
                status_code=500,
                content={"detail": f"Database setup failed: {e}"}
            )
        
        # Test knowledge base
        try:
            if improved_aws_knowledge_base:
                print("✅ Knowledge base available")
                # Test user preference retrieval
                user_pref = await improved_aws_knowledge_base.get_user_preference(request.user_id)
                if user_pref:
                    print(f"✅ User preferences loaded: {len(user_pref.interaction_history)} interactions")
                else:
                    print("⚠️ No user preferences found, creating new user")
            else:
                print("❌ Knowledge base not available")
                return JSONResponse(
                    status_code=500,
                    content={"detail": "Knowledge base not available"}
                )
        except Exception as kb_error:
            print(f"❌ Knowledge base test failed: {kb_error}")
            return JSONResponse(
                status_code=500,
                content={"detail": f"Knowledge base test failed: {kb_error}"}
            )
        
        # Generate humor with timeout protection
        try:
            print("🚀 Starting humor generation...")
            
            # Check if humor system is available
            if humor_system is None:
                print("❌ Humor system not initialized")
                return JSONResponse(
                    status_code=500,
                    content={"detail": "Humor system not initialized"}
                )
            
            # Use asyncio.wait_for to prevent infinite hanging
            result = await asyncio.wait_for(
                humor_system.generate_and_evaluate_humor(request),
                timeout=120.0  # 120 second timeout for complex humor generation
            )
            
            print("✅ Humor generation completed successfully")
            
            # Ensure result has all required fields for frontend
            if isinstance(result, dict):
                # Standardize the response structure
                standardized_result = {
                    "success": result.get("success", True),
                    "results": result.get("results", []),
                    "top_results": result.get("top_results", []),
                    "num_results": result.get("num_results", 0),
                    "generation_time": result.get("generation_time", 0.0),
                    "fallback_used": result.get("fallback_used", False),
                    "best_result": result.get("best_result"),
                    "recommended_personas": result.get("recommended_personas", [])
                }

                # If zero generations, return 500 so frontend shows an error instead of silent 0 cards
                try:
                    has_results = bool(standardized_result.get("results"))
                    num_results = int(standardized_result.get("num_results", 0) or 0)
                except Exception:
                    has_results = False
                    num_results = 0

                if (not has_results) or num_results <= 0:
                    return JSONResponse(
                        status_code=500,
                        content={
                            "detail": "No humor generated",
                            "success": False,
                            "results": [],
                            "top_results": [],
                            "num_results": 0,
                            "generation_time": float(standardized_result.get("generation_time", 0.0) or 0.0),
                            "fallback_used": bool(standardized_result.get("fallback_used", False)),
                            "recommended_personas": standardized_result.get("recommended_personas", [])
                        }
                    )

                # Convert numpy types to Python types
                standardized_result = convert_numpy_types(standardized_result)
                return standardized_result
            else:
                # Fallback if result is not a dict
                return {
                    "success": True,
                    "results": [],
                    "top_results": [],
                    "num_results": 0,
                    "generation_time": 0.0,
                    "fallback_used": False,
                    "best_result": None,
                    "recommended_personas": []
                }
            
        except asyncio.TimeoutError:
            print("❌ Humor generation timed out after 120 seconds")
            return JSONResponse(
                status_code=408,
                content={
                    "detail": "Humor generation timed out. Please try again.",
                    "success": False,
                    "results": [],
                    "top_results": [],
                    "num_results": 0,
                    "generation_time": 0.0,
                    "fallback_used": False
                }
            )
        except Exception as gen_error:
            print(f"❌ Humor generation failed: {gen_error}")
            return JSONResponse(
                status_code=500,
                content={
                    "detail": f"Error generating humor: {gen_error}",
                    "success": False,
                    "results": [],
                    "top_results": [],
                    "num_results": 0,
                    "generation_time": 0.0,
                    "fallback_used": False
                }
            )
            
    except Exception as e:
        print(f"❌ Unexpected error in generate endpoint: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "detail": f"Unexpected error: {e}",
                "success": False,
                "results": [],
                "top_results": [],
                "num_results": 0,
                "generation_time": 0.0,
                "fallback_used": False
            }
        )

@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """Submit user feedback for a generation and update persona preferences"""
    try:
        # Store feedback in database and update preferences
        from agent_system.models.database import UserFeedback, get_session_local, PersonaPreference, Persona
        from agent_system.config.settings import settings
        from agent_system.personas.persona_manager import PersonaManager
        from datetime import datetime
        from sqlalchemy import and_, text
        
        # Get database session
        SessionLocal = get_session_local(settings.database_url)
        db = SessionLocal()
        
        try:
            # Store feedback
            feedback = UserFeedback(
                user_id=request.user_id,
                persona_name=request.persona_name,
                feedback_score=request.feedback_score,
                context=request.context,
                response_text=request.response_text,
                topic=request.topic,
                audience=request.audience,
                created_at=datetime.now()
            )
            
            db.add(feedback)
            
            # FIXED: Update persona preferences and interaction count
            persona = db.query(Persona).filter(Persona.name == request.persona_name).first()
            
            if persona:
                # Check if preference exists
                preference = db.query(PersonaPreference).filter(
                    and_(
                        PersonaPreference.user_id == request.user_id,
                        PersonaPreference.persona_id == persona.id
                    )
                ).first()
                
                if preference:
                    # Update existing preference
                    old_count = preference.interaction_count
                    old_score = preference.preference_score
                    
                    # Calculate new average score
                    new_count = old_count + 1
                    new_score = ((old_score * old_count) + request.feedback_score) / new_count
                    
                    preference.interaction_count = new_count
                    preference.preference_score = new_score
                    preference.last_interaction = datetime.now()
                    
                    print(f"  Updated preference: {request.user_id} -> {request.persona_name}: "
                          f"{new_score:.1f}/10 ({new_count} interactions)")
                else:
                    # Create new preference
                    new_preference = PersonaPreference(
                        user_id=request.user_id,
                        persona_id=persona.id,
                        interaction_count=1,
                        preference_score=request.feedback_score,
                        last_interaction=datetime.now()
                    )
                    db.add(new_preference)
                    
                    print(f"  Created preference: {request.user_id} -> {request.persona_name}: "
                          f"{request.feedback_score}/10 (1 interaction)")
            
            db.commit()
            
            print(f"  Updated feedback: {request.user_id} -> {request.persona_name}: {request.feedback_score}/10")
            
            return {
                "success": True,
                "message": "Feedback recorded and preferences updated successfully"
            }
        finally:
            db.close()
            
    except Exception as e:
        print(f"ERROR in feedback: {e}")
        raise HTTPException(status_code=500, detail=f"Error recording feedback: {str(e)}")

@app.get("/analytics/{user_id}")
async def get_user_analytics(user_id: str):
    """Get user analytics and learning data from database"""
    try:
        from agent_system.models.database import get_session_local, PersonaPreference, UserFeedback, Persona
        from agent_system.config.settings import settings
        from datetime import datetime
        
        # Get database session
        SessionLocal = get_session_local(settings.database_url)
        db = SessionLocal()
        
        try:
            # Get user preferences from database with persona relationship
            
            preferences = db.query(PersonaPreference).join(Persona).filter(
                PersonaPreference.user_id == user_id
            ).all()
            
            # Get user feedback from database  
            feedback_records = db.query(UserFeedback).filter(
                UserFeedback.user_id == user_id
            ).all()
            
            if not preferences and not feedback_records:
                return {
                    "success": True,
                    "analytics": {
                        'user_id': user_id,
                        'total_interactions': 0,
                        'average_score': 0,
                        'liked_personas': [],
                        'disliked_personas': [],
                        'top_personas': [],
                        'persona_performance': {},
                        'last_updated': datetime.now().isoformat(),
                        'favorite_persona': None
                    }
                }
            
            # Calculate analytics from feedback records (primary source)
            total_feedback = len(feedback_records)
            total_interactions = total_feedback  # Each feedback is an interaction
            
            # Calculate average score from feedback records
            if feedback_records:
                feedback_scores = [fb.feedback_score for fb in feedback_records]
                average_score = sum(feedback_scores) / len(feedback_scores)
            else:
                average_score = 0
            
            # Build comprehensive persona performance from feedback records
            from collections import defaultdict
            persona_stats = defaultdict(lambda: {'scores': [], 'count': 0})
            
            # Aggregate feedback by persona
            for fb in feedback_records:
                persona_name = fb.persona_name
                if persona_name and persona_name != "Unknown Persona":  # Filter out unknown personas
                    persona_stats[persona_name]['scores'].append(fb.feedback_score)
                    persona_stats[persona_name]['count'] += 1
            
            # Build persona performance data from feedback
            persona_performance = {}
            for persona_name, stats in persona_stats.items():
                if stats['count'] > 0:
                    avg_score = sum(stats['scores']) / len(stats['scores'])
                    persona_performance[persona_name] = {
                        'avg_score': round(avg_score, 1),
                        'interaction_count': stats['count'],
                        'status': 'liked' if avg_score >= 7.0 else 
                                 'disliked' if avg_score < 5.0 else 'neutral'
                    }
            
            # Determine liked/disliked personas from feedback data
            liked_personas = [name for name, perf in persona_performance.items() if perf['avg_score'] >= 7.0]
            disliked_personas = [name for name, perf in persona_performance.items() if perf['avg_score'] < 5.0]
            
            # Find favorite persona (highest average score with at least 1 interaction)
            favorite_persona = None
            if persona_performance:
                # Get persona with highest average score
                favorite_entry = max(persona_performance.items(), key=lambda x: x[1]['avg_score'])
                favorite_persona = favorite_entry[0]
            
            # Get top personas (sorted by average score)
            top_personas = sorted(
                [{'persona_name': name, 
                  'avg_score': perf['avg_score'],
                  'interaction_count': perf['interaction_count']} 
                 for name, perf in persona_performance.items()],
                key=lambda x: x['avg_score'], 
                reverse=True
            )[:5]
            
            analytics = {
                'user_id': user_id,
                'total_interactions': total_interactions,
                'total_feedback': total_feedback,
                'average_score': round(average_score, 1),
                'liked_personas': liked_personas,
                'disliked_personas': disliked_personas,
                'top_personas': top_personas,
                'persona_performance': persona_performance,
                'last_updated': max([pref.last_interaction for pref in preferences], default=datetime.now()).isoformat(),
                'favorite_persona': favorite_persona,
                'recent_feedback': [
                    {
                        'persona_name': fb.persona_name,
                        'score': fb.feedback_score,
                        'context': fb.context,
                        'created_at': fb.created_at.isoformat() if fb.created_at else None
                    }
                    for fb in feedback_records[-5:]  # Last 5 feedback records
                ]
            }
            
            return {
                "success": True,
                "analytics": analytics
            }
            
        finally:
            db.close()
            
    except Exception as e:
        print(f"ERROR in analytics: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting analytics: {str(e)}")

@app.get("/recommendations/{user_id}")
async def get_persona_recommendations(user_id: str, context: str, audience: str = "general"):
    """Get personalized persona recommendations"""
    try:
        recommendations = await improved_aws_knowledge_base.get_persona_recommendations(
            user_id=user_id,
            context=context,
            audience=audience
        )
        
        return {
            "success": True,
            "recommendations": recommendations
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting recommendations: {str(e)}")

@app.post("/content-filter")
async def check_content_safety(content: Dict[str, str]):
    """Check if content is safe using content filter"""
    try:
        text = content.get("text", "")
        is_safe, toxicity_score, scores = content_filter.is_content_safe(text)
        sanitized = content_filter.sanitize_content(text) if not is_safe else text
        
        return ContentFilterResponse(
            is_safe=is_safe,
            toxicity_score=toxicity_score,
            sanitized_content=sanitized
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking content: {str(e)}")

@app.post("/bws/start-evaluation")
async def start_bws_evaluation(request: BWS_EvaluationRequest):
    """Start Best-Worst Scaling evaluation for a set of generated humor items"""
    try:
        # Create BWS items from generation IDs (in practice, would fetch from database)
        items = []
        for i, gen_id in enumerate(request.generation_ids):
            # Mock humor text for demonstration - in practice, fetch from database
            mock_texts = [
                "Something unexpectedly hilarious",
                "A clever wordplay response", 
                "Dark humor that's surprisingly appropriate",
                "Absurdist comedy gold"
            ]
            
            item = BWS_Item(
                id=gen_id,
                text=mock_texts[i % len(mock_texts)],
                metadata={"generation_id": gen_id, "user_id": request.user_id}
            )
            items.append(item)
        
        # Add items to BWS evaluator
        bws_evaluator.add_items(items)
        
        # Generate comparisons
        comparisons = bws_evaluator.generate_comparisons(n_comparisons=len(items) * 2)
        
        return {
            "success": True,
            "total_items": len(items),
            "total_comparisons": len(comparisons),
            "message": f"BWS evaluation started with {len(comparisons)} comparisons"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting BWS evaluation: {str(e)}")

@app.get("/bws/next-comparison/{user_id}")
async def get_next_bws_comparison(user_id: str):
    """Get next BWS comparison for user evaluation"""
    try:
        comparison = bws_evaluator.get_comparison_for_user(user_id)
        
        if not comparison:
            return {
                "success": False,
                "message": "No more comparisons available",
                "completed": True
            }
        
        # Format comparison for frontend
        items_data = []
        for item in comparison.items:
            items_data.append({
                "id": item.id,
                "text": item.text,
                "metadata": item.metadata
            })
        
        return BWS_ComparisonResponse(
            comparison_id=comparison.comparison_id,
            items=items_data,
            instruction="Select the FUNNIEST and LEAST FUNNY responses from the 4 options below"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting BWS comparison: {str(e)}")

@app.post("/bws/submit-judgment")
async def submit_bws_judgment(request: BWS_JudgmentRequest):
    """Submit BWS judgment (best and worst selections)"""
    try:
        # Record the judgment
        bws_evaluator.record_judgment(
            comparison_id=request.comparison_id,
            best_item_id=request.best_item_id,
            worst_item_id=request.worst_item_id,
            user_id=request.user_id
        )
        
        return {
            "success": True,
            "message": "BWS judgment recorded successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error submitting BWS judgment: {str(e)}")

@app.get("/bws/results")
async def get_bws_results():
    """Get BWS evaluation results and analysis"""
    try:
        # Calculate BWS scores
        bws_results = bws_evaluator.calculate_bws_scores()
        
        # Generate summary
        summary = bws_evaluator.generate_evaluation_summary()
        
        # Convert to JSON-serializable format
        results = {
            "item_scores": bws_results.item_scores,
            "item_rankings": bws_results.item_rankings,
            "total_comparisons": bws_results.total_comparisons,
            "confidence_intervals": bws_results.confidence_intervals,
            "summary": summary
        }
        
        return {
            "success": True,
            "results": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting BWS results: {str(e)}")

@app.post("/bws/compare-with-likert")
async def compare_bws_with_likert(likert_scores: Dict[str, float]):
    """Compare BWS results with traditional Likert scale ratings"""
    try:
        comparison = bws_evaluator.compare_with_likert(likert_scores)
        
        return {
            "success": True,
            "comparison": comparison,
            "interpretation": {
                "pearson_correlation": "Measures linear relationship between BWS and Likert scores",
                "spearman_correlation": "Measures rank-order relationship", 
                "high_correlation": "> 0.7 indicates good agreement",
                "literature_note": "BWS typically more reliable with fewer judgments (Horvitz et al.)"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error comparing BWS with Likert: {str(e)}")

# Multiplayer Game Endpoints
if AuthenticatedMultiplayerCAHGame is not None:
    @app.on_event("startup")
    async def initialize_multiplayer():
        """Initialize multiplayer game manager"""
        try:
            from agent_system.agents.improved_humor_agents import ImprovedHumorOrchestrator
            from agent_system.personas.persona_manager import PersonaManager
            
            # Get the global humor orchestrator and persona manager
            global humor_system, persona_manager
            
            if humor_system and persona_manager:
                print("✅ Multiplayer game manager initialized")
            else:
                print("⚠️ Humor orchestrator or persona manager not available")
        except Exception as e:
            print(f"❌ Failed to initialize multiplayer game manager: {e}")
    
    class CreateGameRequest(BaseModel):
        game_id: str
        host_username: str
        settings: Optional[Dict[str, Any]] = None
    
    @app.post("/game/create")
    async def create_multiplayer_game(request: CreateGameRequest):
        """Create a new multiplayer game"""
        try:
            print(f"🎮 Creating game: {request.game_id} for host: {request.host_username}")
            
            # Simple test response first
            print("✅ Basic request processing working")
            
            # Test game manager initialization
            try:
                game_manager = get_game_manager()
                print("✅ Game manager already initialized")
            except Exception as e:
                print(f"⚠️ Game manager not initialized: {e}")
                # Try to initialize it
                from agent_system.agents.improved_humor_agents import ImprovedHumorOrchestrator
                from agent_system.personas.persona_manager import PersonaManager
                
                global humor_system, persona_manager
                print(f"🔍 Checking global variables - humor_system: {humor_system is not None}, persona_manager: {persona_manager is not None}")
                
                if humor_system and persona_manager:
                    print("✅ Multiplayer game manager initialized")
                    game_manager = get_game_manager()
                else:
                    print("❌ Humor orchestrator or persona manager not available")
                    return {
                        "success": True,
                        "game_id": request.game_id,
                        "host_user_id": f"user_{hash(request.host_username) % 10000}",
                        "status": "waiting",
                        "message": "Game creation endpoint working (game manager not available)"
                    }
            
            # Now try to create the actual game
            host_user_id = f"user_{hash(request.host_username) % 10000}"
            print(f"👤 Generated host user ID: {host_user_id}")
            
            print("🔄 About to call game_manager.create_game...")
            game_state = await game_manager.create_game(
                game_id=request.game_id,
                host_user_id=host_user_id,
                host_username=request.host_username,
                settings=request.settings or {}
            )
            print("✅ game_manager.create_game completed")
            
            print(f"✅ Game created successfully: {request.game_id}")
            return {
                "success": True,
                "game_id": request.game_id,
                "host_user_id": host_user_id,
                "status": game_state.status.value
            }
            
            # TODO: Re-enable full game creation logic after testing
            # Initialize game manager if not already done
            # try:
            #     game_manager = get_game_manager()
            #     print("✅ Game manager already initialized")
            # except Exception as e:
            #     print(f"⚠️ Game manager not initialized, initializing now: {e}")
            #     # Initialize the game manager
            #     from agent_system.agents.improved_humor_agents import ImprovedHumorOrchestrator
            #     from agent_system.personas.persona_manager import PersonaManager
            #     
            #     global humor_orchestrator, persona_manager
            #     print(f"🔍 Checking global variables - humor_orchestrator: {humor_orchestrator is not None}, persona_manager: {persona_manager is not None}")
            #     
            #     if humor_orchestrator and persona_manager:
            #         initialize_game_manager(humor_orchestrator, persona_manager)
            #         print("✅ Multiplayer game manager initialized on first request")
            #     else:
            #         print("❌ Humor orchestrator or persona manager not available")
            #         # Try to reinitialize the humor system
            #         await initialize_humor_system()
            #         if humor_orchestrator and persona_manager:
            #             initialize_game_manager(humor_orchestrator, persona_manager)
            #             print("✅ Multiplayer game manager initialized after reinitialization")
            #         else:
            #             raise Exception("Failed to initialize humor system components")
            #     
            #     game_manager = get_game_manager()
            # 
            # host_user_id = f"user_{hash(request.host_username) % 10000}"
            # print(f"👤 Generated host user ID: {host_user_id}")
            # 
            # print("🔄 About to call game_manager.create_game...")
            # game_state = await game_manager.create_game(
            #     game_id=request.game_id,
            #     host_user_id=host_user_id,
            #     host_username=request.host_username,
            #     settings=request.settings or {}
            # )
            # print("✅ game_manager.create_game completed")
            # 
            # print(f"✅ Game created successfully: {request.game_id}")
            # return {
            #     "success": True,
            #     "game_id": request.game_id,
            #     "host_user_id": host_user_id,
            #     "status": game_state.status.value
            # }
        except Exception as e:
            print(f"❌ Error creating game: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))
    
    class JoinGameRequest(BaseModel):
        game_id: str
        username: str
    
    @app.post("/game/join")
    async def join_multiplayer_game(request: JoinGameRequest):
        """Join an existing game"""
        try:
            game_manager = get_game_manager()
            user_id = f"user_{hash(request.username) % 10000}"
            
            game_state = await game_manager.join_game(request.game_id, user_id, request.username)
            player_view = game_manager.get_player_view(request.game_id, user_id)
            
            # FIXED: Broadcast player joined to all other players
            # Note: Broadcasting is handled by the authenticated multiplayer system
            print(f"📨 Player {user_id} ({request.username}) joined game {request.game_id}")
            
            return {
                "success": True,
                "user_id": user_id,
                "game_state": player_view
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/game/{game_id}/state")
    async def get_game_state(game_id: str, user_id: str):
        """Get current game state for a player"""
        try:
            game_manager = get_game_manager()
            player_view = game_manager.get_player_view(game_id, user_id)
            
            if not player_view:
                raise HTTPException(status_code=404, detail="Game not found or player not in game")
            
            return {
                "success": True,
                "game_state": player_view
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/game/{game_id}/submit-card")
    async def submit_card(game_id: str, user_id: str, white_card: str):
        """Submit a white card for the current round"""
        try:
            game_manager = get_game_manager()
            game_state = await game_manager.submit_card(game_id, user_id, white_card)
            
            # FIXED: Broadcast updated game state to ALL players with their individual views
            # TODO: WebSocket broadcasting moved to main.py
            # await broadcast_game_state_to_all_players(game_id, "card_submitted", {
            #     "submitter_id": user_id,
            #     "submitter_username": game_state.players[user_id].username if user_id in game_state.players else "Unknown"
            # })
            
            return {
                "success": True,
                "game_state": game_manager.get_player_view(game_id, user_id)
            }
        except ValueError as e:
            # FIXED: Provide more specific error messages for common issues
            error_msg = str(e)
            if "Card not in player's hand" in error_msg:
                error_msg = "Card no longer available. Please select a different card."
            elif "Player has already submitted" in error_msg:
                error_msg = "You have already submitted a card for this round."
            elif "Judge cannot submit" in error_msg:
                error_msg = "The judge cannot submit cards during this phase."
            elif "Card submission phase is over" in error_msg:
                error_msg = "Card submission phase has ended."
            elif "Game is not in progress" in error_msg:
                error_msg = "Game is not currently in progress."
            
            raise HTTPException(status_code=400, detail=error_msg)
        except Exception as e:
            print(f"❌ Unexpected error in submit_card: {e}")
            raise HTTPException(status_code=500, detail="An unexpected error occurred while submitting the card.")
    
    @app.post("/game/{game_id}/judge")
    async def judge_round(game_id: str, judge_user_id: str, winning_card: str):
        """Judge the current round"""
        try:
            game_manager = get_game_manager()
            game_state = await game_manager.judge_round(game_id, judge_user_id, winning_card)
            
            # FIXED: Broadcast round complete to all players with individual views
            # TODO: WebSocket broadcasting moved to main.py
            # await broadcast_game_state_to_all_players(game_id, "round_complete", {
            #     "winner_id": game_state.current_round.winner_id if game_state.current_round else None,
            #     "winning_card": game_state.current_round.winning_card if game_state.current_round else None,
            #     "judge_id": judge_user_id
            # })
            
            return {
                "success": True,
                "game_state": game_manager.get_player_view(game_id, judge_user_id)
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/game/{game_id}/start")
    async def start_game(game_id: str, host_user_id: str):
        """Start the game"""
        try:
            game_manager = get_game_manager()
            game_state = await game_manager.start_game(game_id, host_user_id)
            
            # FIXED: Broadcast game started to all connected players with individual views
            # TODO: WebSocket broadcasting moved to main.py
            # await broadcast_game_state_to_all_players(game_id, "game_started", {
            #     "host_id": host_user_id
            # })
            
            return {
                "success": True,
                "game_state": game_manager.get_player_view(game_id, host_user_id)
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/game/{game_id}/leave")
    async def leave_game(game_id: str, user_id: str):
        """Leave the game"""
        try:
            game_manager = get_game_manager()
            await game_manager.leave_game(game_id, user_id)
            
            return {"success": True, "message": "Left game successfully"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    # @app.get("/debug/connections")
    # async def debug_connections():
    #     """Debug endpoint to see active WebSocket connections - DISABLED: moved to main.py"""
    #     # WebSocket connections are now managed in main.py
    #     return {"message": "WebSocket debugging moved to main.py"}
    
    @app.get("/games")
    async def get_all_games():
        """Get all active games"""
        try:
            game_manager = get_game_manager()
            games = game_manager.get_all_games()
            
            return {
                "success": True,
                "games": games
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # WebSocket connection management for authenticated multiplayer (using module-level variable)
    
    async def broadcast_to_authenticated_game(game_id: str, message: dict, exclude_user: str = None):
        """Broadcast message to all players in an authenticated game"""
        if game_id not in active_connections:
            print(f"⚠️ No active WebSocket connections for game {game_id}")
            return
        
        print(f"🚀 Broadcasting to {len(active_connections[game_id])} players in game {game_id}")
        
        # Get fresh game state from authenticated system
        try:
            from .multiplayer_routes import get_game_manager
            from agent_system.models.database import get_db
            db = next(get_db())
            game_manager = get_game_manager(db)
            game_state = game_manager.get_game_state(game_id)
            
            if game_state:
                # Convert to serializable format
                game_data = {
                    "game_id": game_id,
                    "status": game_state.status.value,
                    "players": [
                        {
                            "user_id": player.user_id,
                            "username": player.username,
                            "is_host": player.is_host,
                            "score": player.score
                        }
                        for player in game_state.players.values()
                    ],
                    "max_players": game_state.max_players,
                    "max_score": game_state.max_score,
                    "current_round": game_state.current_round.round_number if game_state.current_round else None
                }
                
                # Merge custom message with game state
                broadcast_message = {**message, "game_state": game_data}
                
                import json
                connections_to_remove = []
                for user_id, websocket in active_connections[game_id].items():
                    if exclude_user and user_id == exclude_user:
                        continue
                    try:
                        await websocket.send_text(json.dumps(broadcast_message))
                        print(f"✅ Broadcasted to {user_id} in game {game_id}")
                    except Exception as e:
                        print(f"❌ Failed to broadcast to {user_id}: {e}")
                        connections_to_remove.append(user_id)
                
                # Clean up dead connections
                for user_id in connections_to_remove:
                    if user_id in active_connections[game_id]:
                        del active_connections[game_id][user_id]
                        print(f"🧹 Removed dead connection for {user_id}")
                
                print(f"✅ Broadcast complete for game {game_id}")
            else:
                print(f"⚠️ Game state not found for {game_id}")
        except Exception as e:
            print(f"❌ Error during broadcast: {e}")
            import traceback
            traceback.print_exc()

    # Set the global broadcast function reference
    def set_broadcast_function():
        global broadcast_function
        broadcast_function = broadcast_to_authenticated_game
    
    set_broadcast_function()

else:
    print("⚠️ Multiplayer game endpoints not available - MultiplayerCAHGame import failed")

@app.get("/cors-test")
async def cors_test():
    """Test CORS middleware"""
    return {
        "message": "CORS test successful",
        "timestamp": datetime.now().isoformat(),
        "cors_working": True
    }

@app.options("/cors-test")
async def cors_test_options():
    """Test CORS preflight"""
    return {"message": "CORS preflight successful"}

# WebSocket endpoint - ALWAYS register this regardless of import status
@app.websocket("/ws/{game_id}/{user_id}")
async def websocket_endpoint(websocket: WebSocket, game_id: str, user_id: str):
    """WebSocket endpoint for real-time game updates with authenticated multiplayer"""
    try:
        await websocket.accept()
        print(f"🔌 WebSocket connected: {user_id} -> {game_id}")
        
                            # Store connection in the authenticated game manager instead of the old system
        try:
            from agent_system.game.authenticated_multiplayer_cah import AuthenticatedMultiplayerCAHGame
            from agent_system.api.multiplayer_routes import get_game_manager
            from agent_system.models.database import get_db
            
            # Get the game manager and add the connection
            db = next(get_db())
            game_manager = get_game_manager(db)
            if hasattr(game_manager, 'add_websocket_connection'):
                game_manager.add_websocket_connection(game_id, int(user_id), websocket)
                print(f"🔍 DEBUG: Stored connection in authenticated game manager for game {game_id}, user {user_id}")
            else:
                print(f"⚠️ WARNING: Game manager doesn't have add_websocket_connection method")
        except Exception as e:
            print(f"❌ ERROR: Failed to store connection in authenticated game manager: {e}")
            # Fallback to old system
            if game_id not in active_connections:
                active_connections[game_id] = {}
            active_connections[game_id][user_id] = websocket
            print(f"🔍 DEBUG: Fallback to old system. active_connections[{game_id}] = {list(active_connections[game_id].keys())}")
        
        # Send connection confirmation (game state will be sent via broadcasts)
        try:
            await websocket.send_text(json.dumps({
                "type": "connection_established",
                "game_id": game_id,
                "user_id": user_id
            }))
            print(f"✅ WebSocket connection established for {user_id}")
        except Exception as e:
            print(f"⚠️ Error sending connection confirmation: {e}")
        
        # Listen for messages
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                print(f"📨 WebSocket message from {user_id}: {message}")
                
                # Handle different message types
                if message.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
                elif message.get("type") == "player_joined":
                    # Just acknowledge the message - broadcasts are handled by the API routes
                    print(f"📨 Player {user_id} joined message received")
                    
            except WebSocketDisconnect:
                print(f"🔌 WebSocket disconnected: {user_id} from {game_id}")
                break
            except Exception as e:
                print(f"❌ WebSocket error: {e}")
                break
                
    except Exception as e:
        print(f"❌ WebSocket connection error: {e}")
    finally:
        # Clean up connection
        if game_id in active_connections and user_id in active_connections[game_id]:
            del active_connections[game_id][user_id]
            print(f"🧹 Cleaned up connection for {user_id}")
        if game_id in active_connections and not active_connections[game_id]:
            del active_connections[game_id]
            print(f"🧹 Cleaned up empty game connections for {game_id}")

# Broadcast functionality is now handled directly in multiplayer_routes.py to avoid circular imports

@app.post("/migrate-database")
async def migrate_database(secret_key: str = ""):
    """Migrate database schema - ADMIN ONLY"""
    if secret_key != "migrate_user_id_2025":
        raise HTTPException(status_code=403, detail="Invalid secret key")
    
    try:
        # Import and run migration
        import os
        from sqlalchemy import create_engine, text
        
        # Get database URL
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            from agent_system.config.settings import settings
            database_url = settings.database_url
        
        engine = create_engine(database_url)
        
        # Check current User.id type
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'users' AND column_name = 'id'
            """))
            
            row = result.fetchone()
            if not row:
                return {"error": "Users table not found"}
            
            current_type = row[1]
            
            if 'character' in current_type.lower() or 'varchar' in current_type.lower():
                return {
                    "status": "already_migrated",
                    "current_type": current_type,
                    "message": "User.id is already String type"
                }
            
            if current_type != 'integer':
                return {
                    "error": f"Unexpected data type: {current_type}",
                    "current_type": current_type
                }
        
        print(f"🚀 Starting migration from {current_type} to VARCHAR...")
        
        # Execute migration with separate connection
        with engine.begin() as conn:
            # Step 1: Drop foreign key constraints
            print("🔗 Dropping foreign key constraints...")
            conn.execute(text("ALTER TABLE game_players DROP CONSTRAINT IF EXISTS game_players_user_id_fkey"))
            conn.execute(text("ALTER TABLE game_rounds DROP CONSTRAINT IF EXISTS game_rounds_judge_user_id_fkey"))
            
            # Step 2: Convert types
            print("🔄 Converting column types...")
            conn.execute(text("ALTER TABLE users ALTER COLUMN id TYPE VARCHAR USING id::VARCHAR"))
            conn.execute(text("ALTER TABLE game_players ALTER COLUMN user_id TYPE VARCHAR USING user_id::VARCHAR"))
            conn.execute(text("ALTER TABLE game_rounds ALTER COLUMN judge_user_id TYPE VARCHAR USING judge_user_id::VARCHAR"))
            
            # Step 3: Recreate foreign keys
            print("🔗 Recreating foreign key constraints...")
            conn.execute(text("ALTER TABLE game_players ADD CONSTRAINT game_players_user_id_fkey FOREIGN KEY (user_id) REFERENCES users(id)"))
            conn.execute(text("ALTER TABLE game_rounds ADD CONSTRAINT game_rounds_judge_user_id_fkey FOREIGN KEY (judge_user_id) REFERENCES users(id)"))
        
        # Verify migration
        print("🔍 Verifying migration...")
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'users' AND column_name = 'id'
            """))
            
            row = result.fetchone()
            new_type = row[1]
            
            return {
                "status": "migration_successful",
                "old_type": current_type,
                "new_type": new_type,
                "message": "User.id successfully migrated to String"
            }
                
    except Exception as e:
        return {"error": f"Database connection failed: {str(e)}"}

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    # Get port from environment or default to 8000
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    print(f"🚀 Starting CAH API server on {host}:{port}")
    print(f"🔗 API will be available at: http://{host}:{port}")
    print(f"📊 Health check: http://{host}:{port}/")
    print(f"🎮 Game endpoints: http://{host}:{port}/game/")
    print(f"🧠 Humor generation: http://{host}:{port}/generate")
    
    # Add timeout protection and better error handling for deployment
    try:
        uvicorn.run(
            app, 
            host=host, 
            port=port,
            timeout_keep_alive=30,
            timeout_graceful_shutdown=30,
            access_log=True
        )
    except Exception as e:
        print(f"❌ Server startup failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 