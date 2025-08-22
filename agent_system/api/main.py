import asyncio
import uuid
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
import json

logger = logging.getLogger(__name__)

# Handle imports for different execution contexts
import sys
from pathlib import Path

# Add the agent_system directory to Python path
current_dir = Path(__file__).parent
agent_system_dir = current_dir.parent
sys.path.insert(0, str(agent_system_dir))

try:
    from models.database import (
        create_database, get_session_local, User, HumorGenerationRequest, 
        HumorGeneration, HumorEvaluation, UserFeedback
    )
    from personas.persona_manager import PersonaManager
    from agents.humor_agents import HumorAgentOrchestrator, HumorRequest
    from config.settings import settings
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print(f"Current directory: {Path.cwd()}")
    print(f"Python path: {sys.path}")
    raise

# Pydantic models for API
class UserCreateRequest(BaseModel):
    username: str
    email: str
    age_range: Optional[str] = None
    occupation: Optional[str] = None
    education_level: Optional[str] = None
    interests: Optional[List[str]] = None

class HumorGenerationApiRequest(BaseModel):
    context: str
    audience: str
    topic: str
    user_id: Optional[int] = None
    humor_type: Optional[str] = None
    num_generators: int = 3
    num_evaluators: int = 2

class UserFeedbackRequest(BaseModel):
    generation_id: int
    liked: bool
    humor_rating: Optional[int] = None  # 1-5 scale
    feedback_text: Optional[str] = None

class PersonaResponse(BaseModel):
    id: int
    name: str
    description: str
    demographics: Dict[str, Any]
    personality_traits: Dict[str, Any]
    expertise_areas: List[str]
    avg_rating: float
    total_generations: int

class HumorGenerationResponse(BaseModel):
    id: int
    text: str
    persona_name: str
    model_used: str
    generation_time: float
    humor_score: float
    creativity_score: float
    appropriateness_score: float
    context_relevance_score: float
    overall_score: float
    evaluations: List[Dict[str, Any]]

# Initialize FastAPI app
app = FastAPI(
    title="Agent-Based Humor Generation API",
    description="Multi-agent system for personalized humor generation with evaluation",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket connection management for authenticated multiplayer
active_websocket_connections: Dict[str, Dict[str, WebSocket]] = {}  # game_id -> {user_id: websocket}

# Global game manager reference for WebSocket synchronization
global_game_manager = None

# Database dependency
def get_db():
    SessionLocal = get_session_local(settings.database_url)
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Global orchestrator instance
orchestrator = None

@app.on_event("startup")
async def startup_event():
    """Initialize the system on startup"""
    global orchestrator, persona_manager
    
    # Create database tables
    create_database(settings.database_url)
    
    # Initialize orchestrator
    SessionLocal = get_session_local(settings.database_url)
    db = SessionLocal()
    try:
        persona_manager = PersonaManager(db)
        orchestrator = HumorAgentOrchestrator(persona_manager)
    finally:
        db.close()

# Health check endpoints
@app.get("/")
async def root():
    return {"message": "CAH API Server", "status": "healthy", "timestamp": datetime.utcnow()}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow()}

# User management endpoints
@app.post("/users", response_model=Dict[str, Any])
async def create_user(user_request: UserCreateRequest, db: Session = Depends(get_db)):
    """Create a new user"""
    # Check if user already exists
    existing_user = db.query(User).filter(
        (User.username == user_request.username) | (User.email == user_request.email)
    ).first()
    
    if existing_user:
        raise HTTPException(status_code=400, detail="User already exists")
    
    user = User(
        username=user_request.username,
        email=user_request.email,
        age_range=user_request.age_range,
        occupation=user_request.occupation,
        education_level=user_request.education_level,
        interests=user_request.interests or [],
        humor_preferences={}
    )
    
    db.add(user)
    db.commit()
    db.refresh(user)
    
    return {"user_id": user.id, "username": user.username, "message": "User created successfully"}

@app.get("/users/{user_id}")
async def get_user(user_id: int, db: Session = Depends(get_db)):
    """Get user information"""
    user = db.query(User).get(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "age_range": user.age_range,
        "occupation": user.occupation,
        "education_level": user.education_level,
        "interests": user.interests,
        "humor_preferences": user.humor_preferences,
        "created_at": user.created_at
    }

# Persona endpoints
@app.get("/personas", response_model=List[PersonaResponse])
async def get_personas(db: Session = Depends(get_db)):
    """Get all generation personas"""
    persona_manager = PersonaManager(db)
    personas = persona_manager.get_generation_personas()
    
    return [
        PersonaResponse(
            id=p.id,
            name=p.name,
            description=p.description,
            demographics=p.demographics or {},
            personality_traits=p.personality_traits or {},
            expertise_areas=p.expertise_areas or [],
            avg_rating=p.avg_rating,
            total_generations=p.total_generations
        )
        for p in personas
    ]

@app.get("/personas/{persona_id}")
async def get_persona(persona_id: int, db: Session = Depends(get_db)):
    """Get specific persona details"""
    persona_manager = PersonaManager(db)
    persona = persona_manager.get_persona_by_id(persona_id)
    
    if not persona:
        raise HTTPException(status_code=404, detail="Persona not found")
    
    return PersonaResponse(
        id=persona.id,
        name=persona.name,
        description=persona.description,
        demographics=persona.demographics or {},
        personality_traits=persona.personality_traits or {},
        expertise_areas=persona.expertise_areas or [],
        avg_rating=persona.avg_rating,
        total_generations=persona.total_generations
    )

# Main humor generation endpoint
@app.post("/generate-humor")
async def generate_humor(
    request: HumorGenerationApiRequest, 
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Generate humor using multiple agents and evaluate with multiple evaluators"""
    global orchestrator
    
    if not orchestrator:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    # Create humor request
    humor_request = HumorRequest(
        context=request.context,
        audience=request.audience,
        topic=request.topic,
        user_id=request.user_id,
        humor_type=request.humor_type
    )
    
    # Generate and evaluate humor
    try:
        result = await orchestrator.generate_and_evaluate_humor(
            humor_request,
            num_generators=request.num_generators,
            num_evaluators=request.num_evaluators
        )
        
        if not result['success']:
            raise HTTPException(status_code=500, detail=result.get('error', 'Generation failed'))
        
        # Ensure result has required fields
        if not result.get('results') or len(result['results']) == 0:
            raise HTTPException(status_code=500, detail='No humor generated')
        
        # Store results in database asynchronously
        background_tasks.add_task(
            store_generation_results, 
            db, 
            humor_request, 
            result
        )
        
        # Format response
        response_data = {
            "success": True,
            "request_id": str(uuid.uuid4()),
            "total_generations": result['total_generations'],
            "top_results": [],
            "generation_personas": result['generation_personas'],
            "evaluation_personas": result['evaluation_personas']
        }
        
        # Format top results
        for ranked_result in result['top_results']:
            generation = ranked_result['generation']
            avg_scores = ranked_result['average_scores']
            
            response_data['top_results'].append(HumorGenerationResponse(
                id=0,  # Will be set after database storage
                text=generation.text,
                persona_name=generation.persona_name,
                model_used=generation.model_used,
                generation_time=generation.generation_time,
                humor_score=avg_scores['humor_score'],
                creativity_score=avg_scores['creativity_score'],
                appropriateness_score=avg_scores['appropriateness_score'],
                context_relevance_score=avg_scores['context_relevance_score'],
                overall_score=avg_scores['overall_score'],
                evaluations=[
                    {
                        "evaluator_name": eval_result.evaluator_name,
                        "model_used": eval_result.model_used,
                        "scores": {
                            "humor_score": eval_result.humor_score,
                            "creativity_score": eval_result.creativity_score,
                            "appropriateness_score": eval_result.appropriateness_score,
                            "context_relevance_score": eval_result.context_relevance_score,
                            "overall_score": eval_result.overall_score
                        },
                        "reasoning": eval_result.reasoning
                    }
                    for eval_result in ranked_result['evaluations']
                ]
            ))
        
        return response_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

# User feedback endpoint
@app.post("/feedback")
async def submit_feedback(
    feedback_request: UserFeedbackRequest, 
    db: Session = Depends(get_db)
):
    """Submit user feedback for a generated humor"""
    # Get the generation
    generation = db.query(HumorGeneration).get(feedback_request.generation_id)
    if not generation:
        raise HTTPException(status_code=404, detail="Generation not found")
    
    # Get the user
    user_id = generation.request.user_id
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID required for feedback")
    
    # Create feedback record
    feedback = UserFeedback(
        user_id=user_id,
        generation_id=feedback_request.generation_id,
        liked=feedback_request.liked,
        humor_rating=feedback_request.humor_rating,
        feedback_text=feedback_request.feedback_text
    )
    
    db.add(feedback)
    db.commit()
    
    # Update persona preferences
    persona_manager = PersonaManager(db)
    feedback_score = 1.0 if feedback_request.liked else 0.0
    
    # Weight by humor rating if provided
    if feedback_request.humor_rating:
        feedback_score = feedback_request.humor_rating / 5.0  # Normalize to 0-1
    
    persona_manager.update_persona_preference(
        user_id=user_id,
        persona_id=generation.persona_id,
        feedback=feedback_score,
        context=generation.request.context
    )
    
    return {"message": "Feedback submitted successfully"}

# Analytics endpoints
@app.get("/analytics/user/{user_id}")
async def get_user_analytics(user_id: int, db: Session = Depends(get_db)):
    """Get analytics for a specific user"""
    user = db.query(User).get(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Get generation history
    generations_count = db.query(HumorGeneration).join(HumorGenerationRequest).filter(
        HumorGenerationRequest.user_id == user_id
    ).count()
    
    # Get feedback history
    feedback_history = db.query(UserFeedback).filter(UserFeedback.user_id == user_id).all()
    
    total_feedback = len(feedback_history)
    positive_feedback = sum(1 for f in feedback_history if f.liked)
    
    # Get top personas for this user
    persona_manager = PersonaManager(db)
    top_personas = persona_manager.get_personalized_personas(user_id, "", limit=5)
    
    return {
        "user_id": user_id,
        "total_generations": generations_count,
        "total_feedback": total_feedback,
        "positive_feedback_rate": positive_feedback / total_feedback if total_feedback > 0 else 0,
        "top_personas": [
            {"id": p.id, "name": p.name, "description": p.description}
            for p in top_personas
        ],
        "humor_preferences": user.humor_preferences
    }

@app.get("/analytics/personas")
async def get_persona_analytics(db: Session = Depends(get_db)):
    """Get analytics for all personas"""
    persona_manager = PersonaManager(db)
    personas = persona_manager.get_generation_personas()
    
    analytics = []
    for persona in personas:
        # Get recent performance metrics
        total_generations = persona.total_generations
        avg_rating = persona.avg_rating
        
        analytics.append({
            "id": persona.id,
            "name": persona.name,
            "total_generations": total_generations,
            "average_rating": avg_rating,
            "expertise_areas": persona.expertise_areas
        })
    
    # Sort by performance
    analytics.sort(key=lambda x: x["average_rating"], reverse=True)
    
    return {"personas": analytics}

# Background task functions
async def store_generation_results(db: Session, humor_request: HumorRequest, result: Dict[str, Any]):
    """Store generation results in database"""
    try:
        # Create generation request record
        request_record = HumorGenerationRequest(
            user_id=humor_request.user_id,
            context=humor_request.context,
            target_audience=humor_request.audience,
            humor_type=humor_request.humor_type
        )
        db.add(request_record)
        db.flush()  # To get the ID
        
        # Store each generation and its evaluations
        for ranked_result in result['all_results']:
            generation = ranked_result['generation']
            evaluations = ranked_result['evaluations']
            
            # Create generation record
            generation_record = HumorGeneration(
                request_id=request_record.id,
                persona_id=generation.persona_id,
                generated_text=generation.text,
                model_used=generation.model_used,
                generation_time=generation.generation_time
            )
            db.add(generation_record)
            db.flush()  # To get the ID
            
            # Store evaluations
            for evaluation in evaluations:
                evaluation_record = HumorEvaluation(
                    generation_id=generation_record.id,
                    evaluator_persona_id=1,  # TODO: Get actual evaluator persona ID
                    humor_score=evaluation.humor_score,
                    creativity_score=evaluation.creativity_score,
                    appropriateness_score=evaluation.appropriateness_score,
                    context_relevance_score=evaluation.context_relevance_score,
                    overall_score=evaluation.overall_score,
                    evaluation_reasoning=evaluation.reasoning,
                    model_used=evaluation.model_used,
                    evaluation_time=evaluation.evaluation_time
                )
                db.add(evaluation_record)
        
        db.commit()
        
    except Exception as e:
        db.rollback()
        print(f"Error storing generation results: {e}")

# Import authentication routes
try:
    from api.auth_routes import router as auth_router
    print("✅ Authentication routes imported successfully")
    
    # Add authentication routes
    app.include_router(auth_router)
    print("✅ Authentication routes added to API")
except ImportError as e:
    print(f"❌ Authentication routes not available: {e}")
    print(f"Error details: {e}")

# Import multiplayer game functionality
try:
    from game.authenticated_multiplayer_cah import AuthenticatedMultiplayerCAHGame
    from api.multiplayer_routes import get_game_manager
    print("✅ Authenticated multiplayer game imported successfully")
    
    # Import multiplayer routes
    try:
        from api.multiplayer_routes import router as multiplayer_router
        print("✅ Multiplayer routes imported successfully")
        
        # Add multiplayer routes
        app.include_router(multiplayer_router)
        print("✅ Multiplayer routes added to API")
    except ImportError as e:
        print(f"⚠️ Multiplayer routes not available: {e}")
        multiplayer_router = None
    
    # Import Google OAuth routes if available
    try:
        from api.google_oauth_routes import router as google_oauth_router
        print("✅ Google OAuth routes imported successfully")
        
        # Add Google OAuth routes
        app.include_router(google_oauth_router)
        print("✅ Google OAuth routes added to API")
    except ImportError as e:
        print(f"⚠️ Google OAuth routes not available: {e}")
        google_oauth_router = None
    
    # Initialize game manager on startup
    @app.on_event("startup")
    async def initialize_multiplayer():
        global orchestrator, persona_manager
        if orchestrator and persona_manager:
            logger.info("✅ Multiplayer system initialized")
    
    # Multiplayer Game Endpoints
    @app.post("/game/create")
    async def create_multiplayer_game(
        game_id: str,
        host_username: str,
        settings: Optional[Dict[str, Any]] = None
    ):
        """Create a new multiplayer game"""
        try:
            game_manager = get_game_manager()
            host_user_id = f"user_{hash(host_username) % 10000}"
            
            game_state = await game_manager.create_game(
                game_id=game_id,
                host_user_id=host_user_id,
                host_username=host_username,
                settings=settings or {}
            )
            
            return {
                "success": True,
                "game_id": game_id,
                "host_user_id": host_user_id,
                "status": game_state.status.value
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/game/join")
    async def join_multiplayer_game(game_id: str, username: str):
        """Join an existing game"""
        try:
            game_manager = get_game_manager()
            user_id = f"user_{hash(username) % 10000}"
            
            game_state = await game_manager.join_game(game_id, user_id, username)
            player_view = game_manager.get_player_view(game_id, user_id)
            
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

    # WebSocket endpoint for authenticated multiplayer games
    @app.websocket("/ws/{game_id}/{user_id}")
    async def websocket_endpoint(websocket: WebSocket, game_id: str, user_id: str):
        """WebSocket endpoint for real-time game updates"""
        try:
            await websocket.accept()
            logger.info(f"🔌 WebSocket connected: {user_id} -> {game_id}")
            
            # Store connection in main.py's connection list
            if game_id not in active_websocket_connections:
                active_websocket_connections[game_id] = {}
            active_websocket_connections[game_id][user_id] = websocket
            
            # ALSO store connection in game manager for proper synchronization
            try:
                db = next(get_db())
                try:
                    from .multiplayer_routes import get_game_manager
                    game_manager = get_game_manager(db)
                    
                    # Add WebSocket connection to game manager
                    if hasattr(game_manager, 'add_websocket_connection'):
                        game_manager.add_websocket_connection(game_id, user_id, websocket)
                        logger.info(f"✅ Added WebSocket connection to game manager for user {user_id}")
                    else:
                        logger.warning(f"⚠️ Game manager doesn't have add_websocket_connection method")
                    
                    # Send initial game state
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
                                    "is_judge": player.is_judge,
                                    "score": player.score,
                                    "connected": player.connected
                                }
                                for player in game_state.players.values()
                            ],
                            "max_players": game_state.max_players,
                            "max_score": game_state.max_score,
                            "current_round": {
                                "round_number": game_state.current_round.round_number,
                                "black_card": game_state.current_round.black_card,
                                "judge_id": game_state.current_round.judge_id,
                                "phase": game_state.current_round.phase.value,
                                "submissions_count": len(game_state.current_round.submissions)
                            } if game_state.current_round else None
                        }
                        
                        await websocket.send_text(json.dumps({
                            "type": "initial_state",
                            "game_state": game_data
                        }))
                        logger.info(f"✅ Sent initial game state to {user_id}")
                    else:
                        logger.warning(f"⚠️ No game state found for game {game_id}")
                        
                finally:
                    db.close()
            except Exception as e:
                logger.error(f"Failed to send initial game state: {e}")
            
            # Listen for messages
            while True:
                try:
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    logger.info(f"📨 WebSocket message from {user_id}: {message}")
                    
                    # Handle different message types
                    if message.get("type") == "player_joined":
                        # Broadcast player joined to all players
                        await broadcast_to_authenticated_game(game_id, {
                            "type": "player_joined",
                            "user_id": user_id
                        })
                        
                except WebSocketDisconnect:
                    logger.info(f"🔌 WebSocket disconnected: {user_id} from {game_id}")
                    break
                except Exception as e:
                    logger.error(f"❌ WebSocket error: {e}")
                    break
                    
        except Exception as e:
            logger.error(f"❌ WebSocket connection error: {e}")
        finally:
            # Clean up connection from both places
            if game_id in active_websocket_connections and user_id in active_websocket_connections[game_id]:
                del active_websocket_connections[game_id][user_id]
                if not active_websocket_connections[game_id]:
                    del active_websocket_connections[game_id]
                logger.info(f"🧹 Cleaned up WebSocket connection for {user_id} in game {game_id}")
            
            # Also clean up from game manager
            try:
                db = next(get_db())
                try:
                    from .multiplayer_routes import get_game_manager
                    game_manager = get_game_manager(db)
                    if hasattr(game_manager, 'remove_websocket_connection'):
                        game_manager.remove_websocket_connection(game_id, user_id)
                        logger.info(f"🧹 Cleaned up WebSocket connection from game manager for user {user_id}")
                finally:
                    db.close()
            except Exception as e:
                logger.error(f"Failed to clean up WebSocket connection from game manager: {e}")

    async def broadcast_to_authenticated_game(game_id: str, message: dict, exclude_user: str = None):
        """Broadcast message to all players in an authenticated game"""
        if game_id not in active_websocket_connections:
            logger.warning(f"⚠️ No active WebSocket connections for game {game_id}")
            return
        
        logger.info(f"🚀 Broadcasting to {len(active_websocket_connections[game_id])} players in game {game_id}")
        
        # Get fresh game state
        db = next(get_db())
        try:
            from .multiplayer_routes import get_game_manager
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
                            "is_judge": player.is_judge,
                            "score": player.score,
                            "connected": player.connected
                        }
                        for player in game_state.players.values()
                    ],
                    "max_players": game_state.max_players,
                    "max_score": game_state.max_score,
                    "current_round": {
                        "round_number": game_state.current_round.round_number,
                        "black_card": game_state.current_round.black_card,
                        "judge_id": game_state.current_round.judge_id,
                        "phase": game_state.current_round.phase.value,
                        "submissions_count": len(game_state.current_round.submissions)
                    } if game_state.current_round else None
                }
                
                # Add game state to message
                message["game_state"] = game_data
                
                # Broadcast to all connected players
                connections_to_remove = []
                for user_id, websocket in active_websocket_connections[game_id].items():
                    if exclude_user and user_id == exclude_user:
                        continue
                        
                    try:
                        await websocket.send_text(json.dumps(message))
                        logger.info(f"✅ Broadcasted to {user_id} in game {game_id}")
                    except Exception as e:
                        logger.error(f"❌ Failed to broadcast to {user_id}: {e}")
                        connections_to_remove.append(user_id)
                
                # Clean up dead connections
                for user_id in connections_to_remove:
                    del active_websocket_connections[game_id][user_id]
                    
        finally:
            db.close()

    async def broadcast_to_authenticated_game(game_id: str, message: dict, exclude_user: str = None):
        """Broadcast message to all players in an authenticated game"""
        if game_id not in active_websocket_connections:
            logger.warning(f"⚠️ No active WebSocket connections for game {game_id}")
            return
        
        logger.info(f"🚀 Broadcasting to {len(active_websocket_connections[game_id])} players in game {game_id}")
        
        # Get fresh game state
        db = next(get_db())
        try:
            from .multiplayer_routes import get_game_manager
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
                            "is_judge": player.is_judge,
                            "score": player.score,
                            "connected": player.connected
                        }
                        for player in game_state.players.values()
                    ],
                    "max_players": game_state.max_players,
                    "max_score": game_state.max_score,
                    "current_round": {
                        "round_number": game_state.current_round.round_number,
                        "black_card": game_state.current_round.black_card,
                        "judge_id": game_state.current_round.judge_id,
                        "phase": game_state.current_round.phase.value,
                        "submissions_count": len(game_state.current_round.submissions)
                    } if game_state.current_round else None
                }
                
                # Add game state to message
                message["game_state"] = game_data
                
                # Broadcast to all connected players
                connections_to_remove = []
                for user_id, websocket in active_websocket_connections[game_id].items():
                    if exclude_user and user_id == exclude_user:
                        continue
                        
                    try:
                        await websocket.send_text(json.dumps(message))
                        logger.info(f"✅ Broadcasted to {user_id} in game {game_id}")
                    except Exception as e:
                        logger.error(f"❌ Failed to broadcast to {user_id}: {e}")
                        connections_to_remove.append(user_id)
                
                # Clean up dead connections
                for user_id in connections_to_remove:
                    del active_websocket_connections[game_id][user_id]
                    
        finally:
            db.close()

    def sync_websocket_connections_with_game_manager(game_manager):
        """Sync WebSocket connections between main.py and game manager"""
        global global_game_manager
        global_game_manager = game_manager
        
        # Sync existing connections
        for game_id, connections in active_websocket_connections.items():
            for user_id, websocket in connections.items():
                try:
                    # Add to game manager with string user_id
                    if hasattr(game_manager, 'add_websocket_connection'):
                        game_manager.add_websocket_connection(game_id, user_id, websocket)
                        logger.info(f"✅ Synced WebSocket connection for user {user_id} in game {game_id}")
                except Exception as e:
                    logger.error(f"❌ Failed to sync WebSocket connection for user {user_id}: {e}")
        
        logger.info(f"✅ Synced {sum(len(conns) for conns in active_websocket_connections.values())} WebSocket connections with game manager")

except ImportError as e:
    logger.error(f"❌ Multiplayer game functionality not available: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 