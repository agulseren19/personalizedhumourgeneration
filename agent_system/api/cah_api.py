#!/usr/bin/env python3
"""
Cards Against Humanity API Server
Exposes the fixed complete humor system as REST APIs
"""

import asyncio
import json
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import time

# Import our humor system
import sys
import os

# Add the parent directory (CAH) to Python path so we can import agent_system
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, parent_dir)

from agent_system.agents.improved_humor_agents import (
    ImprovedHumorOrchestrator, 
    HumorRequest, 
    ContentFilter
)
from agent_system.knowledge.improved_aws_knowledge_base import improved_aws_knowledge_base
from agent_system.personas.enhanced_persona_templates import get_all_personas
from agent_system.personas.dynamic_persona_generator import dynamic_persona_generator

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

# Initialize FastAPI app
app = FastAPI(
    title="Cards Against Humanity API",
    description="AI-powered humor generation with personalized learning",
    version="1.0.0"
)

# Add CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],  # Next.js dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize humor system
humor_system = None
content_filter = None

@app.on_event("startup")
async def startup_event():
    """Initialize the humor system on startup"""
    global humor_system, content_filter
    print("🚀 Starting CAH API Server...")
    
    humor_system = ImprovedHumorOrchestrator()
    content_filter = ContentFilter()
    
    print("✅ CAH API Server ready!")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Cards Against Humanity API is running!", "status": "healthy"}

@app.get("/test-humor")
async def test_humor():
    """Test humor generation system"""
    try:
        if humor_system is None:
            return {"error": "Humor system not initialized"}
        
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
            "humor_system_type": str(type(humor_system)),
            "result_keys": list(result.keys()),
            "result_success": result.get('success'),
            "num_results": len(result.get('results', []))
        }
        
    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }

@app.get("/personas")
async def get_personas():
    """Get all available personas, including dynamic ones"""
    try:
        personas = get_all_personas()
        dynamic_personas = dynamic_persona_generator.get_all_personas()
        all_personas = {**personas, **dynamic_personas}
        return {
            "success": True,
            "personas": [
                {
                    "id": key,
                    "name": persona.name,
                    "description": persona.description,
                    "expertise": persona.expertise_areas,
                    "humor_style": persona.humor_style
                }
                for key, persona in all_personas.items()
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting personas: {str(e)}")

@app.post("/generate")
async def generate_humor(request: GenerateHumorRequest):
    """Generate humor based on context and user preferences"""
    try:
        # Check if humor system is initialized
        if humor_system is None:
            return {"error": "Humor system not initialized yet"}
        
        start_time = time.time()
        
        # Create humor request
        humor_request = HumorRequest(
            context=request.context,
            audience=request.audience,
            topic=request.topic,
            user_id=request.user_id,
            card_type=request.card_type
        )
        
        # Generate humor
        result = await humor_system.generate_and_evaluate_humor(humor_request)
        generation_time = time.time() - start_time
        
        if result['success']:
            try:
                # Format generations for frontend
                generations = []
                
                # Handle both old and new response structures
                results_key = 'top_results' if 'top_results' in result else 'results'
                results_data = result.get(results_key, [])
                
                for i, evaluated_result in enumerate(results_data):
                    try:
                        # Handle both old and new result structures
                        if 'generation' in evaluated_result and 'evaluation' in evaluated_result:
                            # New structure
                            generation = evaluated_result['generation']
                            evaluation = evaluated_result['evaluation']
                        elif 'generation' in evaluated_result and 'evaluations' in evaluated_result:
                            # Old structure
                            generation = evaluated_result['generation']
                            evaluations = evaluated_result['evaluations']
                            evaluation = evaluations[0] if evaluations else None
                        else:
                            # Fallback structure
                            generation = evaluated_result.get('generation', evaluated_result)
                            evaluation = evaluated_result.get('evaluation', evaluated_result)
                        
                        # Extract scores with fallbacks
                        humor_score = getattr(evaluation, 'humor_score', 7.0) if evaluation else 7.0
                        creativity_score = getattr(evaluation, 'creativity_score', 7.0) if evaluation else 7.0
                        appropriateness_score = getattr(evaluation, 'appropriateness_score', 7.0) if evaluation else 7.0
                        context_relevance_score = getattr(evaluation, 'context_relevance_score', 7.0) if evaluation else 7.0
                        overall_score = getattr(evaluation, 'overall_score', 7.0) if evaluation else 7.0
                        surprise_index = getattr(evaluation, 'surprise_index', 5.0) if evaluation else 5.0
                        
                        generation_dict = {
                            "id": f"{generation.persona_name}_{getattr(generation, 'model_used', 'unknown')}_{int(time.time())}",
                            "text": str(getattr(generation, 'text', '')),
                            "persona_name": str(getattr(generation, 'persona_name', 'Unknown')),
                            "model_used": str(getattr(generation, 'model_used', 'unknown')),
                            "score": convert_numpy_types(overall_score),
                            "humor_score": convert_numpy_types(humor_score),
                            "creativity_score": convert_numpy_types(creativity_score),
                            "appropriateness_score": convert_numpy_types(appropriateness_score),
                            "context_relevance_score": convert_numpy_types(context_relevance_score),
                            "surprise_index": convert_numpy_types(surprise_index),  # Add surprise index
                            "is_safe": bool(getattr(generation, 'is_safe', True)),
                            "toxicity_score": convert_numpy_types(getattr(generation, 'toxicity_score', 0.0)),
                            "reasoning": str(getattr(evaluation, 'reasoning', 'Generated with AI humor system')) if evaluation else 'Generated with AI humor system'
                        }
                        
                        # Ensure all values are properly converted
                        generation_dict = convert_numpy_types(generation_dict)
                        
                        generations.append(generation_dict)
                        
                    except Exception as e:
                        print(f"Error processing result {i}: {str(e)}")
                        print(f"Result structure: {evaluated_result}")
                        continue
                
                # Create response data
                response_data = {
                    "success": True,
                    "generations": generations,
                    "recommended_personas": result.get('recommended_personas', []),
                    "generation_time": convert_numpy_types(generation_time),
                    "evaluator_insights": result.get('evaluator_insights', {}),  # Add evaluator insights
                    "error": None
                }
                
                # Ensure all response data is properly serializable
                response_data = convert_numpy_types(response_data)
                
                return response_data
                
            except Exception as e:
                return {"error": f"Error in response formatting: {str(e)}", "type": "response_formatting"}
        else:
            return HumorResponse(
                success=False,
                generations=[],
                recommended_personas=[],
                generation_time=generation_time,
                error=result.get('error', 'Unknown error')
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating humor: {str(e)}")

@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """Submit user feedback for a generation"""
    try:
        await improved_aws_knowledge_base.update_user_feedback(
            user_id=request.user_id,
            persona_name=request.persona_name,
            feedback_score=request.feedback_score,
            context=request.context,
            response_text=request.response_text,
            topic=request.topic,
            audience=request.audience
        )
        
        return {
            "success": True,
            "message": "Feedback recorded successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error recording feedback: {str(e)}")

@app.get("/analytics/{user_id}")
async def get_user_analytics(user_id: str):
    """Get user analytics and learning data"""
    try:
        analytics = await improved_aws_knowledge_base.get_user_analytics(user_id)
        
        if 'error' in analytics:
            return {"success": False, "error": analytics['error']}
        
        return {
            "success": True,
            "analytics": analytics
        }
    except Exception as e:
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

@app.get("/group/{group_id}")
async def get_group_context(group_id: str):
    """Get group context and consensus"""
    try:
        # This would implement group humor generation
        return {
            "success": True,
            "message": "Group context feature coming soon"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting group context: {str(e)}")

@app.get("/debug-ratings/{user_id}")
async def debug_user_ratings(user_id: str):
    """Debug endpoint to show detailed rating calculations"""
    try:
        user_pref = await improved_aws_knowledge_base.get_user_preference(user_id)
        if not user_pref:
            return {"error": "User not found"}
        
        # Get detailed breakdown for each persona
        persona_details = {}
        for persona, score in user_pref.persona_scores.items():
            # Get all interactions for this persona
            persona_interactions = [
                interaction for interaction in user_pref.interaction_history
                if interaction['persona_name'] == persona
            ]
            
            # Calculate what the average should be
            if persona_interactions:
                raw_scores = [i['feedback_score'] for i in persona_interactions]
                simple_average = sum(raw_scores) / len(raw_scores)
                
                # Show step-by-step weighted average calculation
                weighted_steps = []
                current_weighted = raw_scores[0] if raw_scores else 0
                for i, new_score in enumerate(raw_scores[1:], 1):
                    old_weighted = current_weighted
                    current_weighted = (current_weighted * 0.5) + (new_score * 0.5)
                    weighted_steps.append({
                        "step": i + 1,
                        "new_score": new_score,
                        "old_weighted": round(old_weighted, 2),
                        "new_weighted": round(current_weighted, 2),
                        "calculation": f"({old_weighted:.2f} * 0.5) + ({new_score} * 0.5) = {current_weighted:.2f}"
                    })
                
                persona_details[persona] = {
                    "current_score": score,
                    "interaction_count": len(persona_interactions),
                    "raw_scores": raw_scores,
                    "simple_average": simple_average,
                    "weighted_steps": weighted_steps,
                    "status": "liked" if persona in user_pref.liked_personas else 
                             "disliked" if persona in user_pref.disliked_personas else "neutral",
                    "threshold_info": {
                        "like_threshold": "≥ 7.0",
                        "dislike_threshold": "≤ 4.0",
                        "meets_like": score >= 7.0,
                        "meets_dislike": score <= 4.0
                    },
                    "interactions": persona_interactions
                }
        
        return {
            "user_id": user_id,
            "total_interactions": len(user_pref.interaction_history),
            "persona_details": persona_details,
            "thresholds": {
                "like_threshold": 7.0,
                "dislike_threshold": 4.0,
                "min_interactions": 1
            }
        }
        
    except Exception as e:
        return {"error": f"Debug failed: {str(e)}"}

if __name__ == "__main__":
    uvicorn.run(
        "agent_system.api.cah_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 