#!/usr/bin/env python3
"""
PostgreSQL Knowledge Base Integration
Uses PostgreSQL for user preferences, feedback storage, and learning
"""

import json
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import hashlib
import os
import threading
import time
from collections import defaultdict
import random

@dataclass
class UserPreference:
    user_id: str
    humor_styles: List[str]
    liked_personas: List[str]
    disliked_personas: List[str]
    context_preferences: Dict[str, float]
    demographic_profile: Dict[str, Any]
    interaction_history: List[Dict[str, Any]]
    persona_scores: Dict[str, float]
    last_updated: datetime

@dataclass
class FeedbackData:
    user_id: str
    persona_name: str
    context: str
    score: float
    timestamp: datetime
    improvement_suggestions: List[str]

class PostgreSQLKnowledgeBase:
    """PostgreSQL-powered knowledge base for user preferences and feedback"""
    
    def __init__(self):
        self.like_threshold = 7.0  # Score >= 7 = liked
        self.dislike_threshold = 4.0  # Score <= 4 = disliked
        self.min_interactions = 1  # Single interaction is enough for classification
    
    async def get_user_preference(self, user_id: str) -> Optional[UserPreference]:
        """Get user preferences from PostgreSQL"""
        try:
            from agent_system.models.database import get_session_local, User, PersonaPreference, Persona
            from agent_system.config.settings import settings
            
            SessionLocal = get_session_local(settings.database_url)
            db = SessionLocal()
            
            try:
                # Get user
                user = db.query(User).filter(User.id == user_id).first()
                if not user:
                    return None
                
                # Get persona preferences
                persona_prefs = db.query(PersonaPreference).filter(
                    PersonaPreference.user_id == user_id
                ).all()
                
                # Build user preference object
                liked_personas = []
                disliked_personas = []
                persona_scores = {}
                
                for pref in persona_prefs:
                    persona = db.query(Persona).filter(Persona.id == pref.persona_id).first()
                    if persona:
                        persona_name = persona.name
                        persona_scores[persona_name] = pref.preference_score
                        
                        if pref.preference_score >= self.like_threshold:
                            liked_personas.append(persona_name)
                        elif pref.preference_score <= self.dislike_threshold:
                            disliked_personas.append(persona_name)
                
                # Get interaction history from feedback
                from agent_system.models.database import UserFeedback
                feedback_history = db.query(UserFeedback).filter(
                    UserFeedback.user_id == user_id
                ).order_by(UserFeedback.created_at.desc()).limit(50).all()
                
                interaction_history = []
                context_preferences = {}
                
                for feedback in feedback_history:
                    interaction = {
                        'persona_name': feedback.persona_name,
                        'context': feedback.context,
                        'feedback_score': feedback.feedback_score,
                        'response_text': feedback.response_text,
                        'topic': feedback.topic,
                        'audience': feedback.audience,
                        'timestamp': feedback.created_at.isoformat(),
                        'user_id': feedback.user_id
                    }
                    interaction_history.append(interaction)
                    
                    # Update context preferences
                    if feedback.context:
                        context_keywords = feedback.context.lower().split()
                        for keyword in context_keywords:
                            if len(keyword) > 3:
                                if keyword in context_preferences:
                                    context_preferences[keyword] = (context_preferences[keyword] + feedback.feedback_score) / 2
                                else:
                                    context_preferences[keyword] = feedback.feedback_score
                
                user_pref = UserPreference(
                    user_id=user_id,
                    humor_styles=[],
                    liked_personas=liked_personas,
                    disliked_personas=disliked_personas,
                    context_preferences=context_preferences,
                    demographic_profile={},
                    interaction_history=interaction_history,
                    persona_scores=persona_scores,
                    last_updated=datetime.now()
                )
                
                return user_pref
                
            finally:
                db.close()
                
        except Exception as e:
            print(f"Error getting user preference: {e}")
            return None
    
    async def update_user_feedback(self, user_id: str, persona_name: str, feedback_score: float, context: str, response_text: str = "", topic: str = "", audience: str = "") -> bool:
        """Update user feedback and properly calculate likes/dislikes"""
        print(f"  Updating feedback: {user_id} -> {persona_name}: {feedback_score}/10")
        
        try:
            from agent_system.models.database import get_session_local, UserFeedback, User, PersonaPreference, Persona
            from agent_system.config.settings import settings
            from sqlalchemy import and_
            
            SessionLocal = get_session_local(settings.database_url)
            db = SessionLocal()
            
            try:
                # Get current user preference
                user = db.query(User).filter(User.id == user_id).first()
                if not user:
                    user = User(id=user_id) # Create user if not found
                    db.add(user)
                    db.commit()
                    print(f"  User {user_id} not found, creating new user.")
                
                # Add new feedback to interaction history
                feedback = UserFeedback(
                    user_id=user_id,
                    persona_name=persona_name,
                    feedback_score=feedback_score,
                    context=context,
                    response_text=response_text,
                    topic=topic,
                    audience=audience,
                    created_at=datetime.now()
                )
                db.add(feedback)
                
                # Update persona scores
                persona = db.query(Persona).filter(Persona.name == persona_name).first()
                if persona:
                    persona_pref = db.query(PersonaPreference).filter(
                        and_(
                            PersonaPreference.user_id == user_id,
                            PersonaPreference.persona_id == persona.id
                        )
                    ).first()
                    
                    if persona_pref:
                        old_count = persona_pref.interaction_count
                        old_score = persona_pref.preference_score
                        
                        new_count = old_count + 1
                        new_score = ((old_score * old_count) + feedback_score) / new_count
                        
                        persona_pref.interaction_count = new_count
                        persona_pref.preference_score = new_score
                        persona_pref.last_interaction = datetime.now()
                    else:
                        # Create new preference if it doesn't exist
                        new_preference = PersonaPreference(
                            user_id=user_id,
                            persona_id=persona.id,
                            interaction_count=1,
                            preference_score=feedback_score,
                            last_interaction=datetime.now()
                        )
                        db.add(new_preference)
                
                db.commit()
                print(f"  ✅ Database save successful for {user_id} -> {persona_name}")
                
                return True
                
            except Exception as db_error:
                print(f"  ⚠️  Database save failed: {db_error}")
                db.rollback()
                return False
                
            finally:
                db.close()
                
        except Exception as e:
            print(f"Error updating feedback: {e}")
            return False
    
    def _update_persona_preferences(self, user_pref: UserPreference, persona_name: str):
        """Update liked/disliked personas based on feedback patterns"""
        # Get interaction count for this persona
        persona_interactions = [
            interaction for interaction in user_pref.interaction_history
            if interaction['persona_name'] == persona_name
        ]
        
        if len(persona_interactions) < self.min_interactions:
            return  # Not enough data yet
        
        # Calculate average score for this persona
        avg_score = user_pref.persona_scores.get(persona_name, 5.0)
        
        # Update liked/disliked lists
        if avg_score >= self.like_threshold and persona_name not in user_pref.liked_personas:
            user_pref.liked_personas.append(persona_name)
            if persona_name in user_pref.disliked_personas:
                user_pref.disliked_personas.remove(persona_name)
        elif avg_score <= self.dislike_threshold and persona_name not in user_pref.disliked_personas:
            user_pref.disliked_personas.append(persona_name)
            if persona_name in user_pref.liked_personas:
                user_pref.liked_personas.remove(persona_name)
    
    def _update_context_preferences(self, user_pref: UserPreference, context: str, score: float):
        """Update context preferences based on feedback"""
        # Simple keyword-based context scoring
        context_keywords = context.lower().split()
        
        for keyword in context_keywords:
            if len(keyword) > 3:  # Ignore short words
                if keyword in user_pref.context_preferences:
                    # Weighted average
                    current = user_pref.context_preferences[keyword]
                    user_pref.context_preferences[keyword] = (current * 0.8) + (score * 0.2)
                else:
                    user_pref.context_preferences[keyword] = score
    
    async def get_persona_recommendations(self, user_id: str, context: str, audience: str) -> List[str]:
        """Get persona recommendations based on user preferences and context"""
        user_pref = await self.get_user_preference(user_id)
        
        if not user_pref:
            # Fallback to context-based recommendations
            try:
                from agent_system.personas.enhanced_persona_templates import recommend_personas_for_context
                return recommend_personas_for_context(context, audience, "general")
            except ImportError:
                try:
                    from personas.enhanced_persona_templates import recommend_personas_for_context
                    return recommend_personas_for_context(context, audience, "general")
                except ImportError:
                    try:
                        from agent_system.personas.enhanced_persona_templates import recommend_personas_for_context
                        return recommend_personas_for_context(context, audience, "general")
                    except ImportError:
                        print("⚠️  Could not import recommend_personas_for_context, using fallback")
                        return ["General Comedian", "Witty Observer", "Sarcastic Commentator"]
        
        # Get all available personas
        try:
            from agent_system.personas.enhanced_persona_templates import get_all_personas
            all_personas = list(get_all_personas().keys())
        except ImportError:
            try:
                from personas.enhanced_persona_templates import get_all_personas
                all_personas = list(get_all_personas().keys())
            except ImportError:
                try:
                    from agent_system.personas.enhanced_persona_templates import get_all_personas
                    all_personas = list(get_all_personas().keys())
                except ImportError:
                    print("⚠️  Could not import get_all_personas, using fallback")
                    all_personas = ["General Comedian", "Witty Observer", "Sarcastic Commentator"]
        
        # Calculate persona scores
        persona_scores = {}
        
        for persona in all_personas:
            # Start with context relevance score (1.0 to 9.0)
            context_score = self._calculate_context_fit(persona, context, audience)
            score = context_score
            
            # User preference adjustments (strong influence)
            if persona in user_pref.liked_personas:
                score += 4.0  # Strong boost for liked personas
            elif persona in user_pref.disliked_personas:
                score = 0.0  # Eliminate disliked personas completely
                
            # Historical performance (moderate influence)
            if persona in user_pref.persona_scores:
                historical_score = user_pref.persona_scores[persona]
                # Scale historical score to influence current recommendation
                performance_modifier = (historical_score - 5.0) * 0.8
                score += performance_modifier
            
            # Interaction frequency bonus (slight influence)
            interaction_count = len([i for i in user_pref.interaction_history if i.get('persona_name') == persona])
            if interaction_count > 0:
                # Slight bonus for familiar personas, but not too much
                familiarity_bonus = min(0.5, interaction_count * 0.1)
                score += familiarity_bonus
            
            # Context preferences (moderate influence)
            context_relevance = self._calculate_context_relevance(context, user_pref.context_preferences)
            score += context_relevance * 0.6
            
            # Audience fit (slight influence)
            audience_fit = self._calculate_audience_fit(persona, audience)
            score += audience_fit * 0.4
            
            # Ensure minimum score for non-disliked personas
            if persona not in user_pref.disliked_personas:
                score = max(score, 2.0)  # Minimum viable score
            
            persona_scores[persona] = score
        
        # Sort personas by score and return top recommendations
        sorted_personas = sorted(persona_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Filter out disliked personas completely
        filtered_personas = [
            persona for persona, score in sorted_personas
            if persona not in user_pref.disliked_personas and score > 0
        ]
        
        print(f"  Persona recommendations for {user_id}:")
        for persona in filtered_personas[:8]:  # Show more for debugging
            score = persona_scores[persona]
            status = ""
            if persona in user_pref.liked_personas:
                status = " (LIKED)"
            elif persona in user_pref.disliked_personas:
                status = " (DISLIKED - FILTERED)"
            print(f"    {persona}: {score:.1f}{status}")
        
        return filtered_personas[:5]
    
    def _calculate_context_fit(self, persona: str, context: str, audience: str) -> float:
        """Calculate how well a persona fits the context - returns 1.0 to 9.0"""
        base_score = 5.0
        context_lower = context.lower()
        
        # Persona-specific context bonuses
        context_bonuses = {
            'office_worker': ['work', 'office', 'job', 'meeting', 'boss', 'colleague', 'legal', 'professional'],
            'dad_humor_enthusiast': ['family', 'parent', 'kid', 'child', 'lunch', 'home', 'dad'],
            'gaming_guru': ['game', 'gaming', 'play', 'strategy', 'secret', 'gaming'],
            'millennial_memer': ['procrastinate', 'internet', 'social', 'online', 'meme'],
            'dark_humor_specialist': ['inappropriate', 'dark', 'twisted', 'edgy'],
            'suburban_parent': ['family', 'home', 'neighborhood', 'parent', 'suburban'],
            'gen_z_chaos': ['chaos', 'random', 'weird', 'unexpected', 'young'],
            'wordplay_master': ['word', 'pun', 'clever', 'witty', 'play'],
            'corporate_ladder_climber': ['work', 'career', 'professional', 'business', 'corporate']
        }
        
        # Check for context matches
        if persona in context_bonuses:
            matches = sum(1 for keyword in context_bonuses[persona] if keyword in context_lower)
            if matches > 0:
                base_score += matches * 1.5  # Bonus for context relevance
        
        # Audience fit
        audience_bonuses = {
            'friends': ['gaming_guru', 'gen_z_chaos', 'dark_humor_specialist'],
            'family': ['dad_humor_enthusiast', 'suburban_parent'],
            'colleagues': ['office_worker', 'corporate_ladder_climber'],
            'general': ['wordplay_master', 'millennial_memer']
        }
        
        if audience in audience_bonuses and persona in audience_bonuses[audience]:
            base_score += 1.0
        
        return min(9.0, max(1.0, base_score))
    
    def _calculate_context_relevance(self, context: str, context_preferences: Dict[str, float]) -> float:
        """Calculate context relevance based on user preferences"""
        if not context_preferences:
            return 0.0
        
        context_keywords = context.lower().split()
        total_score = 0.0
        keyword_count = 0
        
        for keyword in context_keywords:
            if len(keyword) > 3 and keyword in context_preferences:
                total_score += context_preferences[keyword]
                keyword_count += 1
        
        if keyword_count == 0:
            return 0.0
        
        return total_score / keyword_count
    
    def _calculate_audience_fit(self, persona: str, audience: str) -> float:
        """Calculate how well a persona fits the audience"""
        audience_fit_scores = {
            'friends': {
                'gaming_guru': 0.8,
                'gen_z_chaos': 0.9,
                'dark_humor_specialist': 0.7,
                'wordplay_master': 0.6
            },
            'family': {
                'dad_humor_enthusiast': 0.9,
                'suburban_parent': 0.8,
                'office_worker': 0.5
            },
            'colleagues': {
                'office_worker': 0.9,
                'corporate_ladder_climber': 0.8,
                'wordplay_master': 0.6
            },
            'general': {
                'wordplay_master': 0.7,
                'millennial_memer': 0.6,
                'gaming_guru': 0.5
            }
        }
        
        return audience_fit_scores.get(audience, {}).get(persona, 0.3)
    
    async def get_user_interaction_history(self, user_id: str) -> List[Dict[str, Any]]:
        """Get user interaction history from database for dynamic persona generation"""
        try:
            from agent_system.models.database import get_session_local, UserFeedback
            from agent_system.config.settings import settings
            
            SessionLocal = get_session_local(settings.database_url)
            db = SessionLocal()
            
            try:
                # Get user feedback history
                feedback_history = db.query(UserFeedback).filter(
                    UserFeedback.user_id == user_id
                ).order_by(UserFeedback.created_at.desc()).limit(50).all()
                
                # Convert to list of dictionaries
                interaction_history = []
                for feedback in feedback_history:
                    interaction = {
                        'persona_name': feedback.persona_name,
                        'context': feedback.context,
                        'feedback_score': feedback.feedback_score,
                        'response_text': feedback.response_text,
                        'topic': feedback.topic,
                        'audience': feedback.audience,
                        'created_at': feedback.created_at.isoformat() if feedback.created_at else None
                    }
                    interaction_history.append(interaction)
                
                print(f"  📊 Retrieved {len(interaction_history)} interactions for user {user_id}")
                return interaction_history
                
            finally:
                db.close()
                
        except Exception as e:
            print(f"  ❌ Error getting user interaction history: {e}")
            return []

# Global instance
improved_aws_knowledge_base = PostgreSQLKnowledgeBase() 