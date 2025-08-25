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
        
        # Persona name mapping to handle inconsistencies between feedback and database
        self.persona_name_mapping = {
            # Template names (with underscores) -> Database names (with spaces)
            'gen_z_chaos': 'Gen Z Chaos Agent',
            'gaming_guru': 'Gaming Culture Comedian',
            'absurd_enthusiast': 'Absurdist Humor Artist',
            'office_worker': 'Corporate Humor Specialist',
            'dad_humor_enthusiast': 'Dad Humor Enthusiast',
            'suburban_parent': 'Suburban Parent Survivor',
            'foodie_comedian': 'Culinary Comedy Expert',
            'college_survivor': 'College Experience Veteran',
            'dark_humor_specialist': 'Dark Humor Connoisseur',
            'wordplay_master': 'Pun and Wordplay Expert',
            'millennial_memer': 'Millennial Memer',
            'marvel_fanatic': 'Marvel Universe Expert',
            
            # Common variations
            'gen z chaos': 'Gen Z Chaos Agent',
            'gaming guru': 'Gaming Culture Comedian',
            'absurd enthusiast': 'Absurdist Humor Artist',
            'office worker': 'Corporate Humor Specialist',
            'dad humor enthusiast': 'Dad Humor Enthusiast',
            'suburban parent': 'Suburban Parent Survivor',
            'foodie comedian': 'Culinary Comedy Expert',
            'college survivor': 'College Experience Veteran',
            'dark humor specialist': 'Dark Humor Connoisseur',
            'wordplay master': 'Pun and Wordplay Expert',
            'millennial memer': 'Millennial Memer',
            'marvel fanatic': 'Marvel Universe Expert',
        }
    
    def _normalize_persona_name(self, persona_name: str) -> str:
        """Normalize persona name to match database names"""
        if not persona_name:
            return persona_name
            
        # Try exact match first
        if persona_name in self.persona_name_mapping:
            return self.persona_name_mapping[persona_name]
        
        # Try case-insensitive match
        persona_name_lower = persona_name.lower()
        for template_name, db_name in self.persona_name_mapping.items():
            if template_name.lower() == persona_name_lower:
                return db_name
        
        # Try partial matches
        for template_name, db_name in self.persona_name_mapping.items():
            if template_name.lower() in persona_name_lower or persona_name_lower in template_name.lower():
                return db_name
        
        # If no match found, return original name
        print(f"  ⚠️  Could not normalize persona name: '{persona_name}'")
        return persona_name
    
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
                
                print(f"  Built preferences for {user_id}:")
                print(f"    Liked personas: {liked_personas}")
                print(f"    Disliked personas: {disliked_personas}")
                print(f"    Persona scores: {persona_scores}")
                
                # Get interaction history from feedback
                from agent_system.models.database import UserFeedback
                feedback_history = db.query(UserFeedback).filter(
                    UserFeedback.user_id == user_id
                ).order_by(UserFeedback.created_at.desc()).limit(50).all()
                
                interaction_history = []
                context_preferences = {}
                
                # Track normalized persona scores for building preferences
                normalized_persona_scores = {}
                
                for feedback in feedback_history:
                    # Normalize the persona name from feedback
                    normalized_persona_name = self._normalize_persona_name(feedback.persona_name)
                    
                    interaction = {
                        'persona_name': normalized_persona_name,  # Use normalized name
                        'context': feedback.context,
                        'feedback_score': feedback.feedback_score,
                        'response_text': feedback.response_text,
                        'topic': feedback.topic,
                        'audience': feedback.audience,
                        'timestamp': feedback.created_at.isoformat(),
                        'user_id': feedback.user_id
                    }
                    interaction_history.append(interaction)
                    
                    # Build normalized persona scores from feedback
                    if normalized_persona_name not in normalized_persona_scores:
                        normalized_persona_scores[normalized_persona_name] = []
                    normalized_persona_scores[normalized_persona_name].append(feedback.feedback_score)
                    
                    # Update context preferences
                    if feedback.context:
                        context_keywords = feedback.context.lower().split()
                        for keyword in context_keywords:
                            if len(keyword) > 3:
                                if keyword in context_preferences:
                                    context_preferences[keyword] = (context_preferences[keyword] + feedback.feedback_score) / 2
                                else:
                                    context_preferences[keyword] = feedback.feedback_score
                
                # Merge database preferences with feedback-based preferences
                final_liked_personas = []
                final_disliked_personas = []
                final_persona_scores = {}
                
                # Start with database preferences
                for persona_name, score in persona_scores.items():
                    final_persona_scores[persona_name] = score
                    if score >= self.like_threshold:
                        final_liked_personas.append(persona_name)
                    elif score <= self.dislike_threshold:
                        final_disliked_personas.append(persona_name)
                
                # Add feedback-based preferences for personas not in database
                for persona_name, scores in normalized_persona_scores.items():
                    if persona_name not in final_persona_scores:
                        avg_score = sum(scores) / len(scores)
                        final_persona_scores[persona_name] = avg_score
                        
                        if avg_score >= self.like_threshold:
                            if persona_name not in final_liked_personas:
                                final_liked_personas.append(persona_name)
                        elif avg_score <= self.dislike_threshold:
                            if persona_name not in final_disliked_personas:
                                final_disliked_personas.append(persona_name)
                
                print(f"  Merged preferences for {user_id}:")
                print(f"    Database preferences: {persona_scores}")
                print(f"    Feedback-based preferences: {normalized_persona_scores}")
                print(f"    Final merged preferences: {final_persona_scores}")
                print(f"    Final liked: {final_liked_personas}")
                print(f"    Final disliked: {final_disliked_personas}")
                
                user_pref = UserPreference(
                    user_id=user_id,
                    humor_styles=[],
                    liked_personas=final_liked_personas,
                    disliked_personas=final_disliked_personas,
                    context_preferences=context_preferences,
                    demographic_profile={},
                    interaction_history=interaction_history,
                    persona_scores=final_persona_scores,
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
                normalized_persona_name = self._normalize_persona_name(persona_name)
                print(f"  Normalized persona name: '{persona_name}' -> '{normalized_persona_name}'")
                
                persona = db.query(Persona).filter(Persona.name == normalized_persona_name).first()
                if persona:
                    print(f"  Found persona in database: {persona.name} (ID: {persona.id})")
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
                        print(f"  Updated existing preference: score {old_score:.1f} -> {new_score:.1f}")
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
                        print(f"  Created new preference: score {feedback_score:.1f}")
                else:
                    print(f"  ⚠️  Could not find persona '{normalized_persona_name}' in database")
                    print(f"  Available personas: {[p.name for p in db.query(Persona).all()]}")
                
                db.commit()
                print(f"  ✅ Database save successful for {user_id} -> {persona_name}")
                
                # ENHANCED: Update user embeddings for personalization (SHEEP-Medium approach)
                try:
                    from agent_system.knowledge.user_embedding_manager import UserEmbeddingManager
                    embedding_manager = UserEmbeddingManager()
                    
                    # Prepare feedback data for embedding update
                    feedback_data = [{
                        'feedback_score': feedback_score,
                        'persona_name': persona_name,
                        'context': context,
                        'topic': topic
                    }]
                    
                    # Update user embedding
                    embedding_manager.update_user_embedding(user_id, feedback_data, db)
                    print(f"  🧠 User embedding updated for personalization")
                    
                except Exception as embedding_error:
                    print(f"  ⚠️ User embedding update failed: {embedding_error}")
                    # Don't fail the whole operation if embeddings fail
                
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
        print(f"  📊 Getting persona recommendations for {user_id} (context: {context}, audience: {audience})")
        
        user_pref = await self.get_user_preference(user_id)
        
        if not user_pref:
            print(f"  ⚠️  No user preferences found for {user_id}, using fallback")
            # Fallback to context-based recommendations
            try:
                from agent_system.personas.enhanced_persona_templates import recommend_personas_for_context
                fallback_recommendations = recommend_personas_for_context(context, audience, "general")
                print(f"  📊 Fallback recommendations: {fallback_recommendations}")
                return fallback_recommendations
            except ImportError:
                try:
                    from personas.enhanced_persona_templates import recommend_personas_for_context
                    fallback_recommendations = recommend_personas_for_context(context, audience, "general")
                    print(f"  📊 Fallback recommendations: {fallback_recommendations}")
                    return fallback_recommendations
                except ImportError:
                    try:
                        from agent_system.personas.enhanced_persona_templates import recommend_personas_for_context
                        fallback_recommendations = recommend_personas_for_context(context, audience, "general")
                        print(f"  📊 Fallback recommendations: {fallback_recommendations}")
                        return fallback_recommendations
                    except ImportError:
                        print("⚠️  Could not import recommend_personas_for_context, using fallback")
                        fallback_recommendations = ["General Comedian", "Witty Observer", "Sarcastic Commentator"]
                        print(f"  📊 Hardcoded fallback recommendations: {fallback_recommendations}")
                        return fallback_recommendations
        
        print(f"  📊 User preferences found:")
        print(f"    Liked personas: {user_pref.liked_personas}")
        print(f"    Disliked personas: {user_pref.disliked_personas}")
        print(f"    Persona scores: {user_pref.persona_scores}")
        
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
        
        print(f"  📊 Available personas: {all_personas}")
        
        # Calculate persona scores
        persona_scores = {}
        
        for persona in all_personas:
            # Start with context relevance score (1.0 to 9.0)
            context_score = self._calculate_context_fit(persona, context, audience)
            score = context_score
            
            # User preference adjustments (strong influence)
            if persona in user_pref.liked_personas:
                score += 4.0  # Strong boost for liked personas
                print(f"    {persona}: +4.0 boost for being liked (total: {score:.1f})")
            elif persona in user_pref.disliked_personas:
                score = 0.0  # Eliminate disliked personas completely
                print(f"    {persona}: Set to 0.0 for being disliked")
                
            # Historical performance (moderate influence)
            if persona in user_pref.persona_scores:
                historical_score = user_pref.persona_scores[persona]
                # Scale historical score to influence current recommendation
                performance_modifier = (historical_score - 5.0) * 0.8
                score += performance_modifier
                print(f"    {persona}: +{performance_modifier:.1f} from historical performance (total: {score:.1f})")
            
            # Interaction frequency bonus (slight influence)
            interaction_count = len([i for i in user_pref.interaction_history if i.get('persona_name') == persona])
            if interaction_count > 0:
                # Slight bonus for familiar personas, but not too much
                familiarity_bonus = min(0.5, interaction_count * 0.1)
                score += familiarity_bonus
                print(f"    {persona}: +{familiarity_bonus:.1f} familiarity bonus (total: {score:.1f})")
            
            # Context preferences (moderate influence)
            context_relevance = self._calculate_context_relevance(context, user_pref.context_preferences)
            score += context_relevance * 0.6
            if context_relevance > 0:
                print(f"    {persona}: +{context_relevance * 0.6:.1f} context relevance (total: {score:.1f})")
            
            # Audience fit (slight influence)
            audience_fit = self._calculate_audience_fit(persona, audience)
            score += audience_fit * 0.4
            if audience_fit > 0:
                print(f"    {persona}: +{audience_fit * 0.4:.1f} audience fit (total: {score:.1f})")
            
            # Ensure minimum score for non-disliked personas
            if persona not in user_pref.disliked_personas:
                score = max(score, 2.0)  # Minimum viable score
                if score == 2.0:
                    print(f"    {persona}: Set to minimum score 2.0")
            
            persona_scores[persona] = score
        
        # Sort personas by score and return top recommendations
        sorted_personas = sorted(persona_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Filter out disliked personas completely
        filtered_personas = []
        for persona, score in sorted_personas:
            # Normalize the persona name to check against user preferences
            normalized_persona_name = self._normalize_persona_name(persona)
            
            # Check if the normalized name is in the disliked list
            if normalized_persona_name in user_pref.disliked_personas:
                print(f"    Filtering out disliked persona: '{persona}' -> '{normalized_persona_name}'")
                continue
            elif score > 0:
                filtered_personas.append(persona)
        
        print(f"  📊 Persona recommendations for {user_id}:")
        for persona in filtered_personas[:8]:  # Show more for debugging
            score = persona_scores[persona]
            normalized_name = self._normalize_persona_name(persona)
            status = ""
            if normalized_name in user_pref.liked_personas:
                status = " (LIKED)"
            elif normalized_name in user_pref.disliked_personas:
                status = " (DISLIKED - FILTERED)"
            print(f"    {persona} -> {normalized_name}: {score:.1f}{status}")
        
        # Convert template names to normalized database names for the final recommendations
        normalized_recommendations = []
        for persona in filtered_personas[:5]:
            # Try to find the corresponding database persona name
            normalized_name = None
            
            # First, try to find an exact match in the database
            try:
                from agent_system.models.database import get_session_local, Persona
                SessionLocal = get_session_local(settings.database_url)
                db = SessionLocal()
                try:
                    db_persona = db.query(Persona).filter(Persona.name == persona).first()
                    if db_persona:
                        normalized_name = db_persona.name
                    else:
                        # Try to find by partial match
                        for db_persona in db.query(Persona).all():
                            if (persona.lower() in db_persona.name.lower() or 
                                db_persona.name.lower() in persona.lower()):
                                normalized_name = db_persona.name
                                break
                finally:
                    db.close()
            except Exception as e:
                print(f"    ⚠️  Error looking up database persona for '{persona}': {e}")
            
            # If no database match found, use the normalized name from our mapping
            if not normalized_name:
                normalized_name = self._normalize_persona_name(persona)
            
            normalized_recommendations.append(normalized_name)
            print(f"    Normalized: '{persona}' -> '{normalized_name}'")
        
        final_recommendations = normalized_recommendations
        print(f"  📊 Final normalized recommendations: {final_recommendations}")
        return final_recommendations
    
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

    async def get_user_analytics(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive user analytics including liked/disliked personas"""
        try:
            from agent_system.models.database import get_session_local, PersonaPreference, UserFeedback, Persona
            from agent_system.config.settings import settings
            from datetime import datetime
            
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
                
                # Calculate analytics from feedback records (primary source)
                total_feedback = len(feedback_records)
                total_interactions = total_feedback  # Each feedback is an interaction
                
                # Calculate average score from feedback records
                if total_feedback > 0:
                    total_score = sum(fb.feedback_score for fb in feedback_records)
                    average_score = total_score / total_feedback
                else:
                    average_score = 0.0
                
                # Build persona performance data
                persona_performance = {}
                liked_personas = []
                disliked_personas = []
                
                for pref in preferences:
                    persona_name = pref.persona.name
                    
                    # Get all feedback for this persona
                    persona_feedback = [fb for fb in feedback_records if fb.persona_name == persona_name]
                    
                    if persona_feedback:
                        avg_score = sum(fb.feedback_score for fb in persona_feedback) / len(persona_feedback)
                        interaction_count = len(persona_feedback)
                        
                        # Determine status based on preference score
                        if pref.preference_score >= self.like_threshold:
                            status = 'liked'
                            if persona_name not in liked_personas:
                                liked_personas.append(persona_name)
                        elif pref.preference_score <= self.dislike_threshold:
                            status = 'disliked'
                            if persona_name not in disliked_personas:
                                disliked_personas.append(persona_name)
                        else:
                            status = 'neutral'
                        
                        persona_performance[persona_name] = {
                            'avg_score': round(avg_score, 1),
                            'interaction_count': interaction_count,
                            'status': status,
                            'preference_score': pref.preference_score
                        }
                
                # Get favorite persona
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
                
                return analytics
                
            finally:
                db.close()
                
        except Exception as e:
            print(f"ERROR in analytics: {e}")
            return {'error': str(e)}

# Global instance
improved_aws_knowledge_base = PostgreSQLKnowledgeBase() 