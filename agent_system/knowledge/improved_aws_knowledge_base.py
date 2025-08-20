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
            from agent_system.models.database import get_session_local, UserFeedback, User
            from agent_system.config.settings import settings
            
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
                persona_pref = db.query(PersonaPreference).filter(
                    PersonaPreference.user_id == user_id,
                    PersonaPreference.persona_id == (
                        db.query(Persona).filter(Persona.name == persona_name).first().id
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
                    new_persona = db.query(Persona).filter(Persona.name == persona_name).first()
                    if new_persona:
                        new_preference = PersonaPreference(
                            user_id=user_id,
                            persona_id=new_persona.id,
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
        
        # Update likes/dislikes
        if avg_score >= self.like_threshold:
            # Add to liked, remove from disliked
            if persona_name not in user_pref.liked_personas:
                user_pref.liked_personas.append(persona_name)
            if persona_name in user_pref.disliked_personas:
                user_pref.disliked_personas.remove(persona_name)
        
        elif avg_score <= self.dislike_threshold:
            # Add to disliked, remove from liked
            if persona_name not in user_pref.disliked_personas:
                user_pref.disliked_personas.append(persona_name)
            if persona_name in user_pref.liked_personas:
                user_pref.liked_personas.remove(persona_name)
        
        else:
            # Neutral - remove from both if present
            if persona_name in user_pref.liked_personas:
                user_pref.liked_personas.remove(persona_name)
            if persona_name in user_pref.disliked_personas:
                user_pref.disliked_personas.remove(persona_name)
    
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
        """Get persona recommendations based on user preferences and context - FIXED NUANCED SCORING"""
        user_pref = await self.get_user_preference(user_id)
        
        if not user_pref:
            # Fallback to context-based recommendations
            try:
                from ..personas.enhanced_persona_templates import recommend_personas_for_context
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
            from ..personas.enhanced_persona_templates import get_all_personas
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
        
        # FIXED: More nuanced scoring system
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
        
        print(f"  FIXED Persona recommendations for {user_id}:")
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
        if audience == 'colleagues' and persona in ['office_worker', 'corporate_ladder_climber']:
            base_score += 1.5
        elif audience == 'family' and persona in ['dad_humor_enthusiast', 'suburban_parent']:
            base_score += 1.5
        elif audience == 'friends' and persona in ['millennial_memer', 'gaming_guru', 'gen_z_chaos']:
            base_score += 1.5
        
        # Add some controlled randomness to avoid exact ties
        base_score += random.uniform(-0.3, 0.3)
        
        return max(1.0, min(9.0, base_score))
    
    def _calculate_audience_fit(self, persona: str, audience: str) -> float:
        """Calculate how well a persona fits the audience"""
        audience_fits = {
            'colleagues': ['office_worker', 'corporate_ladder_climber', 'wordplay_master'],
            'family': ['dad_humor_enthusiast', 'suburban_parent', 'wordplay_master'],
            'friends': ['millennial_memer', 'gaming_guru', 'gen_z_chaos'],
            'adults': ['dark_humor_specialist', 'wordplay_master', 'office_worker']
        }
        
        if audience in audience_fits and persona in audience_fits[audience]:
            return 1.0
        return 0.0
    
    def _calculate_context_relevance(self, context: str, context_preferences: Dict[str, float]) -> float:
        """Calculate how well this context matches user preferences"""
        if not context_preferences:
            return 0.0
        
        context_words = context.lower().split()
        relevance_scores = []
        
        for word in context_words:
            if word in context_preferences:
                relevance_scores.append(context_preferences[word])
        
        if not relevance_scores:
            return 0.0
        
        return sum(relevance_scores) / len(relevance_scores) - 5.0  # Normalize around 0
    
    async def create_group_context(self, group_id: str, member_ids: List[str]) -> Dict[str, Any]:
        """Create group context by combining member preferences"""
        print(f"  Creating group context for {group_id} with members: {member_ids}")
        
        # Get preferences for all members
        member_preferences = []
        for member_id in member_ids:
            pref = await self.get_user_preference(member_id)
            if pref:
                member_preferences.append(pref)
        
        if not member_preferences:
            return {
                'group_id': group_id,
                'member_ids': member_ids,
                'common_humor_styles': [],
                'group_preferences': {},
                'consensus_personas': []
            }
        
        # Find common liked personas
        liked_sets = [set(pref.liked_personas) for pref in member_preferences]
        common_liked = set.intersection(*liked_sets) if liked_sets else set()
        
        # Find commonly disliked personas
        disliked_sets = [set(pref.disliked_personas) for pref in member_preferences]
        common_disliked = set.intersection(*disliked_sets) if disliked_sets else set()
        
        # Calculate consensus persona scores
        all_personas = set()
        for pref in member_preferences:
            all_personas.update(pref.persona_scores.keys())
        
        consensus_personas = []
        for persona in all_personas:
            if persona in common_disliked:
                continue  # Skip commonly disliked personas
            
            scores = [pref.persona_scores.get(persona, 5.0) for pref in member_preferences]
            avg_score = sum(scores) / len(scores)
            
            if avg_score >= 6.0:  # Only include personas with decent group rating
                consensus_personas.append((persona, avg_score))
        
        # Sort by consensus score
        consensus_personas.sort(key=lambda x: x[1], reverse=True)
        
        group_context = {
            'group_id': group_id,
            'member_ids': member_ids,
            'common_humor_styles': list(common_liked),
            'group_preferences': {
                'liked_personas': list(common_liked),
                'disliked_personas': list(common_disliked),
                'consensus_personas': [persona for persona, score in consensus_personas[:5]]
            },
            'member_count': len(member_preferences)
        }
        
        print(f"  Group consensus personas: {group_context['group_preferences']['consensus_personas']}")
        
        return group_context
    
    async def get_user_analytics(self, user_id: str) -> Dict[str, Any]:
        """Get detailed analytics for a user"""
        user_pref = await self.get_user_preference(user_id)
        
        if not user_pref:
            return {'error': 'User not found'}
        
        # Calculate interaction statistics
        total_interactions = len(user_pref.interaction_history)
        
        if total_interactions == 0:
            return {'error': 'No interaction history'}
        
        # Calculate average scores
        all_scores = [interaction['feedback_score'] for interaction in user_pref.interaction_history]
        avg_score = sum(all_scores) / len(all_scores)
        
        # Persona performance
        persona_performance = {}
        for persona, score in user_pref.persona_scores.items():
            interactions = [
                i for i in user_pref.interaction_history
                if i['persona_name'] == persona
            ]
            persona_performance[persona] = {
                'avg_score': score,
                'interaction_count': len(interactions),
                'status': 'liked' if persona in user_pref.liked_personas else 
                         'disliked' if persona in user_pref.disliked_personas else 'neutral'
            }
        
        return {
            'user_id': user_id,
            'total_interactions': total_interactions,
            'average_score': avg_score,
            'liked_personas': user_pref.liked_personas,
            'disliked_personas': user_pref.disliked_personas,
            'persona_performance': persona_performance,
            'last_updated': user_pref.last_updated.isoformat()
        }

# Global instance
improved_aws_knowledge_base = PostgreSQLKnowledgeBase() 