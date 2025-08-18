#!/usr/bin/env python3
"""
AWS Knowledge Base Integration
Stores user preferences, humor patterns, and learning data
"""

import json
import boto3
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import hashlib
import os
import threading
import time
from decimal import Decimal

def convert_floats_to_decimal(obj):
    """Convert float values to Decimal for DynamoDB compatibility"""
    if isinstance(obj, dict):
        return {k: convert_floats_to_decimal(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_floats_to_decimal(v) for v in obj]
    elif isinstance(obj, float):
        return Decimal(str(obj))
    else:
        return obj

def convert_decimal_to_float(obj):
    """Convert Decimal values back to float when reading from DynamoDB"""
    if isinstance(obj, dict):
        return {k: convert_decimal_to_float(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_decimal_to_float(v) for v in obj]
    elif isinstance(obj, Decimal):
        return float(obj)
    else:
        return obj

@dataclass
class UserPreference:
    user_id: str
    humor_styles: List[str]
    liked_personas: List[str]
    disliked_personas: List[str]
    context_preferences: Dict[str, float]
    demographic_profile: Dict[str, Any]
    interaction_history: List[Dict[str, Any]]
    last_updated: datetime

@dataclass
class HumorPattern:
    pattern_id: str
    user_id: str
    context: str
    successful_humor: List[str]
    failed_humor: List[str]
    persona_performance: Dict[str, float]
    audience_type: str
    topic_preferences: Dict[str, float]

@dataclass
class GroupContext:
    group_id: str
    member_ids: List[str]
    group_preferences: Dict[str, Any]
    common_humor_styles: List[str]
    group_dynamics: Dict[str, Any]
    session_history: List[Dict[str, Any]]

class AWSKnowledgeBase:
    """AWS-powered knowledge base for user preferences and humor learning"""
    
    def __init__(self, mock_mode=False):
        self.mock_mode = mock_mode
        
        # Initialize mock storage
        self.mock_users = {}
        self.mock_patterns = {}
        self.mock_groups = {}
        
        # Initialize AWS attributes to None
        self.dynamodb = None
        self.bedrock = None
        self.opensearch = None
        self.users_table = None
        self.patterns_table = None
        self.groups_table = None
        
        # Track if we've already warned about bedrock access
        self.bedrock_access_warned = False
        
        # Check if we should use mock mode
        if not mock_mode:
            try:
                # Check for AWS credentials first
                aws_key = os.getenv("AWS_ACCESS_KEY_ID")
                aws_secret = os.getenv("AWS_SECRET_ACCESS_KEY")
                aws_region = os.getenv("AWS_REGION", "us-east-1")
                
                if not aws_key or not aws_secret:
                    print(f"WARNING: AWS credentials not found:")
                    print(f"  AWS_ACCESS_KEY_ID: {'Set' if aws_key else 'Missing'}")
                    print(f"  AWS_SECRET_ACCESS_KEY: {'Set' if aws_secret else 'Missing'}")
                    print(f"  AWS_REGION: {aws_region}")
                    print("Falling back to mock mode...")
                    self.mock_mode = True
                    return
                
                print(f"Attempting AWS initialization with region: {aws_region}")
                
                # Try to initialize AWS clients
                self.dynamodb = boto3.resource('dynamodb', region_name=aws_region)
                self.bedrock = boto3.client('bedrock-runtime', region_name=aws_region)
                self.opensearch = boto3.client('opensearch', region_name=aws_region)
                
                # Table names
                self.users_table_name = 'humor-user-preferences'
                self.patterns_table_name = 'humor-patterns'
                self.groups_table_name = 'humor-groups'
                
                # Test AWS connection by listing tables
                try:
                    list(self.dynamodb.tables.all())
                    print("AWS DynamoDB connection successful")
                except Exception as e:
                    print(f"WARNING: AWS DynamoDB connection failed: {e}")
                    print("Falling back to mock mode...")
                    self.mock_mode = True
                    return
                
                # Initialize tables
                self._initialize_tables()
                print("AWS Knowledge Base initialized successfully")
                
            except Exception as e:
                print(f"WARNING: AWS initialization failed: {e}")
                print(f"Error type: {type(e).__name__}")
                print("Falling back to mock mode...")
                self.mock_mode = True
        
        if self.mock_mode:
            print("Running in mock mode - simulating AWS functionality")
    
    def _initialize_tables(self):
        """Initialize DynamoDB tables if they don't exist"""
        print("Initializing DynamoDB tables...")
        
        try:
            # User preferences table
            self.users_table = self.dynamodb.Table(self.users_table_name)
            self.users_table.wait_until_exists()
        except:
            self._create_users_table()
        
        try:
            # Humor patterns table
            self.patterns_table = self.dynamodb.Table(self.patterns_table_name)
            self.patterns_table.wait_until_exists()
        except:
            self._create_patterns_table()
        
        try:
            # Groups table
            self.groups_table = self.dynamodb.Table(self.groups_table_name)
            self.groups_table.wait_until_exists()
        except:
            self._create_groups_table()
    
    def _create_users_table(self):
        """Create user preferences table"""
        self.users_table = self.dynamodb.create_table(
            TableName=self.users_table_name,
            KeySchema=[
                {'AttributeName': 'user_id', 'KeyType': 'HASH'}
            ],
            AttributeDefinitions=[
                {'AttributeName': 'user_id', 'AttributeType': 'S'}
            ],
            BillingMode='PAY_PER_REQUEST'
        )
        self.users_table.wait_until_exists()
        print(f"Created {self.users_table_name} table")
    
    def _create_patterns_table(self):
        """Create humor patterns table"""
        self.patterns_table = self.dynamodb.create_table(
            TableName=self.patterns_table_name,
            KeySchema=[
                {'AttributeName': 'pattern_id', 'KeyType': 'HASH'}
            ],
            AttributeDefinitions=[
                {'AttributeName': 'pattern_id', 'AttributeType': 'S'},
                {'AttributeName': 'user_id', 'AttributeType': 'S'}
            ],
            GlobalSecondaryIndexes=[
                {
                    'IndexName': 'user-id-index',
                    'KeySchema': [
                        {'AttributeName': 'user_id', 'KeyType': 'HASH'}
                    ],
                    'Projection': {'ProjectionType': 'ALL'}
                }
            ],
            BillingMode='PAY_PER_REQUEST'
        )
        self.patterns_table.wait_until_exists()
        print(f"Created {self.patterns_table_name} table")
    
    def _create_groups_table(self):
        """Create groups table"""
        self.groups_table = self.dynamodb.create_table(
            TableName=self.groups_table_name,
            KeySchema=[
                {'AttributeName': 'group_id', 'KeyType': 'HASH'}
            ],
            AttributeDefinitions=[
                {'AttributeName': 'group_id', 'AttributeType': 'S'}
            ],
            BillingMode='PAY_PER_REQUEST'
        )
        self.groups_table.wait_until_exists()
        print(f"Created {self.groups_table_name} table")
    
    async def store_user_preference(self, user_preference: UserPreference):
        """Store user preferences in AWS or mock storage"""
        try:
            if self.mock_mode:
                # Store in memory
                self.mock_users[user_preference.user_id] = user_preference
                print(f"(Mock) Stored preferences for user {user_preference.user_id}")
                return
            
            # Convert dataclass to dict for DynamoDB
            item = asdict(user_preference)
            
            # Convert datetime to string
            item['last_updated'] = user_preference.last_updated.isoformat()
            
            # Convert floats to Decimal for DynamoDB compatibility
            item = convert_floats_to_decimal(item)
            
            # Store in DynamoDB
            self.users_table.put_item(Item=item)
            
            # Also create vector embeddings for semantic search
            await self._create_preference_embedding(user_preference)
            
            print(f"Stored preferences for user {user_preference.user_id}")
            
        except Exception as e:
            print(f"ERROR: Error storing user preference: {e}")
    
    async def get_user_preference(self, user_id: str) -> Optional[UserPreference]:
        """Retrieve user preferences from AWS or mock storage"""
        try:
            if self.mock_mode:
                # Return from memory
                return self.mock_users.get(user_id)
            
            response = self.users_table.get_item(Key={'user_id': user_id})
            
            if 'Item' in response:
                item = response['Item']
                # Convert Decimals back to floats
                item = convert_decimal_to_float(item)
                # Convert string back to datetime
                item['last_updated'] = datetime.fromisoformat(item['last_updated'])
                return UserPreference(**item)
            
            return None
            
        except Exception as e:
            print(f"ERROR: Error retrieving user preference: {e}")
            return None
    
    async def update_user_feedback(self, user_id: str, persona_name: str, 
                                  feedback_score: float, context: str):
        """Update user preferences based on feedback"""
        try:
            # Get existing preferences
            user_pref = await self.get_user_preference(user_id)
            
            if not user_pref:
                # Create new user preference
                user_pref = UserPreference(
                    user_id=user_id,
                    humor_styles=[],
                    liked_personas=[],
                    disliked_personas=[],
                    context_preferences={},
                    demographic_profile={},
                    interaction_history=[],
                    last_updated=datetime.now()
                )
            
            # Update based on feedback
            feedback_entry = {
                'timestamp': datetime.now().isoformat(),
                'persona_name': persona_name,
                'feedback_score': feedback_score,
                'context': context
            }
            
            user_pref.interaction_history.append(feedback_entry)
            
            # Update liked/disliked personas
            if feedback_score >= 8:  # High score
                if persona_name not in user_pref.liked_personas:
                    user_pref.liked_personas.append(persona_name)
                # Remove from disliked if present
                if persona_name in user_pref.disliked_personas:
                    user_pref.disliked_personas.remove(persona_name)
            
            elif feedback_score <= 3:  # Low score
                if persona_name not in user_pref.disliked_personas:
                    user_pref.disliked_personas.append(persona_name)
                # Remove from liked if present
                if persona_name in user_pref.liked_personas:
                    user_pref.liked_personas.remove(persona_name)
            
            # Update context preferences
            if context not in user_pref.context_preferences:
                user_pref.context_preferences[context] = 0.5
            
            # Moving average for context preference
            current_score = user_pref.context_preferences[context]
            user_pref.context_preferences[context] = (
                current_score * 0.8 + (feedback_score / 10) * 0.2
            )
            
            # Keep only last 50 interactions
            user_pref.interaction_history = user_pref.interaction_history[-50:]
            user_pref.last_updated = datetime.now()
            
            # Store updated preferences
            await self.store_user_preference(user_pref)
            
            # Also store as humor pattern
            await self._store_humor_pattern(user_id, context, persona_name, feedback_score)
            
        except Exception as e:
            print(f"ERROR: Error updating user feedback: {e}")
    
    async def _store_humor_pattern(self, user_id: str, context: str, 
                                  persona_name: str, feedback_score: float):
        """Store humor patterns for analysis"""
        try:
            if self.mock_mode:
                # In mock mode, store in memory
                pattern_id = hashlib.md5(f"{user_id}:{context}".encode()).hexdigest()
                if pattern_id not in self.mock_patterns:
                    self.mock_patterns[pattern_id] = {
                        'pattern_id': pattern_id,
                        'user_id': user_id,
                        'context': context,
                        'successful_humor': [],
                        'failed_humor': [],
                        'persona_performance': {},
                        'audience_type': 'general',
                        'topic_preferences': {}
                    }
                
                pattern = self.mock_patterns[pattern_id]
                
                # Update pattern based on feedback
                if feedback_score >= 7:
                    if persona_name not in pattern['successful_humor']:
                        pattern['successful_humor'].append(persona_name)
                elif feedback_score <= 4:
                    if persona_name not in pattern['failed_humor']:
                        pattern['failed_humor'].append(persona_name)
                
                # Update persona performance
                if persona_name not in pattern['persona_performance']:
                    pattern['persona_performance'][persona_name] = []
                
                pattern['persona_performance'][persona_name].append(feedback_score)
                
                # Keep only last 20 scores per persona
                pattern['persona_performance'][persona_name] = (
                    pattern['persona_performance'][persona_name][-20:]
                )
                
                print(f"(Mock) Stored humor pattern for {user_id}: {persona_name} -> {feedback_score}")
                return
            
            # AWS mode - check if patterns_table exists
            if not self.patterns_table:
                print(f"ERROR: Error storing humor pattern: patterns_table not initialized")
                return
            
            # Create pattern ID based on user, context
            pattern_id = hashlib.md5(f"{user_id}:{context}".encode()).hexdigest()
            
            # Try to get existing pattern
            response = self.patterns_table.get_item(Key={'pattern_id': pattern_id})
            
            if 'Item' in response:
                pattern = response['Item']
            else:
                pattern = {
                    'pattern_id': pattern_id,
                    'user_id': user_id,
                    'context': context,
                    'successful_humor': [],
                    'failed_humor': [],
                    'persona_performance': {},
                    'audience_type': 'general',
                    'topic_preferences': {}
                }
            
            # Update pattern based on feedback
            if feedback_score >= 7:
                if persona_name not in pattern['successful_humor']:
                    pattern['successful_humor'].append(persona_name)
            elif feedback_score <= 4:
                if persona_name not in pattern['failed_humor']:
                    pattern['failed_humor'].append(persona_name)
            
            # Update persona performance
            if persona_name not in pattern['persona_performance']:
                pattern['persona_performance'][persona_name] = []
            
            pattern['persona_performance'][persona_name].append(feedback_score)
            
            # Keep only last 20 scores per persona
            pattern['persona_performance'][persona_name] = (
                pattern['persona_performance'][persona_name][-20:]
            )
            
            # Convert floats to Decimal for DynamoDB compatibility
            pattern = convert_floats_to_decimal(pattern)
            
            # Store updated pattern
            self.patterns_table.put_item(Item=pattern)
            
        except Exception as e:
            print(f"ERROR: Error storing humor pattern: {e}")
    
    async def get_persona_recommendations(self, user_id: str, context: str, 
                                        audience: str = "general") -> List[str]:
        """Get recommended personas based on user's history"""
        try:
            if self.mock_mode:
                # Simple mock recommendations based on context keywords
                if any(word in context.lower() for word in ['work', 'office', 'job']):
                    return ['office_worker', 'corporate_ladder_climber', 'millennial_memer']
                elif any(word in context.lower() for word in ['family', 'parent', 'kid']):
                    return ['dad_humor_enthusiast', 'suburban_parent', 'millennial_memer']
                elif any(word in context.lower() for word in ['game', 'gaming', 'play']):
                    return ['gaming_guru', 'millennial_memer', 'absurdist_artist']
                else:
                    return ['millennial_memer', 'dad_humor_enthusiast', 'office_worker']
            
            # AWS mode - check if patterns_table exists
            if not self.patterns_table:
                print(f"ERROR: Error getting persona recommendations: patterns_table not initialized")
                return ['millennial_memer', 'dad_humor_enthusiast', 'office_worker']
            
            # Get user preferences
            user_pref = await self.get_user_preference(user_id)
            
            if not user_pref:
                return ['millennial_memer', 'dad_humor_enthusiast', 'office_worker']
            
            # Get humor patterns for this context
            pattern_id = hashlib.md5(f"{user_id}:{context}".encode()).hexdigest()
            pattern_response = self.patterns_table.get_item(Key={'pattern_id': pattern_id})
            
            recommendations = []
            
            # Priority 1: Successful personas from similar contexts
            if 'Item' in pattern_response:
                pattern = convert_decimal_to_float(pattern_response['Item'])
                recommendations.extend(pattern.get('successful_humor', []))
            
            # Priority 2: Generally liked personas
            recommendations.extend(user_pref.liked_personas)
            
            # Priority 3: High-performing personas from persona performance
            if 'Item' in pattern_response:
                pattern = convert_decimal_to_float(pattern_response['Item'])
                persona_scores = {}
                
                for persona, scores in pattern.get('persona_performance', {}).items():
                    if scores:
                        avg_score = sum(scores) / len(scores)
                        if avg_score >= 6:  # Good performance threshold
                            persona_scores[persona] = avg_score
                
                # Sort by performance and add top performers
                top_performers = sorted(persona_scores.items(), 
                                      key=lambda x: x[1], reverse=True)
                recommendations.extend([persona for persona, score in top_performers[:3]])
            
            # Remove duplicates and disliked personas
            unique_recommendations = []
            for persona in recommendations:
                if (persona not in unique_recommendations and 
                    persona not in user_pref.disliked_personas):
                    unique_recommendations.append(persona)
            
            # If not enough recommendations, add defaults
            if len(unique_recommendations) < 3:
                defaults = ['millennial_memer', 'dad_humor_enthusiast', 'office_worker']
                for default in defaults:
                    if (default not in unique_recommendations and 
                        len(unique_recommendations) < 3):
                        unique_recommendations.append(default)
            
            return unique_recommendations[:3]
            
        except Exception as e:
            print(f"ERROR: Error getting persona recommendations: {e}")
            return ['millennial_memer', 'dad_humor_enthusiast', 'office_worker']
    
    async def create_group_context(self, group_id: str, member_ids: List[str]) -> GroupContext:
        """Create group context by analyzing member preferences"""
        try:
            # Get preferences for all members
            member_preferences = []
            for member_id in member_ids:
                pref = await self.get_user_preference(member_id)
                if pref:
                    member_preferences.append(pref)
            
            if not member_preferences:
                return GroupContext(
                    group_id=group_id,
                    member_ids=member_ids,
                    group_preferences={},
                    common_humor_styles=['millennial_memer', 'dad_humor_enthusiast'],
                    group_dynamics={},
                    session_history=[]
                )
            
            # Find common liked personas
            common_personas = set(member_preferences[0].liked_personas)
            for pref in member_preferences[1:]:
                common_personas &= set(pref.liked_personas)
            
            # Find common humor styles by analyzing liked personas
            humor_style_votes = {}
            for pref in member_preferences:
                for persona in pref.liked_personas:
                    # Map persona to humor style (simplified)
                    if 'dad' in persona.lower():
                        humor_style_votes['family_friendly'] = humor_style_votes.get('family_friendly', 0) + 1
                    elif 'millennial' in persona.lower():
                        humor_style_votes['pop_culture'] = humor_style_votes.get('pop_culture', 0) + 1
                    elif 'office' in persona.lower():
                        humor_style_votes['workplace'] = humor_style_votes.get('workplace', 0) + 1
            
            # Get top humor styles
            top_styles = sorted(humor_style_votes.items(), key=lambda x: x[1], reverse=True)
            common_humor_styles = [style for style, votes in top_styles[:3]]
            
            group_context = GroupContext(
                group_id=group_id,
                member_ids=member_ids,
                group_preferences={
                    'member_count': len(member_ids),
                    'common_personas': list(common_personas),
                    'style_preferences': humor_style_votes
                },
                common_humor_styles=common_humor_styles,
                group_dynamics={
                    'created_at': datetime.now().isoformat(),
                    'interaction_count': 0
                },
                session_history=[]
            )
            
            # Store group context
            await self.store_group_context(group_context)
            
            return group_context
            
        except Exception as e:
            print(f"ERROR: Error creating group context: {e}")
            return GroupContext(
                group_id=group_id,
                member_ids=member_ids,
                group_preferences={},
                common_humor_styles=['millennial_memer'],
                group_dynamics={},
                session_history=[]
            )
    
    async def store_group_context(self, group_context: GroupContext):
        """Store group context in AWS or mock storage"""
        try:
            if self.mock_mode:
                # Store in memory
                self.mock_groups[group_context.group_id] = group_context
                print(f"(Mock) Stored group context for {group_context.group_id}")
                return
            
            # AWS mode - check if groups_table exists
            if not self.groups_table:
                print(f"ERROR: Error storing group context: groups_table not initialized")
                return
            
            item = asdict(group_context)
            # Convert floats to Decimal for DynamoDB compatibility
            item = convert_floats_to_decimal(item)
            self.groups_table.put_item(Item=item)
            print(f"Stored group context for {group_context.group_id}")
        except Exception as e:
            print(f"ERROR: Error storing group context: {e}")
    
    async def get_group_context(self, group_id: str) -> Optional[GroupContext]:
        """Retrieve group context from AWS or mock storage"""
        try:
            if self.mock_mode:
                # Return from memory
                return self.mock_groups.get(group_id)
            
            # AWS mode - check if groups_table exists
            if not self.groups_table:
                print(f"ERROR: Error retrieving group context: groups_table not initialized")
                return None
            
            response = self.groups_table.get_item(Key={'group_id': group_id})
            
            if 'Item' in response:
                item = convert_decimal_to_float(response['Item'])
                return GroupContext(**item)
            
            return None
            
        except Exception as e:
            print(f"ERROR: Error retrieving group context: {e}")
            return None
    
    async def _create_preference_embedding(self, user_preference: UserPreference):
        """Create vector embeddings for semantic similarity search"""
        try:
            # Skip embedding creation in mock mode
            if self.mock_mode:
                return
                
            # Check if bedrock client is available
            if not self.bedrock:
                return
            
            # Create text representation of user preferences
            pref_text = f"""
            User preferences:
            Liked personas: {', '.join(user_preference.liked_personas)}
            Disliked personas: {', '.join(user_preference.disliked_personas)}
            Humor styles: {', '.join(user_preference.humor_styles)}
            Context preferences: {user_preference.context_preferences}
            """
            
            # Use AWS Bedrock to create embeddings
            response = self.bedrock.invoke_model(
                modelId='amazon.titan-embed-text-v1',
                body=json.dumps({
                    'inputText': pref_text
                })
            )
            
            embedding = json.loads(response['body'].read())['embedding']
            
            # Store embedding in OpenSearch for similarity search
            # Implementation would depend on OpenSearch setup
            
        except Exception as e:
            # Only warn once about bedrock access issues to reduce spam
            if "AccessDenied" in str(e) or "access denied" in str(e).lower():
                if not self.bedrock_access_warned:
                    self.bedrock_access_warned = True
                    print(f"WARNING: AWS Bedrock access denied - embeddings disabled (this warning shown once)")
            else:
                # Log other errors normally but make them less verbose
                if not hasattr(self, '_other_embedding_errors_shown'):
                    self._other_embedding_errors_shown = True
                    print(f"WARNING: Embedding creation failed: {type(e).__name__} (further embedding errors suppressed)")
            # Embeddings are optional, so we continue without them
    
    async def find_similar_users(self, user_id: str, limit: int = 5) -> List[str]:
        """Find users with similar humor preferences"""
        try:
            # This would use the embeddings to find similar users
            # Implementation depends on OpenSearch setup
            
            # For now, return placeholder
            return []
            
        except Exception as e:
            print(f"ERROR: Error finding similar users: {e}")
            return []

# Global instance - try AWS first, fall back to mock if needed
aws_knowledge_base = AWSKnowledgeBase(mock_mode=False) 