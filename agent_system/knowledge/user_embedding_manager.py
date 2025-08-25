#!/usr/bin/env python3
"""
Minimal User Embedding Manager for SHEEP-Medium/HuBi-Medium Personalization
Integrates with existing CAH system without breaking changes
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import random

class UserEmbeddingManager:
    """Minimal user embedding manager for personalized humor generation"""
    
    def __init__(self, embedding_dimension: int = 128):
        self.embedding_dimension = embedding_dimension
        self.embeddings_cache = {}  # In-memory cache
        
    def initialize_user_embedding(self, user_id: str) -> Dict[str, Any]:
        """Initialize a new user embedding with random values (SHEEP-Medium approach)"""
        # Initialize with small random values (following Xavier initialization)
        embedding_vector = np.random.normal(0, 0.1, self.embedding_dimension).tolist()
        
        return {
            'user_id': user_id,
            'embedding_vector': embedding_vector,
            'embedding_dimension': self.embedding_dimension,
            'training_samples': 0,
            'last_trained': datetime.now(),
            'model_version': 'v1.0'
        }
    
    def get_user_embedding(self, user_id: str, db_session=None) -> Dict[str, Any]:
        """Get or create user embedding from database"""
        if user_id in self.embeddings_cache:
            return self.embeddings_cache[user_id]
        
        # Try to load from database
        if db_session:
            try:
                from ..models.database import UserEmbedding
                db_embedding = db_session.query(UserEmbedding).filter(
                    UserEmbedding.user_id == user_id
                ).first()
                
                if db_embedding:
                    embedding_data = {
                        'user_id': db_embedding.user_id,
                        'embedding_vector': db_embedding.embedding_vector,
                        'embedding_dimension': db_embedding.embedding_dimension,
                        'training_samples': db_embedding.training_samples,
                        'last_trained': db_embedding.last_trained,
                        'model_version': db_embedding.model_version
                    }
                    self.embeddings_cache[user_id] = embedding_data
                    return embedding_data
            except Exception as e:
                print(f"Warning: Could not load embedding for user {user_id}: {e}")
        
        # Create new embedding if not found
        embedding_data = self.initialize_user_embedding(user_id)
        self.embeddings_cache[user_id] = embedding_data
        return embedding_data
    
    def update_user_embedding(self, user_id: str, feedback_data: List[Dict[str, Any]], db_session=None):
        """Update user embedding based on feedback data using SHEEP-Medium approach"""
        if not feedback_data:
            return
        
        embedding_data = self.get_user_embedding(user_id, db_session)
        
        # Calculate embedding update based on feedback patterns
        embedding_update = self._calculate_embedding_update(embedding_data, feedback_data)
        
        # Apply update with learning rate (SHEEP-Medium gradient descent)
        learning_rate = 0.01
        new_embedding = []
        for i, val in enumerate(embedding_data['embedding_vector']):
            new_val = val + learning_rate * embedding_update[i]
            new_embedding.append(float(new_val))
        
        # Update embedding
        embedding_data['embedding_vector'] = new_embedding
        embedding_data['training_samples'] += len(feedback_data)
        embedding_data['last_trained'] = datetime.now()
        
        # Save to database if session available
        if db_session:
            self._save_embedding_to_db(embedding_data, db_session)
        
        # Update cache
        self.embeddings_cache[user_id] = embedding_data
        
        print(f"✅ Updated embedding for user {user_id} with {len(feedback_data)} feedback samples")
    
    def _calculate_embedding_update(self, embedding_data: Dict[str, Any], feedback_data: List[Dict[str, Any]]) -> List[float]:
        """Calculate embedding update based on feedback patterns (SHEEP-Medium formula)"""
        update_vector = [0.0] * self.embedding_dimension
        
        for feedback in feedback_data:
            # Extract features from feedback
            score = feedback.get('feedback_score', 5.0)
            persona_name = feedback.get('persona_name', '')
            context = feedback.get('context', '')
            topic = feedback.get('topic', '')
            
            # Normalize score to [-1, 1] range
            normalized_score = (score - 5.0) / 5.0
            
            # Create feature vector from feedback
            feature_vector = self._create_feature_vector(persona_name, context, topic)
            
            # Calculate update contribution
            for i in range(self.embedding_dimension):
                if i < len(feature_vector):
                    update_vector[i] += normalized_score * feature_vector[i]
        
        # Normalize update vector
        if any(update_vector):
            magnitude = np.linalg.norm(update_vector)
            if magnitude > 0:
                update_vector = [v / magnitude for v in update_vector]
        
        return update_vector
    
    def _create_feature_vector(self, persona_name: str, context: str, topic: str) -> List[float]:
        """Create feature vector from feedback components"""
        features = []
        
        # Persona features (hash-based)
        persona_hash = hash(persona_name) % 1000
        features.extend([float(persona_hash % 2**i) / 2**i for i in range(8)])
        
        # Context features (hash-based)
        context_hash = hash(context) % 1000
        features.extend([float(context_hash % 2**i) / 2**i for i in range(8)])
        
        # Topic features (hash-based)
        topic_hash = hash(topic) % 1000
        features.extend([float(topic_hash % 2**i) / 2**i for i in range(8)])
        
        # Pad to embedding dimension
        while len(features) < self.embedding_dimension:
            features.append(0.0)
        
        return features[:self.embedding_dimension]
    
    def get_personalized_prediction(self, user_id: str, text_embedding: List[float], 
                                  persona_name: str, context: str, topic: str, db_session=None) -> float:
        """Get personalized humor prediction using SHEEP-Medium formula"""
        user_embedding = self.get_user_embedding(user_id, db_session)
        
        # Create context feature vector
        context_features = self._create_feature_vector(persona_name, context, topic)
        
        # Apply SHEEP-Medium formula: y(t,u) = W_TU * (a(W_T * x_t) ⊙ a(W_U * x_u)) + word_biases
        # Simplified version for your system:
        
        # Element-wise multiplication of text and user embeddings
        interaction = []
        for i in range(min(len(text_embedding), len(user_embedding['embedding_vector']))):
            interaction.append(text_embedding[i] * user_embedding['embedding_vector'][i])
        
        # Add context features
        for i in range(min(len(context_features), len(interaction))):
            interaction[i] += context_features[i] * 0.1
        
        # Calculate prediction score (0-10 scale)
        prediction = np.mean(interaction) * 5 + 5  # Scale to 0-10
        prediction = max(0, min(10, prediction))  # Clamp to range
        
        return float(prediction)
    
    def get_similar_users(self, user_id: str, top_k: int = 5, db_session=None) -> List[Tuple[str, float]]:
        """Find users with similar humor preferences"""
        target_embedding = self.get_user_embedding(user_id, db_session)
        
        similarities = []
        for cached_user_id, cached_embedding in self.embeddings_cache.items():
            if cached_user_id == user_id:
                continue
            
            similarity = self._calculate_cosine_similarity(
                target_embedding['embedding_vector'],
                cached_embedding['embedding_vector']
            )
            similarities.append((cached_user_id, similarity))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def _calculate_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        if not vec1 or not vec2:
            return 0.0
        
        vec1_array = np.array(vec1)
        vec2_array = np.array(vec2)
        
        dot_product = np.dot(vec1_array, vec2_array)
        norm1 = np.linalg.norm(vec1_array)
        norm2 = np.linalg.norm(vec2_array)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    def _save_embedding_to_db(self, embedding_data: Dict[str, Any], db_session):
        """Save user embedding to database"""
        try:
            from ..models.database import UserEmbedding
            
            # Check if embedding exists
            existing = db_session.query(UserEmbedding).filter(
                UserEmbedding.user_id == embedding_data['user_id']
            ).first()
            
            if existing:
                # Update existing
                existing.embedding_vector = embedding_data['embedding_vector']
                existing.training_samples = embedding_data['training_samples']
                existing.last_trained = embedding_data['last_trained']
                existing.model_version = embedding_data['model_version']
            else:
                # Create new
                new_embedding = UserEmbedding(
                    user_id=embedding_data['user_id'],
                    embedding_vector=embedding_data['embedding_vector'],
                    embedding_dimension=embedding_data['embedding_dimension'],
                    training_samples=embedding_data['training_samples'],
                    last_trained=embedding_data['last_trained'],
                    model_version=embedding_data['model_version']
                )
                db_session.add(new_embedding)
            
            db_session.commit()
            
        except Exception as e:
            print(f"❌ Error saving embedding for user {embedding_data['user_id']}: {e}")
            db_session.rollback()
    
    def get_embedding_stats(self, user_id: str, db_session=None) -> Dict[str, Any]:
        """Get statistics about user embedding"""
        embedding_data = self.get_user_embedding(user_id, db_session)
        
        return {
            'user_id': user_id,
            'embedding_dimension': embedding_data['embedding_dimension'],
            'training_samples': embedding_data['training_samples'],
            'last_trained': embedding_data['last_trained'].isoformat() if embedding_data['last_trained'] else None,
            'model_version': embedding_data['model_version'],
            'embedding_magnitude': np.linalg.norm(embedding_data['embedding_vector']),
            'embedding_mean': np.mean(embedding_data['embedding_vector']),
            'embedding_std': np.std(embedding_data['embedding_vector'])
        }
