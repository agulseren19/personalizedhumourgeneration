"""
Production Configuration for CAH System
Use this for production deployment with PostgreSQL
"""

import os
from typing import List

class ProductionConfig:
    """Production configuration settings"""
    
    # Database Configuration
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/cah_db")
    
    # JWT Configuration
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "change-this-in-production")
    JWT_ALGORITHM = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES = 30
    
    # API Keys (set these in production environment)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
    
    # AWS Configuration
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
    
    # Redis Configuration
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # CORS Configuration
    ALLOWED_ORIGINS = [
        "https://cah-frontend.onrender.com",
        "https://yourdomain.com",  # Change this to your actual domain
        "http://localhost:3000"    # Keep for local testing
    ]
    
    # Security
    SECRET_KEY = os.getenv("SECRET_KEY", "change-this-in-production")
    ENVIRONMENT = "production"
    
    # Performance Settings
    MAX_CONNECTIONS = 20
    POOL_TIMEOUT = 30
    POOL_RECYCLE = 3600
    
    @classmethod
    def get_database_url(cls) -> str:
        """Get database URL with fallback"""
        if cls.DATABASE_URL:
            return cls.DATABASE_URL
        
        # Fallback to local PostgreSQL
        return "postgresql://postgres:postgres@localhost:5432/cah_db"
    
    @classmethod
    def validate_config(cls) -> List[str]:
        """Validate production configuration"""
        errors = []
        
        if not cls.JWT_SECRET_KEY or cls.JWT_SECRET_KEY == "change-this-in-production":
            errors.append("JWT_SECRET_KEY must be set to a secure value")
        
        if not cls.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY must be set for production")
        
        if not cls.ANTHROPIC_API_KEY:
            errors.append("ANTHROPIC_API_KEY must be set for production")
        
        if cls.DATABASE_URL.startswith("sqlite"):
            errors.append("DATABASE_URL must use PostgreSQL for production")
        
        return errors

# Production settings instance
production_config = ProductionConfig()
