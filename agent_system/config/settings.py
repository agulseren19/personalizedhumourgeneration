import os
from dotenv import load_dotenv
from typing import List, Optional

# Load .env file from the agent_system directory
import pathlib
env_path = pathlib.Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

class Settings:
    def __init__(self):
        # API Keys
        self.openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
        self.anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
        self.deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
        
        # Environment settings
        self.environment: str = os.getenv("ENVIRONMENT", "development")
        self.default_content_filter: str = os.getenv("DEFAULT_CONTENT_FILTER", "moderate")
        
        # AWS Configuration
        self.aws_access_key_id: str = os.getenv("AWS_ACCESS_KEY_ID", "")
        self.aws_secret_access_key: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
        self.aws_region: str = os.getenv("AWS_REGION", "us-east-1")
        self.aws_bedrock_region: str = os.getenv("AWS_BEDROCK_REGION", "us-east-1")
        
        # Database
        self.database_url: str = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/cah_db")
        
        # Redis
        self.redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379")
        
        # JWT Configuration
        self.jwt_secret_key: str = os.getenv("JWT_SECRET_KEY", "dd9b35c0a7f5e5a1099ffd29a02b0b4b04a650f7a92bcc9a310cf2ccff1e647bf92320c3b313b6003d65c2e17a63b53a05507350cf0161f958c7f8b88b4c0d57")
        self.jwt_algorithm: str = os.getenv("JWT_ALGORITHM", "HS256")
        self.jwt_access_token_expire_minutes: int = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
        
        # Security
        self.secret_key: str = os.getenv("SECRET_KEY", "your-app-secret-key-change-this-in-production")
        
        # CORS Configuration
        self.allowed_origins: str = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:3001,https://cah-backend.onrender.com,https://personalizedhumourgenerationcah.vercel.app")
        
        # Google OAuth Configuration
        self.google_client_id: str = os.getenv("GOOGLE_CLIENT_ID", "")
        self.google_client_secret: str = os.getenv("GOOGLE_CLIENT_SECRET", "")
        self.google_redirect_uri: str = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8000/auth/google/callback")
        
        # Agent Configuration
        self.max_generation_agents: int = int(os.getenv("MAX_GENERATION_AGENTS", "5"))
        self.max_evaluation_agents: int = int(os.getenv("MAX_EVALUATION_AGENTS", "3"))
        self.top_k_results: int = int(os.getenv("TOP_K_RESULTS", "3"))
        
        # Model Configuration
        self.default_generation_model: str = os.getenv("DEFAULT_GENERATION_MODEL", "gpt-4")
        self.default_evaluation_model: str = os.getenv("DEFAULT_EVALUATION_MODEL", "claude-3-opus")
        self.temperature: float = float(os.getenv("TEMPERATURE", "0.9"))  # Higher for creativity
        self.max_tokens: int = int(os.getenv("MAX_TOKENS", "50"))  # Shorter for CAH responses
        
        # Optimized settings
        self.default_num_generators: int = int(os.getenv("DEFAULT_NUM_GENERATORS", "3"))
        self.default_num_evaluators: int = int(os.getenv("DEFAULT_NUM_EVALUATORS", "1"))
        self.evaluation_temperature: float = float(os.getenv("EVALUATION_TEMPERATURE", "0.1"))
        self.evaluation_max_tokens: int = int(os.getenv("EVALUATION_MAX_TOKENS", "10"))
        self.enable_parallel_generation: bool = os.getenv("ENABLE_PARALLEL_GENERATION", "true").lower() == "true"
        self.enable_single_result_optimization: bool = os.getenv("ENABLE_SINGLE_RESULT_OPTIMIZATION", "true").lower() == "true"
        self.use_simplified_prompts: bool = os.getenv("USE_SIMPLIFIED_PROMPTS", "true").lower() == "true"
        self.cah_specific_prompting: bool = os.getenv("CAH_SPECIFIC_PROMPTING", "true").lower() == "true"
        
        # Available Models
        self.available_generation_models: List[str] = [
            "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo",
            "claude-3-sonnet", "claude-3-haiku", "claude-3-opus",
            "deepseek-chat", "deepseek-coder"
        ]
        
        self.available_evaluation_models: List[str] = [
            "gpt-4", "claude-3-sonnet", "claude-3-opus"
        ]

settings = Settings() 