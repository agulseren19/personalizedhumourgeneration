import asyncio
import aiohttp
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum

import openai
from anthropic import Anthropic
import boto3
from botocore.exceptions import ClientError

# Handle imports for different execution contexts
import sys
from pathlib import Path

# Add the agent_system directory to Python path
current_dir = Path(__file__).parent
agent_system_dir = current_dir.parent
sys.path.insert(0, str(agent_system_dir))

from config.settings import settings

class LLMProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    DEEPSEEK = "deepseek"
    AWS_BEDROCK = "aws_bedrock"

@dataclass
class LLMResponse:
    content: str
    model: str
    provider: LLMProvider
    tokens_used: int
    response_time: float
    cost_estimate: float = 0.0

@dataclass
class LLMRequest:
    prompt: str
    model: str
    temperature: float = 0.8
    max_tokens: int = 500
    system_prompt: Optional[str] = None

class RateLimiter:
    def __init__(self, max_requests_per_minute: int = 60):
        self.max_requests = max_requests_per_minute
        self.requests = []
    
    async def wait_if_needed(self):
        now = time.time()
        # Remove requests older than 1 minute
        self.requests = [req_time for req_time in self.requests if now - req_time < 60]
        
        if len(self.requests) >= self.max_requests:
            sleep_time = 60 - (now - self.requests[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        
        self.requests.append(now)

class LLMManager:
    def __init__(self):
        self.openai_client = None
        self.anthropic_client = None
        self.bedrock_client = None
        
        # Initialize clients
        self._init_clients()
        
        # Rate limiters for each provider
        self.rate_limiters = {
            LLMProvider.OPENAI: RateLimiter(60),
            LLMProvider.ANTHROPIC: RateLimiter(50),
            LLMProvider.DEEPSEEK: RateLimiter(100),
            LLMProvider.AWS_BEDROCK: RateLimiter(100)
        }
        
        # Model mappings
        self.model_provider_map = {
            "gpt-4": LLMProvider.OPENAI,
            "gpt-4-turbo": LLMProvider.OPENAI,
            "gpt-3.5-turbo": LLMProvider.OPENAI,
            "claude-3-sonnet": LLMProvider.ANTHROPIC,
            "claude-3-haiku": LLMProvider.ANTHROPIC,
            "claude-3-opus": LLMProvider.ANTHROPIC,
            "deepseek-chat": LLMProvider.DEEPSEEK,
            "deepseek-coder": LLMProvider.DEEPSEEK,
        }
        
        # Cost per token (rough estimates in USD)
        self.cost_per_token = {
            "gpt-4": {"input": 0.00003, "output": 0.00006},
            "gpt-4-turbo": {"input": 0.00001, "output": 0.00003},
            "gpt-3.5-turbo": {"input": 0.0000015, "output": 0.000002},
            "claude-3-sonnet": {"input": 0.000003, "output": 0.000015},
            "claude-3-haiku": {"input": 0.00000025, "output": 0.00000125},
            "claude-3-opus": {"input": 0.000015, "output": 0.000075},
            "deepseek-chat": {"input": 0.000001, "output": 0.000002},
            "deepseek-coder": {"input": 0.000001, "output": 0.000002},
        }
    
    def _init_clients(self):
        """Initialize LLM clients"""
        if settings.openai_api_key:
            self.openai_client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
        
        if settings.anthropic_api_key:
            self.anthropic_client = Anthropic(api_key=settings.anthropic_api_key)
        
        if settings.aws_access_key_id and settings.aws_secret_access_key:
            self.bedrock_client = boto3.client(
                'bedrock-runtime',
                region_name=settings.aws_bedrock_region,
                aws_access_key_id=settings.aws_access_key_id,
                aws_secret_access_key=settings.aws_secret_access_key
            )
    
    async def generate_response(self, request: LLMRequest, retry_count: int = 3) -> LLMResponse:
        """Generate response with fallback to other models if primary fails"""
        provider = self.model_provider_map.get(request.model)
        
        if not provider:
            raise ValueError(f"Unknown model: {request.model}")
        
        # Try primary model first
        try:
            return await self._generate_with_provider(request, provider)
        except Exception as e:
            print(f"⚠️ Primary model {request.model} failed: {e}")
            if retry_count > 0:
                # Try fallback models
                fallback_models = self._get_fallback_models(request.model)
                print(f"🔄 Trying fallback models: {fallback_models}")
                
                for fallback_model in fallback_models:
                    try:
                        print(f"🔄 Attempting fallback with {fallback_model}")
                        fallback_request = LLMRequest(
                            prompt=request.prompt,
                            model=fallback_model,
                            temperature=request.temperature,
                            max_tokens=request.max_tokens,
                            system_prompt=request.system_prompt
                        )
                        return await self._generate_with_provider(
                            fallback_request, 
                            self.model_provider_map[fallback_model]
                        )
                    except Exception as fallback_error:
                        print(f"❌ Fallback {fallback_model} failed: {fallback_error}")
                        continue
                
                # If all fallbacks fail, try ANY available model
                print("🔄 All fallbacks failed, trying ANY available model...")
                available_models = self._get_available_models()
                for available_model in available_models:
                    try:
                        print(f"🔄 Trying available model: {available_model}")
                        available_request = LLMRequest(
                            prompt=request.prompt,
                            model=available_model,
                            temperature=request.temperature,
                            max_tokens=request.max_tokens,
                            system_prompt=request.system_prompt
                        )
                        return await self._generate_with_provider(
                            available_request, 
                            self.model_provider_map[available_model]
                        )
                    except Exception as available_error:
                        print(f"❌ Available model {available_model} failed: {available_error}")
                        continue
                
                # If all available models fail, retry with original model
                print(f"🔄 All models failed, retrying with {request.model}...")
                return await self.generate_response(request, retry_count - 1)
            else:
                print(f"❌ All retry attempts failed for {request.model}")
                raise e
    
    def _get_fallback_models(self, primary_model: str) -> List[str]:
        """Get fallback models based on the primary model - OpenAI-only strategy"""
        # Since we only have OpenAI, prioritize OpenAI models
        openai_fallback_map = {
            "gpt-4": ["gpt-4-turbo", "gpt-3.5-turbo"],
            "gpt-4-turbo": ["gpt-4", "gpt-3.5-turbo"],
            "gpt-3.5-turbo": ["gpt-4-turbo", "gpt-4"],
            "claude-3-sonnet": ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
            "claude-3-haiku": ["gpt-3.5-turbo", "gpt-4-turbo", "gpt-4"],
            "claude-3-opus": ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
            "deepseek-chat": ["gpt-3.5-turbo", "gpt-4-turbo", "gpt-4"],
            "deepseek-coder": ["gpt-3.5-turbo", "gpt-4-turbo", "gpt-4"],
        }
        return openai_fallback_map.get(primary_model, [])
    
    def _get_available_models(self) -> List[str]:
        """Get all available models based on initialized clients - OpenAI-only strategy"""
        available_models = []
        
        if self.openai_client:
            available_models.extend(["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"])
            print(f"✅ OpenAI models available: {available_models}")
        else:
            print("❌ OpenAI client not available")
        
        # Since we only have OpenAI, don't check other clients
        # if self.anthropic_client:
        #     available_models.extend(["claude-3-sonnet", "claude-3-haiku", "claude-3-opus"])
        
        # if self.bedrock_client:
        #     available_models.extend(["deepseek-chat", "deepseek-coder"])
        
        # Remove duplicates and return
        return list(set(available_models))
    
    async def _generate_with_provider(self, request: LLMRequest, provider: LLMProvider) -> LLMResponse:
        """Generate response with specific provider"""
        start_time = time.time()
        
        # Rate limiting
        await self.rate_limiters[provider].wait_if_needed()
        
        if provider == LLMProvider.OPENAI:
            return await self._generate_openai(request, start_time)
        elif provider == LLMProvider.ANTHROPIC:
            return await self._generate_anthropic(request, start_time)
        elif provider == LLMProvider.DEEPSEEK:
            return await self._generate_deepseek(request, start_time)
        else:
            raise ValueError(f"Provider {provider} not implemented")
    
    async def _generate_openai(self, request: LLMRequest, start_time: float) -> LLMResponse:
        """Generate response using OpenAI"""
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized")
        
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})
        
        response = await self.openai_client.chat.completions.create(
            model=request.model,
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        content = response.choices[0].message.content
        tokens_used = response.usage.total_tokens
        response_time = time.time() - start_time
        
        cost = self._calculate_cost(request.model, tokens_used)
        
        return LLMResponse(
            content=content,
            model=request.model,
            provider=LLMProvider.OPENAI,
            tokens_used=tokens_used,
            response_time=response_time,
            cost_estimate=cost
        )
    
    async def _generate_anthropic(self, request: LLMRequest, start_time: float) -> LLMResponse:
        """Generate response using Anthropic"""
        if not self.anthropic_client:
            raise ValueError("Anthropic client not initialized")
        
        # Map Claude models
        model_map = {
            "claude-3-sonnet": "claude-3-sonnet-20240229",
            "claude-3-haiku": "claude-3-haiku-20240307",
            "claude-3-opus": "claude-3-opus-20240229"
        }
        
        mapped_model = model_map.get(request.model, request.model)
        
        response = await asyncio.to_thread(
            self.anthropic_client.messages.create,
            model=mapped_model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            system=request.system_prompt or "",
            messages=[{"role": "user", "content": request.prompt}]
        )
        
        content = response.content[0].text
        tokens_used = response.usage.input_tokens + response.usage.output_tokens
        response_time = time.time() - start_time
        
        cost = self._calculate_cost(request.model, tokens_used)
        
        return LLMResponse(
            content=content,
            model=request.model,
            provider=LLMProvider.ANTHROPIC,
            tokens_used=tokens_used,
            response_time=response_time,
            cost_estimate=cost
        )
    
    async def _generate_deepseek(self, request: LLMRequest, start_time: float) -> LLMResponse:
        """Generate response using DeepSeek API"""
        if not settings.deepseek_api_key:
            raise ValueError("DeepSeek API key not provided")
        
        # DeepSeek uses OpenAI-compatible API
        headers = {
            "Authorization": f"Bearer {settings.deepseek_api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": request.model,
            "messages": [
                {"role": "system", "content": request.system_prompt or "You are a helpful assistant."},
                {"role": "user", "content": request.prompt}
            ],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers=headers,
                json=data
            ) as response:
                if response.status != 200:
                    raise Exception(f"DeepSeek API error: {response.status}")
                
                result = await response.json()
                
                content = result["choices"][0]["message"]["content"]
                tokens_used = result["usage"]["total_tokens"]
                response_time = time.time() - start_time
                
                cost = self._calculate_cost(request.model, tokens_used)
                
                return LLMResponse(
                    content=content,
                    model=request.model,
                    provider=LLMProvider.DEEPSEEK,
                    tokens_used=tokens_used,
                    response_time=response_time,
                    cost_estimate=cost
                )
    
    def _calculate_cost(self, model: str, tokens_used: int) -> float:
        """Calculate estimated cost for the request"""
        if model not in self.cost_per_token:
            return 0.0
        
        # Simplified cost calculation (assumes equal input/output tokens)
        avg_cost = (self.cost_per_token[model]["input"] + self.cost_per_token[model]["output"]) / 2
        return tokens_used * avg_cost
    
    async def batch_generate(self, requests: List[LLMRequest]) -> List[LLMResponse]:
        """Generate multiple responses concurrently"""
        tasks = [self.generate_response(req) for req in requests]
        return await asyncio.gather(*tasks, return_exceptions=True)

# Global instance
llm_manager = LLMManager() 