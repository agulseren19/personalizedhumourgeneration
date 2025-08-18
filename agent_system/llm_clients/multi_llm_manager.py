#!/usr/bin/env python3
"""
Multi-LLM Manager for Agent System
Supports GPT-4, Claude, DeepSeek with intelligent model selection
"""

import asyncio
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()  # This will load from .env file in current directory or parent directories

# LLM Client imports
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import httpx
    DEEPSEEK_AVAILABLE = True
except ImportError:
    DEEPSEEK_AVAILABLE = False

class LLMProvider(Enum):
    OPENAI_GPT4 = "gpt-4"
    OPENAI_GPT4_TURBO = "gpt-4-turbo"
    OPENAI_GPT35 = "gpt-3.5-turbo"
    CLAUDE_OPUS = "claude-3-opus"
    CLAUDE_SONNET = "claude-3-sonnet"
    CLAUDE_HAIKU = "claude-3-haiku"
    DEEPSEEK_CHAT = "deepseek-chat"
    DEEPSEEK_CODER = "deepseek-coder"

@dataclass
class LLMRequest:
    prompt: str
    model: LLMProvider
    temperature: float = 0.8
    max_tokens: int = 500
    system_prompt: Optional[str] = None
    persona_context: Optional[Dict[str, Any]] = None

@dataclass
class LLMResponse:
    content: str
    model: str
    tokens_used: int
    cost_estimate: float
    response_time: float
    provider: LLMProvider

class MultiLLMManager:
    """Manages multiple LLM providers for different agents"""
    
    def __init__(self):
        self.openai_client = None
        self.anthropic_client = None
        self.deepseek_base_url = "https://api.deepseek.com"
        
        # Initialize available clients
        self._initialize_clients()
        
        # Model assignment for different persona types
        self.persona_model_mapping = {
            # Creative personas get GPT-4 for best creativity
            "absurdist_artist": LLMProvider.OPENAI_GPT4,
            "gen_z_chaos": LLMProvider.OPENAI_GPT4,
            "wordplay_master": LLMProvider.OPENAI_GPT4,
            
            # Professional personas get Claude for thoughtfulness
            "office_worker": LLMProvider.CLAUDE_SONNET,
            "dark_humor_specialist": LLMProvider.CLAUDE_SONNET,
            
            # Pop culture personas get DeepSeek for efficiency
            "marvel_fanatic": LLMProvider.DEEPSEEK_CHAT,
            "gaming_guru": LLMProvider.DEEPSEEK_CHAT,
            "millennial_memer": LLMProvider.DEEPSEEK_CHAT,
            
            # Family-friendly gets GPT-3.5 for reliability
            "dad_humor_enthusiast": LLMProvider.OPENAI_GPT35,
            "suburban_parent": LLMProvider.OPENAI_GPT35,
            
            # Food and lifestyle get Haiku for speed
            "foodie_comedian": LLMProvider.CLAUDE_HAIKU,
            "college_survivor": LLMProvider.CLAUDE_HAIKU,
        }
        
        # Evaluation model preferences
        self.evaluator_models = {
            "primary": LLMProvider.CLAUDE_OPUS,  # Best reasoning
            "secondary": LLMProvider.OPENAI_GPT4,  # Good backup
            "fast": LLMProvider.CLAUDE_HAIKU  # Quick evaluation
        }
    
    def _initialize_clients(self):
        """Initialize all available LLM clients"""
        # OpenAI
        if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
            self.openai_client = openai.AsyncOpenAI(
                api_key=os.getenv("OPENAI_API_KEY")
            )
            print("OpenAI client initialized")
        
        # Anthropic
        if ANTHROPIC_AVAILABLE and os.getenv("ANTHROPIC_API_KEY"):
            self.anthropic_client = Anthropic(
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )
            print("Anthropic client initialized")
        
        # DeepSeek setup
        if DEEPSEEK_AVAILABLE and os.getenv("DEEPSEEK_API_KEY"):
            print("DeepSeek client available")
    
    def get_best_model_for_persona(self, persona_name: str) -> LLMProvider:
        """Get the best LLM model for a specific persona"""
        # Check if we have a specific mapping
        for persona_key, model in self.persona_model_mapping.items():
            if persona_key.lower() in persona_name.lower():
                return model
        
        # Default fallbacks based on available clients
        if self.openai_client:
            return LLMProvider.OPENAI_GPT4
        elif self.anthropic_client:
            return LLMProvider.CLAUDE_SONNET
        else:
            return LLMProvider.DEEPSEEK_CHAT
    
    def get_evaluator_model(self, evaluation_type: str = "primary") -> LLMProvider:
        """Get the best model for evaluation tasks"""
        return self.evaluator_models.get(evaluation_type, LLMProvider.CLAUDE_SONNET)
    
    async def generate_response(self, request: LLMRequest) -> LLMResponse:
        """Generate response using the specified LLM"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            if request.model.value.startswith("gpt"):
                response = await self._generate_openai(request)
            elif request.model.value.startswith("claude"):
                response = await self._generate_anthropic(request)
            elif request.model.value.startswith("deepseek"):
                response = await self._generate_deepseek(request)
            else:
                raise ValueError(f"Unsupported model: {request.model}")
            
            response.response_time = asyncio.get_event_loop().time() - start_time
            return response
            
        except Exception as e:
            print(f"ERROR: Error with {request.model.value}: {e}")
            # Fallback to available model
            return await self._fallback_generate(request, start_time)
    
    async def _generate_openai(self, request: LLMRequest) -> LLMResponse:
        """Generate using OpenAI models"""
        if not self.openai_client:
            raise Exception("OpenAI client not available")
        
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})
        
        response = await self.openai_client.chat.completions.create(
            model=request.model.value,
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        content = response.choices[0].message.content.strip()
        tokens_used = response.usage.total_tokens
        
        # Estimate cost (rough estimates)
        cost_per_token = {
            "gpt-4": 0.00003,
            "gpt-4-turbo": 0.00001,
            "gpt-3.5-turbo": 0.000002
        }
        cost = tokens_used * cost_per_token.get(request.model.value, 0.00001)
        
        return LLMResponse(
            content=content,
            model=request.model.value,
            tokens_used=tokens_used,
            cost_estimate=cost,
            response_time=0,  # Will be set by caller
            provider=request.model
        )
    
    async def _generate_anthropic(self, request: LLMRequest) -> LLMResponse:
        """Generate using Anthropic Claude models"""
        if not self.anthropic_client:
            raise Exception("Anthropic client not available")
        
        # Anthropic is sync, so run in executor
        def _sync_generate():
            messages = [{"role": "user", "content": request.prompt}]
            if request.system_prompt:
                # Claude handles system prompts differently
                system_context = request.system_prompt
            else:
                system_context = "You are a helpful assistant."
            
            response = self.anthropic_client.messages.create(
                model=request.model.value,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                system=system_context,
                messages=messages
            )
            return response
        
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, _sync_generate)
        
        content = response.content[0].text.strip()
        tokens_used = response.usage.input_tokens + response.usage.output_tokens
        
        # Claude cost estimates
        cost_per_token = {
            "claude-3-opus": 0.000015,
            "claude-3-sonnet": 0.000003,
            "claude-3-haiku": 0.00000025
        }
        cost = tokens_used * cost_per_token.get(request.model.value, 0.000003)
        
        return LLMResponse(
            content=content,
            model=request.model.value,
            tokens_used=tokens_used,
            cost_estimate=cost,
            response_time=0,
            provider=request.model
        )
    
    async def _generate_deepseek(self, request: LLMRequest) -> LLMResponse:
        """Generate using DeepSeek models"""
        if not DEEPSEEK_AVAILABLE:
            raise Exception("DeepSeek dependencies not available")
        
        headers = {
            "Authorization": f"Bearer {os.getenv('DEEPSEEK_API_KEY')}",
            "Content-Type": "application/json"
        }
        
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})
        
        data = {
            "model": request.model.value,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.deepseek_base_url}/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=30.0
            )
            response.raise_for_status()
            result = response.json()
        
        content = result["choices"][0]["message"]["content"].strip()
        tokens_used = result["usage"]["total_tokens"]
        
        # DeepSeek is very cost-effective
        cost = tokens_used * 0.0000002  # Very low cost
        
        return LLMResponse(
            content=content,
            model=request.model.value,
            tokens_used=tokens_used,
            cost_estimate=cost,
            response_time=0,
            provider=request.model
        )
    
    async def _fallback_generate(self, original_request: LLMRequest, start_time: float) -> LLMResponse:
        """Fallback to available models when primary fails"""
        # Try available models in order of preference
        fallback_models = [
            LLMProvider.OPENAI_GPT35,
            LLMProvider.CLAUDE_HAIKU,
            LLMProvider.DEEPSEEK_CHAT
        ]
        
        for model in fallback_models:
            try:
                fallback_request = LLMRequest(
                    prompt=original_request.prompt,
                    model=model,
                    temperature=original_request.temperature,
                    max_tokens=original_request.max_tokens,
                    system_prompt=original_request.system_prompt
                )
                
                response = await self.generate_response(fallback_request)
                response.response_time = asyncio.get_event_loop().time() - start_time
                return response
                
            except Exception as e:
                print(f"WARNING: Fallback {model.value} also failed: {e}")
                continue
        
        # If all fails, return mock response
        return LLMResponse(
            content="Sorry, I couldn't generate a response right now.",
            model="fallback",
            tokens_used=10,
            cost_estimate=0.0,
            response_time=asyncio.get_event_loop().time() - start_time,
            provider=original_request.model
        )
    
    async def batch_generate(self, requests: List[LLMRequest]) -> List[LLMResponse]:
        """Generate multiple responses in parallel"""
        tasks = [self.generate_response(request) for request in requests]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    def get_available_models(self) -> List[LLMProvider]:
        """Get list of currently available models"""
        available = []
        
        if self.openai_client:
            available.extend([
                LLMProvider.OPENAI_GPT4,
                LLMProvider.OPENAI_GPT4_TURBO,
                LLMProvider.OPENAI_GPT35
            ])
        
        if self.anthropic_client:
            available.extend([
                LLMProvider.CLAUDE_OPUS,
                LLMProvider.CLAUDE_SONNET,
                LLMProvider.CLAUDE_HAIKU
            ])
        
        if DEEPSEEK_AVAILABLE and os.getenv("DEEPSEEK_API_KEY"):
            available.extend([
                LLMProvider.DEEPSEEK_CHAT,
                LLMProvider.DEEPSEEK_CODER
            ])
        
        return available
    
    def get_cost_summary(self) -> Dict[str, float]:
        """Get cost estimates for different models"""
        return {
            "GPT-4": 0.00003,
            "GPT-4-Turbo": 0.00001,
            "GPT-3.5-Turbo": 0.000002,
            "Claude-Opus": 0.000015,
            "Claude-Sonnet": 0.000003,
            "Claude-Haiku": 0.00000025,
            "DeepSeek": 0.0000002
        }

# Global instance
multi_llm_manager = MultiLLMManager() 