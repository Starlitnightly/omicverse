"""
Model Configuration and Management for OmicVerse Smart Agent
Based on pantheon-cli model manager design
"""

from typing import Dict, Optional, Tuple
import os

# Available models configuration - based on pantheon-cli
AVAILABLE_MODELS = {
    # OpenAI Models - GPT-5 Series (Latest)
    "gpt-5": "OpenAI GPT-5 (Latest)",
    "gpt-5-mini": "OpenAI GPT-5 Mini",
    "gpt-5-nano": "OpenAI GPT-5 Nano",
    "gpt-5-chat-latest": "OpenAI GPT-5 Chat Latest",
    
    # OpenAI Models - GPT-4 Series
    "gpt-4.1": "OpenAI GPT-4.1", 
    "gpt-4.1-mini": "OpenAI GPT-4.1 Mini",
    "gpt-4.1-nano": "OpenAI GPT-4.1 Nano",
    "gpt-4o": "OpenAI GPT-4o",
    "gpt-4o-2024-05-13": "OpenAI GPT-4o (2024-05-13)",
    "gpt-4o-mini": "OpenAI GPT-4o Mini",
    
    # OpenAI Models - o-Series (Reasoning)
    "o1": "OpenAI o1 (Reasoning)",
    "o1-pro": "OpenAI o1 Pro (Reasoning)",
    "o3-pro": "OpenAI o3 Pro (Reasoning)",
    "o3": "OpenAI o3 (Reasoning)",
    "o3-mini": "OpenAI o3 Mini (Reasoning)",
    "o1-mini": "OpenAI o1 Mini (Reasoning)",
    
    # Anthropic Models - Claude 4 Series (Latest)
    "anthropic/claude-opus-4-1-20250805": "Claude Opus 4.1 (Latest)",
    "anthropic/claude-opus-4-20250514": "Claude Opus 4",
    "anthropic/claude-sonnet-4-20250514": "Claude Sonnet 4",
    "anthropic/claude-3-7-sonnet-20250219": "Claude Sonnet 3.7",
    "anthropic/claude-3-5-haiku-20241022": "Claude Haiku 3.5",
    
    # Anthropic Models - Claude 3 Series (Legacy)
    "anthropic/claude-3-opus-20240229": "Claude 3 Opus (Legacy)",
    "anthropic/claude-3-sonnet-20240229": "Claude 3 Sonnet (Legacy)", 
    "anthropic/claude-3-haiku-20240307": "Claude 3 Haiku (Legacy)",
    
    # Google Models
    "gemini/gemini-2.5-pro": "Gemini 2.5 Pro",
    "gemini/gemini-2.5-flash": "Gemini 2.5 Flash",
    "gemini/gemini-2.0-pro": "Gemini 2.0 Pro",
    "gemini/gemini-2.0-flash": "Gemini 2.0 Flash",
    "gemini/gemini-pro": "Gemini Pro",
    
    # DeepSeek Models
    "deepseek/deepseek-chat": "DeepSeek Chat",
    "deepseek/deepseek-reasoner": "DeepSeek Reasoner",
    
    # Qwen/Alibaba Models - Latest 2025 Series
    "qwq-plus": "QwQ Plus (Reasoning)",
    "qwen-max": "Qwen Max (Latest)",
    "qwen-max-latest": "Qwen Max Latest",
    "qwen-plus": "Qwen Plus (Latest)",
    "qwen-turbo": "Qwen Turbo (Latest)",
    
    # Kimi/Moonshot Models - Latest K2 Series
    "moonshot/kimi-k2-0711-preview": "Kimi K2 (Preview)",
    "moonshot/kimi-k2-turbo-preview": "Kimi K2 Turbo (Preview)",
    "moonshot/kimi-latest": "Kimi Latest (Auto Context)",
    "moonshot/moonshot-v1-8k": "Moonshot V1 8K",
    "moonshot/moonshot-v1-32k": "Moonshot V1 32K",
    "moonshot/moonshot-v1-128k": "Moonshot V1 128K",
    
    # Grok/xAI Models
    "grok/grok-beta": "Grok Beta",
    "grok/grok-2": "Grok 2",
    
    # Zhipu AI/GLM Models
    "zhipu/glm-4.5": "GLM-4.5 (Zhipu AI - Latest)",
    "zhipu/glm-4.5-air": "GLM-4.5 Air (Zhipu AI - Latest)",
    "zhipu/glm-4.5-flash": "GLM-4.5 Flash (Zhipu AI - Latest)",
    "zhipu/glm-4": "GLM-4 (Zhipu AI)",
    "zhipu/glm-4-plus": "GLM-4 Plus (Zhipu AI)",
    "zhipu/glm-4-air": "GLM-4 Air (Zhipu AI)",
    "zhipu/glm-4-flash": "GLM-4 Flash (Zhipu AI - Free)",
}

# API Key requirements by provider
PROVIDER_API_KEYS = {
    # OpenAI models
    "gpt-5": "OPENAI_API_KEY",
    "gpt-5-mini": "OPENAI_API_KEY", 
    "gpt-5-nano": "OPENAI_API_KEY",
    "gpt-4.1": "OPENAI_API_KEY",
    "gpt-4.1-mini": "OPENAI_API_KEY",
    "gpt-4o": "OPENAI_API_KEY",
    "gpt-4o-mini": "OPENAI_API_KEY",
    "o1": "OPENAI_API_KEY",
    "o1-pro": "OPENAI_API_KEY",
    "o3": "OPENAI_API_KEY",
    "o3-mini": "OPENAI_API_KEY",
    
    # Anthropic models
    "anthropic/claude-opus-4-1-20250805": "ANTHROPIC_API_KEY",
    "anthropic/claude-opus-4-20250514": "ANTHROPIC_API_KEY",
    "anthropic/claude-sonnet-4-20250514": "ANTHROPIC_API_KEY",
    "anthropic/claude-3-7-sonnet-20250219": "ANTHROPIC_API_KEY",
    "anthropic/claude-3-5-haiku-20241022": "ANTHROPIC_API_KEY",
    "anthropic/claude-3-opus-20240229": "ANTHROPIC_API_KEY",
    "anthropic/claude-3-sonnet-20240229": "ANTHROPIC_API_KEY",
    "anthropic/claude-3-haiku-20240307": "ANTHROPIC_API_KEY",
    
    # Google models
    "gemini/gemini-2.5-pro": "GOOGLE_API_KEY",
    "gemini/gemini-2.5-flash": "GOOGLE_API_KEY",
    "gemini/gemini-2.0-pro": "GOOGLE_API_KEY",
    "gemini/gemini-pro": "GOOGLE_API_KEY",
    
    # DeepSeek models
    "deepseek/deepseek-chat": "DEEPSEEK_API_KEY",
    "deepseek/deepseek-reasoner": "DEEPSEEK_API_KEY",
    
    # Qwen models
    "qwq-plus": "DASHSCOPE_API_KEY",
    "qwen-max": "DASHSCOPE_API_KEY",
    "qwen-plus": "DASHSCOPE_API_KEY",
    "qwen-turbo": "DASHSCOPE_API_KEY",
    
    # Moonshot/Kimi models
    "moonshot/kimi-k2-0711-preview": "MOONSHOT_API_KEY",
    "moonshot/kimi-k2-turbo-preview": "MOONSHOT_API_KEY", 
    "moonshot/kimi-latest": "MOONSHOT_API_KEY",
    "moonshot/moonshot-v1-8k": "MOONSHOT_API_KEY",
    "moonshot/moonshot-v1-32k": "MOONSHOT_API_KEY",
    "moonshot/moonshot-v1-128k": "MOONSHOT_API_KEY",
    
    # Grok models
    "grok/grok-beta": "XAI_API_KEY",
    "grok/grok-2": "XAI_API_KEY",
    
    # Zhipu models (using ZAI_API_KEY as expected by pantheon)
    "zhipu/glm-4.5": "ZAI_API_KEY",
    "zhipu/glm-4.5-air": "ZAI_API_KEY",
    "zhipu/glm-4.5-flash": "ZAI_API_KEY",
    "zhipu/glm-4": "ZAI_API_KEY",
    "zhipu/glm-4-plus": "ZAI_API_KEY",
    "zhipu/glm-4-air": "ZAI_API_KEY",
    "zhipu/glm-4-flash": "ZAI_API_KEY",
}

# Custom endpoints for different providers
PROVIDER_ENDPOINTS = {
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com",
    "google": "https://generativelanguage.googleapis.com/v1beta",
    "deepseek": "https://api.deepseek.com/v1",
    "dashscope": "https://dashscope.aliyuncs.com/api/v1",
    "moonshot": "https://api.moonshot.cn/v1",
    "xai": "https://api.x.ai/v1",
    "zhipu": "https://open.bigmodel.cn/api/paas/v4",
}

# Provider-level default API keys (fallback when a specific model isn't mapped)
PROVIDER_DEFAULT_KEYS = {
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "dashscope": "DASHSCOPE_API_KEY",
    "moonshot": "MOONSHOT_API_KEY",
    "xai": "XAI_API_KEY",
    "zhipu": "ZAI_API_KEY",
}

# Model ID aliases for backward compatibility
MODEL_ALIASES: Dict[str, str] = {
    # Claude 4.5 variations
    "claude-sonnet-4-5": "anthropic/claude-sonnet-4-20250514",
    "claude-4-5-sonnet": "anthropic/claude-sonnet-4-20250514",
    "claude-sonnet-4-5-20250929": "anthropic/claude-sonnet-4-20250514",
    # Claude 3.7
    "claude-sonnet-3-7": "anthropic/claude-3-7-sonnet-20250219",
    "claude-3-7-sonnet": "anthropic/claude-3-7-sonnet-20250219",
    # Claude 3.5
    "claude-3-5-haiku": "anthropic/claude-3-5-haiku-20241022",
    "claude-haiku-3-5": "anthropic/claude-3-5-haiku-20241022",
    # Claude 3 legacy
    "claude-3-opus": "anthropic/claude-3-opus-20240229",
    "claude-opus-3": "anthropic/claude-3-opus-20240229",
    "claude-3-sonnet": "anthropic/claude-3-sonnet-20240229",
    "claude-sonnet-3": "anthropic/claude-3-sonnet-20240229",
    "claude-3-haiku": "anthropic/claude-3-haiku-20240307",
    "claude-haiku-3": "anthropic/claude-3-haiku-20240307",
    # Gemini
    "gemini-2.5-pro": "gemini/gemini-2.5-pro",
    "gemini-2-5-pro": "gemini/gemini-2.5-pro",
    "gemini-2.5-flash": "gemini/gemini-2.5-flash",
    "gemini-2-5-flash": "gemini/gemini-2.5-flash",
    "gemini-2.0-pro": "gemini/gemini-2.0-pro",
    "gemini-2-0-pro": "gemini/gemini-2.0-pro",
    "gemini-2.0-flash": "gemini/gemini-2.0-flash",
    "gemini-2-0-flash": "gemini/gemini-2.0-flash",
    "gemini-pro": "gemini/gemini-pro",
    # Deepseek
    "deepseek-chat": "deepseek/deepseek-chat",
    "deepseek-reasoner": "deepseek/deepseek-reasoner",
}

class ModelConfig:
    """Model configuration and validation for OmicVerse Smart Agent"""
    
    @staticmethod
    def normalize_model_id(model: str) -> str:
        """Normalize model ID to canonical form using alias mapping."""
        if model in AVAILABLE_MODELS:
            return model
        alias = MODEL_ALIASES.get(model.lower())
        return alias or model

    @staticmethod
    def is_model_supported(model: str) -> bool:
        """Check if a model is supported"""
        normalized = ModelConfig.normalize_model_id(model)
        return normalized in AVAILABLE_MODELS
    
    @staticmethod
    def get_model_description(model: str) -> str:
        """Get human-readable description for a model"""
        normalized = ModelConfig.normalize_model_id(model)
        return AVAILABLE_MODELS.get(normalized, f"Unknown model: {model}")
    
    @staticmethod
    def get_provider_from_model(model: str) -> str:
        """Determine provider from model name"""
        # Normalize first to handle aliases/unprefixed
        model = ModelConfig.normalize_model_id(model)
        if model.startswith("anthropic/"):
            return "anthropic"
        elif model.startswith(("qwq-", "qwen-", "qvq-")) or model.startswith("qwen/"):
            return "dashscope"
        elif model.startswith(("kimi-", "moonshot-")) or model.startswith("moonshot/"):
            return "moonshot"
        elif model.startswith("grok/"):
            return "xai"
        elif model.startswith("gemini/"):
            return "google"
        elif model.startswith("deepseek/"):
            return "deepseek"
        elif model.startswith("zhipu/"):
            return "zhipu"
        else:
            return "openai"
    
    @staticmethod
    def check_api_key_availability(model: str) -> Tuple[bool, str]:
        """Check if required API key is available for the model"""
        normalized = ModelConfig.normalize_model_id(model)
        required_key = PROVIDER_API_KEYS.get(normalized)
        if not required_key:
            provider = ModelConfig.get_provider_from_model(normalized)
            required_key = PROVIDER_DEFAULT_KEYS.get(provider)
        if not required_key:
            return True, "No API key required"
        if os.getenv(required_key):
            provider = ModelConfig.get_provider_from_model(normalized)
            return True, f"{provider.title()} API key available"
        else:
            provider = ModelConfig.get_provider_from_model(normalized)
            return False, f"{provider.title()} API key required: set {required_key}"
    
    @staticmethod
    def get_endpoint_for_model(model: str) -> str:
        """Get the API endpoint for a model"""
        provider = ModelConfig.get_provider_from_model(model)
        return PROVIDER_ENDPOINTS.get(provider, PROVIDER_ENDPOINTS["openai"])
    
    @staticmethod
    def list_supported_models(show_all: bool = False) -> str:
        """List all supported models grouped by provider"""
        result = "🤖 Supported Models:\n\n"
        
        # Group models by provider
        providers = {}
        for model_id, description in AVAILABLE_MODELS.items():
            provider = ModelConfig.get_provider_from_model(model_id)
            provider_name = provider.replace("dashscope", "Qwen").replace("xai", "Grok").replace("zhipu", "Zhipu AI").title()
            
            if provider_name not in providers:
                providers[provider_name] = []
            providers[provider_name].append((model_id, description))
        
        for provider_name, models in providers.items():
            result += f"**{provider_name}**:\n"
            display_models = models if show_all else models[:3]
            
            for model_id, description in display_models:
                # Check API key status
                key_available, key_msg = ModelConfig.check_api_key_availability(model_id)
                key_status = " ✅" if key_available else " ❌"
                
                result += f"  • `{model_id}`: {description}{key_status}\n"
            
            if not show_all and len(models) > 3:
                result += f"  ... and {len(models) - 3} more models\n"
            result += "\n"
        
        result += "Legend: ✅ API key available | ❌ API key missing\n\n"
        result += "💡 Usage: `agent = ov.Agent(model='model_id', api_key='your_key')`"
        
        return result
    
    @staticmethod
    def validate_model_setup(model: str, api_key: Optional[str] = None) -> Tuple[bool, str]:
        """Validate if model can be used with current setup"""
        normalized = ModelConfig.normalize_model_id(model)
        if not ModelConfig.is_model_supported(normalized):
            return False, f"Model '{model}' is not supported. Use ov.list_supported_models() to see available models."
        
        # Check API key if needed (model mapping or provider default)
        required_key = PROVIDER_API_KEYS.get(normalized)
        if not required_key:
            provider = ModelConfig.get_provider_from_model(normalized)
            required_key = PROVIDER_DEFAULT_KEYS.get(provider)
        if required_key:
            if api_key or os.getenv(required_key):
                return True, f"✅ Model {normalized} ready to use"
            else:
                provider = ModelConfig.get_provider_from_model(normalized)
                return False, f"❌ Model {normalized} requires {required_key}. Set environment variable or pass api_key parameter."
        
        return True, f"✅ Model {normalized} ready to use"
