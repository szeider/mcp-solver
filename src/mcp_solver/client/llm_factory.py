from dataclasses import dataclass
from typing import Optional, Dict, Tuple, Any
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
try:
    # Import Google chat model if available
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None
from langchain.chat_models.base import BaseChatModel
import os
from dotenv import load_dotenv
import uuid

# Load environment variables from .env file
load_dotenv()

@dataclass
class ModelInfo:
    """Information about a model parsed from the model code."""
    platform: str  # OR, OA, AT, GO
    provider: str  # openai, anthropic, google
    model_name: str  # The actual model name

    @property
    def model_string(self) -> str:
        if self.platform == "OR":
            return f"{self.provider}/{self.model_name}"
        return self.model_name

    @property
    def api_key_name(self) -> str:
        platform_to_key = {
            "OR": "OPENROUTER_API_KEY",
            "OA": "OPENAI_API_KEY",
            "AT": "ANTHROPIC_API_KEY",
            "GO": "GOOGLE_API_KEY"
        }
        return platform_to_key[self.platform]

class LLMFactory:
    """Factory for creating LLM instances based on model code."""
    
    # Store model info using unique IDs
    _model_info: Dict[str, ModelInfo] = {}
    
    @staticmethod
    def parse_model_code(model_code: str) -> ModelInfo:
        try:
            platform = model_code[:2]
            if platform not in ["OR", "OA", "AT", "GO"]:
                raise ValueError(f"Unsupported platform prefix: {platform}")
            remaining = model_code[3:]
            if platform == "OR":
                provider, model_name = remaining.split("/", 1)
            else:
                provider = "openai" if platform == "OA" else "anthropic" if platform == "AT" else "google"
                # Handle reasoning_effort parameter in model code (format: OA:o3-mini:high)
                if platform == "OA" and ":" in remaining:
                    model_parts = remaining.split(":", 1)
                    model_name = model_parts[0]
                    # Store reasoning_effort as an additional attribute
                    reasoning_effort = model_parts[1]
                    model_info = ModelInfo(platform=platform, provider=provider, model_name=model_name)
                    setattr(model_info, 'reasoning_effort', reasoning_effort)
                    return model_info
                else:
                    model_name = remaining
            return ModelInfo(platform=platform, provider=provider, model_name=model_name)
        except Exception as e:
            raise ValueError(
                f"Invalid model code format: {model_code}. "
                "Expected format: 'OR:provider/model' for OpenRouter, 'OA:model' for OpenAI, "
                "'OA:model:reasoning_effort' for OpenAI with reasoning effort, "
                "'AT:model' for Anthropic or 'GO:model' for Google Gemini"
            ) from e

    @staticmethod
    def get_api_key(model_info: ModelInfo) -> str:
        api_key = os.environ.get(model_info.api_key_name)
        if not api_key:
            raise ValueError(
                f"{model_info.api_key_name} not found in environment variables. Make sure it's set in your .env file."
            )
        return api_key

    @classmethod
    def create_model(cls, model_code: str, **kwargs) -> BaseChatModel:
        model_info = cls.parse_model_code(model_code)
        api_key = cls.get_api_key(model_info)
        
        if model_info.platform == "OR":
            
            model = ChatOpenAI(
                model=model_info.model_string,
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
                **kwargs
            )
        
        elif model_info.platform == "OA":
            model_kwargs = kwargs.copy()
            
            # Add reasoning_effort if specified in the model code
            if hasattr(model_info, 'reasoning_effort'):
                model_kwargs['model_kwargs'] = model_kwargs.get('model_kwargs', {})
                model_kwargs['model_kwargs']['reasoning_effort'] = model_info.reasoning_effort
            
            model = ChatOpenAI(
                model=model_info.model_string,
                api_key=api_key,
                **model_kwargs
            )
        elif model_info.platform == "AT":
            # Set default max_tokens for Anthropic models if not specified in kwargs
            anthropic_kwargs = kwargs.copy()
            if 'max_tokens' not in anthropic_kwargs:
                anthropic_kwargs['max_tokens'] = 4096
            
            model = ChatAnthropic(
                model=model_info.model_string,
                anthropic_api_key=api_key,
                **anthropic_kwargs
            )
        elif model_info.platform == "GO":
            model = ChatGoogleGenerativeAI(
                model=model_info.model_string,
                api_key=api_key,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported platform: {model_info.platform}")
        
        # Generate a unique ID and store it as an attribute of the model
        model_id = str(uuid.uuid4())
        setattr(model, '_factory_id', model_id)
        cls._model_info[model_id] = model_info
        return model

    @classmethod
    def get_provider(cls, model: BaseChatModel) -> Optional[str]:
        """Get the provider name for a model instance."""
        if hasattr(model, '_factory_id'):
            if model_info := cls._model_info.get(model._factory_id):
                return model_info.provider
        return None

    @classmethod
    def get_model_info(cls, model: BaseChatModel) -> Optional[ModelInfo]:
        """Get the full ModelInfo for a model instance."""
        if hasattr(model, '_factory_id'):
            return cls._model_info.get(model._factory_id)
        return None
        
    # ===== TESTING UTILITIES =====
    
    @classmethod
    def check_api_key_available(cls, model_code: str) -> Tuple[bool, str]:
        """
        Check if the required API key for a given model code is available.
        
        Args:
            model_code: The model code to check
            
        Returns:
            Tuple containing:
            - Boolean indicating if the key is available
            - String with the name of the required environment variable
        """
        try:
            model_info = cls.parse_model_code(model_code)
            key_name = model_info.api_key_name
            key_available = bool(os.environ.get(key_name))
            return key_available, key_name
        except Exception as e:
            return False, str(e)
            
    @classmethod
    def get_expected_model_type(cls, model_code: str) -> Tuple[Any, str]:
        """
        Get the expected model class type for a given model code.
        
        Args:
            model_code: The model code to check
            
        Returns:
            Tuple containing:
            - The expected model class type
            - Provider name string
        """
        model_info = cls.parse_model_code(model_code)
        
        if model_info.platform == "OA":
            return ChatOpenAI, "OpenAI"
        elif model_info.platform == "AT":
            return ChatAnthropic, "Anthropic"
        elif model_info.platform == "GO":
            return ChatGoogleGenerativeAI, "Google Gemini"
        else:  # OpenRouter
            if model_info.provider == "openai":
                return ChatOpenAI, "OpenRouter (OpenAI)"
            elif model_info.provider == "anthropic":
                return ChatAnthropic, "OpenRouter (Anthropic)"
            else:
                raise ValueError(f"Unsupported provider for OpenRouter: {model_info.provider}")
    
    @classmethod
    def test_create_model(cls, model_code: str) -> Tuple[bool, str, Optional[BaseChatModel]]:
        """
        Test if a model can be created without making API calls.
        
        Args:
            model_code: The model code to test
            
        Returns:
            Tuple containing:
            - Boolean indicating success
            - Message with details
            - The created model instance (if successful) or None
        """
        try:
            # First verify we have a valid model code
            model_info = cls.parse_model_code(model_code)
            
            # Check if API key is available 
            api_key_available, key_name = cls.check_api_key_available(model_code)
            if not api_key_available:
                return False, f"API key not available: {key_name}", None
                
            # Get the expected model type
            expected_type, provider_name = cls.get_expected_model_type(model_code)
            
            # For test purposes, we'll only verify the model instantiation
            # without making actual API calls
            model = None
            
            if model_info.platform == "OR":
                model = ChatOpenAI(
                    model=model_info.model_string,
                    api_key=os.environ.get(key_name),
                    base_url="https://openrouter.ai/api/v1"
                )
            elif model_info.platform == "OA":
                model = ChatOpenAI(
                    model=model_info.model_string,
                    api_key=os.environ.get(key_name)
                )
            elif model_info.platform == "AT":
                model = ChatAnthropic(
                    model=model_info.model_string,
                    anthropic_api_key=os.environ.get(key_name)
                )
            elif model_info.platform == "GO":
                model = ChatGoogleGenerativeAI(
                    model=model_info.model_string,
                    api_key=os.environ.get(key_name)
                )
                
            if model and isinstance(model, expected_type):
                return True, f"Successfully created {provider_name} model instance", model
            else:
                return False, f"Failed to create model instance of correct type", None
                
        except Exception as e:
            return False, f"Error during model creation: {str(e)}", None 