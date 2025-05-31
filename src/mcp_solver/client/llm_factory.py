from dataclasses import dataclass, field
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI


try:
    # Import Google chat model if available
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None
import os
import re
import uuid

from dotenv import load_dotenv
from langchain.chat_models.base import BaseChatModel


# Load environment variables from .env file
load_dotenv()


@dataclass
class ModelInfo:
    """Information about a model parsed from the model code."""

    platform: str  # OR, OA, AT, GO, LM
    provider: str  # openai, anthropic, google, lmstudio, ollama
    model_name: str  # The actual model name
    params: dict[str, Any] = field(default_factory=dict)  # Additional parameters

    @property
    def model_string(self) -> str:
        if self.platform == "OR":
            base_string = f"{self.provider}/{self.model_name}"
            # Add tier if present
            if "tier" in self.params:
                return f"{base_string}:{self.params['tier']}"
            return base_string
        return self.model_name

    @property
    def api_key_name(self) -> str:
        platform_to_key = {
            "OR": "OPENROUTER_API_KEY",
            "OA": "OPENAI_API_KEY",
            "AT": "ANTHROPIC_API_KEY",
            "GO": "GOOGLE_API_KEY",
            "LM": "LMSTUDIO_API_KEY",  # Not actually required for LM Studio
        }
        return platform_to_key[self.platform]

    @property
    def base_url(self) -> str | None:
        """Get the API base URL for the platform."""
        if self.platform == "LM" and hasattr(self, "url"):
            return self.url
        elif self.platform == "OR":
            return "https://openrouter.ai/api/v1"
        return None

    def get_param(self, name: str, default: Any = None) -> Any:
        """Get a parameter value with optional default."""
        return self.params.get(name, default)


class LLMFactory:
    """Factory for creating LLM instances based on model code."""

    # Store model info using unique IDs
    _model_info: dict[str, ModelInfo] = {}

    @staticmethod
    def parse_model_code(model_code: str) -> ModelInfo:
        try:
            platform = model_code[:2]
            if platform not in ["OR", "OA", "AT", "GO", "LM"]:
                raise ValueError(f"Unsupported platform prefix: {platform}")
            remaining = model_code[3:]

            # Handle LM Studio format: LM:model(param=value)@url
            if platform == "LM":
                # Extract parameters if present
                params = {}
                param_match = re.match(r"(.+?)(?:\((.+?)\))?@(.+)", remaining)

                if not param_match:
                    # Try original format for backward compatibility
                    match = re.match(r"(.+)@(.+)", remaining)
                    if not match:
                        raise ValueError(
                            f"Invalid LM Studio format. Expected 'LM:model@url' or 'LM:model(params)@url', got '{model_code}'"
                        )

                    model_name = match.group(1)
                    url = match.group(2)
                else:
                    model_name = param_match.group(1)
                    param_str = param_match.group(2)
                    url = param_match.group(3)

                    # Parse parameters if present
                    if param_str:
                        # Split by commas, handling potential comma in values
                        for param_pair in re.findall(
                            r"([^,=]+)=([^,]+)(?:,|$)", param_str
                        ):
                            key, value = param_pair
                            # Convert value types appropriately
                            if value.lower() == "true":
                                parsed_value = True
                            elif value.lower() == "false":
                                parsed_value = False
                            elif re.match(r"^-?\d+$", value):
                                parsed_value = int(value)
                            elif re.match(r"^-?\d*\.\d+$", value):
                                parsed_value = float(value)
                            else:
                                parsed_value = value
                            params[key.strip()] = parsed_value

                model_info = ModelInfo(
                    platform=platform,
                    provider="lmstudio",
                    model_name=model_name,
                    params=params,
                )
                model_info.url = url
                return model_info
            elif platform == "OR":
                provider, model_name = remaining.split("/", 1)
                # Handle any additional parameters after the model name (e.g., :free)
                if ":" in model_name:
                    model_parts = model_name.split(":", 1)
                    model_name = model_parts[0]
                    # Store additional parameter as a param
                    tier = model_parts[1]
                    model_info = ModelInfo(
                        platform=platform,
                        provider=provider,
                        model_name=model_name,
                        params={"tier": tier},
                    )
                    return model_info
            else:
                provider = (
                    "openai"
                    if platform == "OA"
                    else "anthropic"
                    if platform == "AT"
                    else "google"
                )
                # Handle reasoning_effort parameter in model code (format: OA:o3-mini:high)
                if platform == "OA" and ":" in remaining:
                    model_parts = remaining.split(":", 1)
                    model_name = model_parts[0]
                    # Store reasoning_effort as a param
                    reasoning_effort = model_parts[1]
                    model_info = ModelInfo(
                        platform=platform,
                        provider=provider,
                        model_name=model_name,
                        params={"reasoning_effort": reasoning_effort},
                    )
                    return model_info
                else:
                    model_name = remaining
            return ModelInfo(
                platform=platform, provider=provider, model_name=model_name
            )
        except Exception as e:
            raise ValueError(
                f"Invalid model code format: {model_code}. "
                "Expected format: 'OR:provider/model' for OpenRouter, 'OA:model' for OpenAI, "
                "'OA:model:reasoning_effort' for OpenAI with reasoning effort, "
                "'AT:model' for Anthropic, 'GO:model' for Google Gemini, "
                "or 'LM:model@url' or 'LM:model(param=value)@url' for local models"
            ) from e

    @staticmethod
    def get_api_key(model_info: ModelInfo) -> str:
        # LM Studio doesn't require an API key, use a placeholder
        if model_info.platform == "LM":
            return "lm-studio"

        # For other platforms, continue with normal API key retrieval
        api_key = os.environ.get(model_info.api_key_name)
        if not api_key:
            raise ValueError(
                f"{model_info.api_key_name} not found in environment variables. Make sure it's set in your .env file."
            )
        return api_key

    @staticmethod
    def detect_local_server_type(url: str) -> str:
        """Detect the type of local server from the URL."""
        url_lower = url.lower()
        if "ollama" in url_lower or ":11434" in url_lower:
            return "ollama"
        elif "lmstudio" in url_lower or ":1234" in url_lower:
            return "lmstudio"
        # Add other server types as needed
        return "unknown"

    @classmethod
    def create_model(cls, model_code: str, **kwargs) -> BaseChatModel:
        model_info = cls.parse_model_code(model_code)
        api_key = cls.get_api_key(model_info)

        if model_info.platform == "OR":
            model = ChatOpenAI(
                model=model_info.model_string,
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
                **kwargs,
            )

        elif model_info.platform == "OA":
            model_kwargs = kwargs.copy()

            # Add reasoning_effort if specified
            reasoning_effort = model_info.get_param("reasoning_effort")
            if reasoning_effort:
                model_kwargs["model_kwargs"] = model_kwargs.get("model_kwargs", {})
                model_kwargs["model_kwargs"]["reasoning_effort"] = reasoning_effort

            model = ChatOpenAI(
                model=model_info.model_string, api_key=api_key, **model_kwargs
            )
        elif model_info.platform == "AT":
            # Set default max_tokens for Anthropic models if not specified in kwargs
            anthropic_kwargs = kwargs.copy()
            if "max_tokens" not in anthropic_kwargs:
                anthropic_kwargs["max_tokens"] = 4096

            model = ChatAnthropic(
                model=model_info.model_string,
                anthropic_api_key=api_key,
                **anthropic_kwargs,
            )
        elif model_info.platform == "GO":
            model = ChatGoogleGenerativeAI(
                model=model_info.model_string, api_key=api_key, **kwargs
            )
        elif model_info.platform == "LM":
            # For local models, use ChatOpenAI with the provided base_url
            base_url = getattr(model_info, "url", None)
            if not base_url:
                raise ValueError("Local models require a URL (format: LM:model@url)")

            # Detect server type
            server_type = cls.detect_local_server_type(base_url)
            model_info.provider = server_type  # Update provider based on detection

            # Apply model parameters
            model_kwargs = kwargs.copy()

            # Apply temperature if specified
            temp = model_info.get_param("temp")
            if temp is not None:
                model_kwargs["temperature"] = temp

            # Apply max_tokens if specified
            max_tokens = model_info.get_param("max_tokens")
            if max_tokens is not None:
                model_kwargs["max_tokens"] = max_tokens

            # Handle format parameter
            format_param = model_info.get_param("format", "native")

            # For JSON format, we need to add json configuration
            if format_param == "json":
                model_kwargs["model_kwargs"] = model_kwargs.get("model_kwargs", {})

                # Check server type for correct JSON format configuration
                server_type = cls.detect_local_server_type(base_url)

                if server_type == "lmstudio":
                    # LM Studio requires a specific JSON schema format with a 'schema' property
                    model_kwargs["model_kwargs"]["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "content_response",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "content": {
                                        "type": "string",
                                        "description": "The main response content",
                                    }
                                },
                                "required": ["content"],
                            },
                        },
                    }
                else:
                    # Default for most APIs
                    model_kwargs["model_kwargs"]["response_format"] = {
                        "type": "json_object"
                    }

            model = ChatOpenAI(
                model=model_info.model_name,
                api_key=api_key,  # This will be "lm-studio"
                base_url=base_url,
                **model_kwargs,
            )
        else:
            raise ValueError(f"Unsupported platform: {model_info.platform}")

        # Generate a unique ID and store it as an attribute of the model
        model_id = str(uuid.uuid4())
        model._factory_id = model_id
        cls._model_info[model_id] = model_info
        return model

    @classmethod
    def get_provider(cls, model: BaseChatModel) -> str | None:
        """Get the provider name for a model instance."""
        if hasattr(model, "_factory_id"):
            if model_info := cls._model_info.get(model._factory_id):
                return model_info.provider
        return None

    @classmethod
    def get_model_info(cls, model: BaseChatModel) -> ModelInfo | None:
        """Get the full ModelInfo for a model instance."""
        if hasattr(model, "_factory_id"):
            return cls._model_info.get(model._factory_id)
        return None

    # ===== TESTING UTILITIES =====

    @classmethod
    def check_api_key_available(cls, model_code: str) -> tuple[bool, str]:
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

            # LM Studio doesn't require an API key
            if model_info.platform == "LM":
                return True, "LMSTUDIO_API_KEY (Not required)"

            key_name = model_info.api_key_name
            key_available = bool(os.environ.get(key_name))
            return key_available, key_name
        except Exception as e:
            return False, str(e)

    @classmethod
    def test_tool_calling_capability(cls, model_code: str) -> tuple[bool, str]:
        """Test if a model supports tool calling.

        Returns:
            Tuple containing:
            - Boolean indicating if tool calling is supported
            - String with details about the supported method (native/json/none)
        """
        try:
            model_info = cls.parse_model_code(model_code)
            if model_info.platform != "LM":
                # For non-local models, assume they support tool calling
                return True, "assumed_supported"

            # For local models, check the format parameter
            format_param = model_info.get_param("format", "native")

            # If explicitly set to json, use json format
            if format_param == "json":
                return True, "json"

            # For local models, we need to check the server type
            base_url = getattr(model_info, "url", None)
            server_type = cls.detect_local_server_type(base_url)

            # Different testing logic based on server type
            if server_type == "ollama":
                # For Ollama, it depends on the model
                # Models like llama3.2 may support native function calling
                # We'd need to test the specific model, but for now we'll
                # assume native support as a default for Ollama
                return True, "native"

            # Default to JSON format for unknown servers
            return True, "json"

        except Exception as e:
            return False, str(e)

    @classmethod
    def get_expected_model_type(cls, model_code: str) -> tuple[Any, str]:
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
        elif model_info.platform == "LM":
            server_type = "LM Studio (local)"
            if hasattr(model_info, "url"):
                server_type = cls.detect_local_server_type(model_info.url)
                if server_type == "ollama":
                    server_type = "Ollama (local)"
                elif server_type == "lmstudio":
                    server_type = "LM Studio (local)"
                else:
                    server_type = f"Unknown local server ({server_type})"
            return ChatOpenAI, server_type
        else:  # OpenRouter
            if model_info.provider == "openai":
                return ChatOpenAI, "OpenRouter (OpenAI)"
            elif model_info.provider == "anthropic":
                return ChatAnthropic, "OpenRouter (Anthropic)"
            else:
                raise ValueError(
                    f"Unsupported provider for OpenRouter: {model_info.provider}"
                )

    @classmethod
    def test_create_model(
        cls, model_code: str
    ) -> tuple[bool, str, BaseChatModel | None]:
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
                    base_url="https://openrouter.ai/api/v1",
                )
            elif model_info.platform == "OA":
                model = ChatOpenAI(
                    model=model_info.model_string, api_key=os.environ.get(key_name)
                )
            elif model_info.platform == "AT":
                model = ChatAnthropic(
                    model=model_info.model_string,
                    anthropic_api_key=os.environ.get(key_name),
                )
            elif model_info.platform == "GO":
                model = ChatGoogleGenerativeAI(
                    model=model_info.model_string, api_key=os.environ.get(key_name)
                )
            elif model_info.platform == "LM":
                # For local models, use the URL
                base_url = getattr(model_info, "url", None)

                # Apply model parameters for testing
                model_kwargs = {}

                # Apply temperature if specified
                temp = model_info.get_param("temp")
                if temp is not None:
                    model_kwargs["temperature"] = temp

                # Apply max_tokens if specified
                max_tokens = model_info.get_param("max_tokens")
                if max_tokens is not None:
                    model_kwargs["max_tokens"] = max_tokens

                # Handle format parameter
                format_param = model_info.get_param("format", "native")

                # For JSON format, add json configuration
                if format_param == "json":
                    model_kwargs["model_kwargs"] = {}

                    # Check server type for correct JSON format configuration
                    server_type = cls.detect_local_server_type(base_url)

                    if server_type == "lmstudio":
                        # LM Studio requires a specific JSON schema format with a 'schema' property
                        model_kwargs["model_kwargs"]["response_format"] = {
                            "type": "json_schema",
                            "json_schema": {
                                "name": "content_response",
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "content": {
                                            "type": "string",
                                            "description": "The main response content",
                                        }
                                    },
                                    "required": ["content"],
                                },
                            },
                        }
                    else:
                        # Default for most APIs
                        model_kwargs["model_kwargs"]["response_format"] = {
                            "type": "json_object"
                        }

                model = ChatOpenAI(
                    model=model_info.model_name,
                    api_key="lm-studio",  # Placeholder value
                    base_url=base_url,
                    **model_kwargs,
                )

            if model and isinstance(model, expected_type):
                return (
                    True,
                    f"Successfully created {provider_name} model instance",
                    model,
                )
            else:
                return False, "Failed to create model instance of correct type", None

        except Exception as e:
            return False, f"Error during model creation: {e!s}", None
