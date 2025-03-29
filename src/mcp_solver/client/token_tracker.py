"""
Token usage tracking module for MCP Solver.

This module provides a model-agnostic approach for tracking token usage
across different LLM providers within LangGraph applications.
"""

import re
from typing import Dict, Optional, Union, List, Any, Callable
from dataclasses import dataclass, field

# Import LangChain callback base class
try:
    from langchain.callbacks.base import BaseCallbackHandler
    LANGCHAIN_CALLBACKS_AVAILABLE = True
except ImportError:
    LANGCHAIN_CALLBACKS_AVAILABLE = False


@dataclass
class TokenUsage:
    """Track token usage for a single component or for cumulative usage."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    call_count: int = 0

    def update(self, prompt_tokens: int = 0, completion_tokens: int = 0):
        """Update token counts."""
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens = self.prompt_tokens + self.completion_tokens
        self.call_count += 1


class TokenTracker:
    """
    Singleton token tracker that works across different LLM providers.
    
    This class provides a model-agnostic approach to track token usage
    for both the ReAct agent and the Review agent.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TokenTracker, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            # Create usage trackers for different components
            self.react_agent_usage = TokenUsage()
            self.review_agent_usage = TokenUsage()
            self.total_usage = TokenUsage()
            
            # State tracking
            self.current_component = "unknown"
            self.enabled = True
            
            self._initialized = True
    
    @staticmethod
    def get_instance():
        """Get the singleton instance."""
        return TokenTracker()
    
    def reset(self):
        """Reset all token usage statistics."""
        # Create fresh TokenUsage instances for all components
        self.react_agent_usage = TokenUsage()
        self.review_agent_usage = TokenUsage()
        self.total_usage = TokenUsage()
        self.current_component = "unknown"
        
        # Also clear any static variables or state that might persist between runs
        if hasattr(self, '_remaining_tokens'):
            delattr(self, '_remaining_tokens')
    
    def debug_state(self):
        """Print the current state of the token tracker for debugging."""
        print("===== TOKEN TRACKER DEBUG INFO =====")
        print(f"Current component: {self.current_component}")
        print(f"ReAct agent calls: {self.react_agent_usage.call_count}")
        print(f"ReAct agent prompt tokens: {self.react_agent_usage.prompt_tokens}")
        print(f"ReAct agent completion tokens: {self.react_agent_usage.completion_tokens}")
        print(f"Review agent calls: {self.review_agent_usage.call_count}")
        print(f"Review agent prompt tokens: {self.review_agent_usage.prompt_tokens}")
        print(f"Review agent completion tokens: {self.review_agent_usage.completion_tokens}")
        print("===================================")
    
    def set_component(self, component_name: str):
        """Set the current component being tracked."""
        self.current_component = component_name
    
    def count_tokens(self, text: str, model: str = "unknown") -> int:
        """
        Count tokens for a given text using approximation.
        
        This uses a simple approximation based on characters and words
        that works reasonably well across different models.
        
        Args:
            text: The text to count tokens for
            model: The model name (used only for classification)
            
        Returns:
            Estimated token count
        """
        if not self.enabled or not text:
            return 0
        
        # Handle different model types based on naming patterns
        model_lower = model.lower()
        if "claude" in model_lower:
            return self._estimate_claude_tokens(text)
        elif any(gpt_model in model_lower for gpt_model in ["gpt-", "o1", "o2", "o3"]):
            return self._estimate_gpt_tokens(text)
        else:
            # Use a general approximation for all other models
            return self._estimate_general_tokens(text)
    
    def _estimate_general_tokens(self, text: str) -> int:
        """
        General token estimation that works reasonably well for most models.
        
        Args:
            text: The text to count tokens for
            
        Returns:
            Estimated token count
        """
        # Simple word-based approximation
        # Most tokenizers average around 0.75 tokens per word
        words = text.split()
        word_estimate = len(words)
        
        # Character-based refinement
        # Short words often become a single token, longer ones split into multiple
        # On average, 1 token is about 4-5 characters in English text
        char_estimate = len(text) / 4
        
        # Combine the estimates with a weighted average
        # This empirically produces a reasonable approximation for most models
        return max(1, int((word_estimate + char_estimate) / 2))
    
    def _estimate_claude_tokens(self, text: str) -> int:
        """
        Estimate token count for Claude models.
        
        This is a reasonable approximation for Anthropic Claude models.
        
        Args:
            text: The text to count tokens for
            
        Returns:
            Estimated token count
        """
        # Claude models typically use about 4 characters per token
        char_count = len(text)
        return max(1, int(char_count / 4))
    
    def _estimate_gpt_tokens(self, text: str) -> int:
        """
        Estimate token count for GPT models.
        
        This is a reasonable approximation for OpenAI GPT models.
        
        Args:
            text: The text to count tokens for
            
        Returns:
            Estimated token count
        """
        # GPT models tend to use slightly more tokens per word than Claude
        # We use a mix of character and word-based approximation
        char_count = len(text)
        word_count = len(text.split())
        
        # GPT tends to use ~4.5 chars/token on average for English text
        # But also about 1.3 tokens per word
        char_based = char_count / 4.5
        word_based = word_count * 1.3
        
        # Take a weighted average
        return max(1, int((char_based + word_based) / 2))
    
    def update(self, 
               prompt: Optional[Union[str, List[Dict[str, str]], int, List]] = None, 
               completion: Optional[Union[str, int]] = None,
               model: str = "unknown"):
        """
        Update token usage with a new prompt and/or completion.
        
        Args:
            prompt: The prompt text, message list, or pre-counted token count
            completion: The completion text or pre-counted token count
            model: The model identifier
        """
        if not self.enabled:
            return
        
        # Count prompt tokens
        prompt_tokens = 0
        if prompt is not None:
            if isinstance(prompt, int):
                # Already counted tokens
                prompt_tokens = prompt
            elif isinstance(prompt, str):
                # Simple string prompt
                prompt_tokens = self.count_tokens(prompt, model)
            elif isinstance(prompt, list):
                # Handle both message lists and plain messages
                for message in prompt:
                    if isinstance(message, dict) and "content" in message:
                        # Dictionary format message
                        content = message["content"]
                        if content:
                            if isinstance(content, str):
                                prompt_tokens += self.count_tokens(content, model)
                            elif isinstance(content, list):
                                # Handle content that might be a list of content chunks
                                for chunk in content:
                                    if isinstance(chunk, dict) and "text" in chunk:
                                        prompt_tokens += self.count_tokens(chunk["text"], model)
                                    elif isinstance(chunk, str):
                                        prompt_tokens += self.count_tokens(chunk, model)
                    elif hasattr(message, "content"):
                        # LangChain message objects
                        content = message.content
                        if content:
                            if isinstance(content, str):
                                prompt_tokens += self.count_tokens(content, model)
                            elif isinstance(content, list):
                                # Handle content that might be a list of content chunks
                                for chunk in content:
                                    if isinstance(chunk, dict) and "text" in chunk:
                                        prompt_tokens += self.count_tokens(chunk["text"], model)
                                    elif isinstance(chunk, str):
                                        prompt_tokens += self.count_tokens(chunk, model)
                    elif isinstance(message, str):
                        # Plain string in a list
                        prompt_tokens += self.count_tokens(message, model)
            elif isinstance(prompt, dict) and "messages" in prompt:
                # Handle LangGraph-style state objects
                messages = prompt["messages"]
                if isinstance(messages, list):
                    for message in messages:
                        if isinstance(message, dict) and "content" in message:
                            prompt_tokens += self.count_tokens(message["content"], model)
                        elif hasattr(message, "content"):
                            prompt_tokens += self.count_tokens(message.content, model)
        
        # Count completion tokens
        completion_tokens = 0
        if completion is not None:
            if isinstance(completion, int):
                # Already counted tokens
                completion_tokens = completion
            elif isinstance(completion, str):
                completion_tokens = self.count_tokens(completion, model)
            elif isinstance(completion, dict):
                # Try to extract content from dictionary format
                if "content" in completion:
                    content = completion["content"]
                    if isinstance(content, str):
                        completion_tokens = self.count_tokens(content, model)
                    elif isinstance(content, list):
                        # Handle content that might be a list of content chunks
                        for chunk in content:
                            if isinstance(chunk, dict) and "text" in chunk:
                                completion_tokens += self.count_tokens(chunk["text"], model)
                            elif isinstance(chunk, str):
                                completion_tokens += self.count_tokens(chunk, model)
            elif hasattr(completion, "content"):
                # Handle object with content attribute (like LangChain messages)
                content = completion.content
                if isinstance(content, str):
                    completion_tokens = self.count_tokens(content, model)
                elif isinstance(content, list):
                    # Handle content that might be a list of content chunks
                    for chunk in content:
                        if isinstance(chunk, dict) and "text" in chunk:
                            completion_tokens += self.count_tokens(chunk["text"], model)
                        elif isinstance(chunk, str):
                            completion_tokens += self.count_tokens(chunk, model)
        
        # Ensure we only update the appropriate component
        if self.current_component == "react_agent":
            # Only update react_agent usage
            self.react_agent_usage.update(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens
            )
            # Update total usage as well
            self.total_usage.update(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens
            )
        elif self.current_component == "review_agent":
            # Only update review_agent usage
            self.review_agent_usage.update(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens
            )
            # Update total usage as well
            self.total_usage.update(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens
            )
        else:
            # For unknown components, just update total
            self.total_usage.update(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens
            )
    
    def create_callback(self, component: str = "react_agent"):
        """
        Create a LangChain-compatible callback for token tracking.
        
        Args:
            component: The component name to associate with this callback
            
        Returns:
            A LangChain callback handler that tracks token usage
        """
        # Only create if LangChain callbacks are available
        if not LANGCHAIN_CALLBACKS_AVAILABLE:
            return None
        
        # Reset call count and token counts for this component to avoid accumulation
        if component == "react_agent":
            self.react_agent_usage = TokenUsage()
        elif component == "review_agent":
            self.review_agent_usage = TokenUsage()
            
        tracker = self
        
        # Create a LangChain callback handler
        class TokenTrackingCallbackHandler(BaseCallbackHandler):
            """LangChain callback handler that tracks token usage."""
            
            def __init__(self):
                super().__init__()
                self.component = component
                
            def on_llm_start(self, serialized, prompts, **kwargs):
                try:
                    # Extract model name if available
                    model_name = "unknown"
                    if isinstance(serialized, dict):
                        if "name" in serialized:
                            model_name = serialized["name"]
                        elif "model_name" in serialized:
                            model_name = serialized["model_name"]
                    
                    # Set component based on the component specified at callback creation
                    # This is crucial to keep react and review agents separate
                    tracker.set_component(self.component)
                    
                    # Always increment call count directly for the specific component
                    if self.component == "react_agent":
                        tracker.react_agent_usage.call_count += 1
                    elif self.component == "review_agent":
                        tracker.review_agent_usage.call_count += 1
                    # Always update total call count
                    tracker.total_usage.call_count += 1
                    
                    # Track tokens, which will be assigned to the correct component
                    tracker.update(prompt=prompts, model=model_name)
                except Exception as e:
                    # More informative error handling
                    import sys
                    print(f"Error in token tracking on_llm_start: {str(e)}", file=sys.stderr)
            
            def on_llm_end(self, response, **kwargs):
                try:
                    # Set component based on the component specified at callback creation time
                    # Not based on the current tracker state which could have changed
                    tracker.set_component(self.component)
                    
                    # Extract model name if available
                    model_name = "unknown"
                    if hasattr(response, "llm_output") and response.llm_output:
                        if isinstance(response.llm_output, dict):
                            model_name = response.llm_output.get("model_name", "unknown")
                            
                            # Extract token usage from metadata if available (OpenAI, Anthropic style)
                            token_usage = response.llm_output.get("token_usage", {})
                            if token_usage:
                                prompt_tokens = token_usage.get("prompt_tokens", 0)
                                completion_tokens = token_usage.get("completion_tokens", 0)
                                if prompt_tokens > 0 or completion_tokens > 0:
                                    # Update for this specific component
                                    tracker.update(
                                        prompt=prompt_tokens if prompt_tokens > 0 else None,
                                        completion=completion_tokens if completion_tokens > 0 else None,
                                        model=model_name
                                    )
                                    return  # Skip text-based extraction if we got metadata
                    
                    # Try extracting usage stats directly
                    usage = getattr(response, "usage", None)
                    if usage:
                        prompt_tokens = getattr(usage, "prompt_tokens", 0)
                        completion_tokens = getattr(usage, "completion_tokens", 0)
                        if prompt_tokens > 0 or completion_tokens > 0:
                            # Update for this specific component
                            tracker.update(
                                prompt=prompt_tokens if prompt_tokens > 0 else None,
                                completion=completion_tokens if completion_tokens > 0 else None,
                                model=model_name
                            )
                            return  # Skip text-based extraction if we got metadata
                            
                    # Check for Anthropic-specific token usage structure
                    if hasattr(response, "usage") and isinstance(response.usage, dict):
                        prompt_tokens = response.usage.get("input_tokens", 0)
                        completion_tokens = response.usage.get("output_tokens", 0)
                        if prompt_tokens > 0 or completion_tokens > 0:
                            # Update for this specific component
                            tracker.update(
                                prompt=prompt_tokens if prompt_tokens > 0 else None,
                                completion=completion_tokens if completion_tokens > 0 else None,
                                model=model_name
                            )
                            return  # Skip text-based extraction if we got metadata
                    
                    # LangGraph specific handling - check for a usage dictionary directly in the response
                    if isinstance(response, dict) and "usage" in response:
                        usage_dict = response["usage"]
                        if isinstance(usage_dict, dict):
                            prompt_tokens = usage_dict.get("prompt_tokens", 0) or usage_dict.get("input_tokens", 0)
                            completion_tokens = usage_dict.get("completion_tokens", 0) or usage_dict.get("output_tokens", 0)
                            if prompt_tokens > 0 or completion_tokens > 0:
                                # Update for this specific component
                                tracker.update(
                                    prompt=prompt_tokens if prompt_tokens > 0 else None,
                                    completion=completion_tokens if completion_tokens > 0 else None,
                                    model=model_name
                                )
                                return  # Skip text-based extraction if we got metadata
                    
                    # Extract text content from the response for token counting
                    response_text = None
                    
                    # Try multiple approaches to extract generation text
                    completion_text = None
                    
                    # If response is directly a string
                    if isinstance(response, str):
                        completion_text = response
                    
                    # Check if response is a message with content
                    elif hasattr(response, "content"):
                        completion_text = response.content
                    
                    # Check for generations attribute (standard LangChain format)
                    elif hasattr(response, "generations"):
                        for gen in response.generations:
                            if hasattr(gen, "text"):
                                completion_text = gen.text
                                break
                            elif isinstance(gen, dict) and "text" in gen:
                                completion_text = gen["text"]
                                break
                    
                    # If response is a dict with content
                    elif isinstance(response, dict) and "content" in response:
                        completion_text = response["content"]
                    
                    # Try message format
                    elif hasattr(response, "message") and hasattr(response.message, "content"):
                        completion_text = response.message.content
                        
                    # Check for LangGraph message structure
                    elif isinstance(response, dict) and "messages" in response:
                        messages = response["messages"]
                        if isinstance(messages, list) and len(messages) > 0:
                            last_message = messages[-1]
                            if hasattr(last_message, "content"):
                                completion_text = last_message.content
                            elif isinstance(last_message, dict) and "content" in last_message:
                                completion_text = last_message["content"]
                    
                    # For any response type, if we found text content, update tokens
                    if completion_text:
                        # This will be tracked to the specific component we set earlier
                        tracker.update(completion=completion_text, model=model_name)
                    else:
                        # Even if we couldn't extract text, record a minimal token count
                        # to ensure we have some completion tokens recorded
                        # This will be tracked to the specific component we set earlier
                        tracker.update(completion=5, model=model_name)
                        
                except Exception as e:
                    # Use a less silent error handling for debugging
                    import sys
                    print(f"Error in token tracking on_llm_end: {str(e)}", file=sys.stderr)
                    # Record at least something even if we had an error
                    # This will be tracked to the specific component we set earlier
                    tracker.update(completion=5, model=model_name)
            
            # Required for LangGraph compatibility
            def run_inline(self) -> bool:
                """Return whether callback handlers should run in the same thread."""
                return True
        
        return TokenTrackingCallbackHandler()
    
    def format_token_count(self, count: int) -> str:
        """
        Format token count with compact magnitude representation.
        
        Args:
            count: The token count to format
            
        Returns:
            Formatted representation with appropriate magnitude suffix
        """
        if count < 1000:
            return str(count)
        elif count < 1000000:
            # Scale to thousands with conditional precision
            scaled = count / 1000
            if scaled < 10:
                # For 1k-9.9k, maintain single decimal precision
                return f"{scaled:.1f}".rstrip('0').rstrip('.') + 'k'
            else:
                # For ≥10k, use integer representation
                return f"{int(scaled)}k"
        else:
            # Scale to millions with conditional precision
            scaled = count / 1000000
            if scaled < 10:
                # For 1M-9.9M, maintain single decimal precision
                return f"{scaled:.1f}".rstrip('0').rstrip('.') + 'M'
            else:
                # For ≥10M, use integer representation
                return f"{int(scaled)}M"
    
    def print_usage(self):
        """Print token usage statistics with concise formatting."""
        # No longer use fallback estimation
        
        react_in = self.format_token_count(self.react_agent_usage.prompt_tokens)
        react_out = self.format_token_count(self.react_agent_usage.completion_tokens)
        review_in = self.format_token_count(self.review_agent_usage.prompt_tokens)
        review_out = self.format_token_count(self.review_agent_usage.completion_tokens)
        total_in = self.format_token_count(self.total_usage.prompt_tokens)
        total_out = self.format_token_count(self.total_usage.completion_tokens)
        
        # Include call counts for better diagnostics
        react_calls = self.react_agent_usage.call_count
        review_calls = self.review_agent_usage.call_count
        total_calls = self.total_usage.call_count
        
        print(f"Token Usage: ReAct agent in={react_in} out={react_out} (calls={react_calls}) | Review agent in={review_in} out={review_out} (calls={review_calls}) | Total in={total_in} out={total_out} (calls={total_calls})")


# Create a singleton instance for easy access
token_tracker = TokenTracker.get_instance() 