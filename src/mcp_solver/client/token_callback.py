"""Token usage callback handler for capturing LLM token counts."""

from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import LLMResult


class TokenUsageCallbackHandler(BaseCallbackHandler):
    """Callback handler to capture token usage from LLM responses."""

    def __init__(self, token_counter, agent_type: str = "main"):
        """Initialize the callback handler.

        Args:
            token_counter: TokenCounter instance to update
            agent_type: Either "main" or "reviewer" to track different agents
        """
        self.token_counter = token_counter
        self.agent_type = agent_type

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """Capture token usage when LLM call completes.

        This method captures exact token counts from Anthropic models.
        Other providers may also provide token counts in different formats.
        For ReAct agents, this accumulates tokens across multiple LLM calls.
        """
        if not response.llm_output:
            return

        # Anthropic format: response.llm_output["usage"]
        if "usage" in response.llm_output:
            usage = response.llm_output["usage"]
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)

            if self.agent_type == "main":
                # For main agent, accumulate tokens (multiple calls in ReAct)
                self.token_counter.main_input_tokens += input_tokens
                self.token_counter.main_output_tokens += output_tokens
                self.token_counter.main_is_exact = True
            else:
                # For reviewer, set tokens (single call)
                self.token_counter.reviewer_input_tokens = input_tokens
                self.token_counter.reviewer_output_tokens = output_tokens
                self.token_counter.reviewer_is_exact = True

        # OpenAI format: response.llm_output["token_usage"]
        elif "token_usage" in response.llm_output:
            usage = response.llm_output["token_usage"]
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)

            if self.agent_type == "main":
                # For main agent, accumulate tokens (multiple calls in ReAct)
                self.token_counter.main_input_tokens += input_tokens
                self.token_counter.main_output_tokens += output_tokens
                self.token_counter.main_is_exact = True
            else:
                # For reviewer, set tokens (single call)
                self.token_counter.reviewer_input_tokens = input_tokens
                self.token_counter.reviewer_output_tokens = output_tokens
                self.token_counter.reviewer_is_exact = True
