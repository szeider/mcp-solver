"""
Token counter for tracking token usage in LangGraph applications.

This module provides a class for tracking token usage during a client run,
separately for the main agent and reviewer agent.
"""

from rich.console import Console
from rich.table import Table
from typing import Union, List, Any


class TokenCounter:
    """Token counter for tracking token usage in LangGraph applications."""

    _instance = None

    @classmethod
    def get_instance(cls):
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = TokenCounter()
        return cls._instance

    def __init__(self):
        """Initialize token counter."""
        self.main_input_tokens = 0
        self.main_output_tokens = 0
        self.reviewer_input_tokens = 0
        self.reviewer_output_tokens = 0
        self.console = Console()
        # Use 3.5 characters per token as default ratio - a reasonable average for many models
        self.chars_per_token = 3.5

    def set_chars_per_token(self, ratio: float):
        """Set the characters per token ratio for estimation."""
        self.chars_per_token = ratio

    def estimate_tokens(self, content: Union[str, List[Any]]) -> int:
        """Estimate the number of tokens based on characters.

        Args:
            content: String or list of objects with content attributes

        Returns:
            Estimated token count
        """
        if isinstance(content, str):
            # Simple case - just a string
            return int(len(content) / self.chars_per_token)
        elif isinstance(content, list):
            # If it's a list, try to extract content from each item
            char_count = 0
            for item in content:
                if hasattr(item, "content"):
                    char_count += len(str(item.content))
                elif isinstance(item, dict) and "content" in item:
                    char_count += len(str(item["content"]))
                else:
                    # Fallback: use the string representation
                    char_count += len(str(item))
            return int(char_count / self.chars_per_token)
        else:
            # Any other object - use string representation
            return int(len(str(content)) / self.chars_per_token)

    def count_main_input(self, content):
        """Count tokens for main agent input."""
        tokens = self.estimate_tokens(content)
        self.add_main_input_tokens(tokens)
        return tokens

    def count_main_output(self, content):
        """Count tokens for main agent output."""
        tokens = self.estimate_tokens(content)
        self.add_main_output_tokens(tokens)
        return tokens

    def count_reviewer_input(self, content):
        """Count tokens for reviewer input."""
        tokens = self.estimate_tokens(content)
        self.add_reviewer_input_tokens(tokens)
        return tokens

    def count_reviewer_output(self, content):
        """Count tokens for reviewer output."""
        tokens = self.estimate_tokens(content)
        self.add_reviewer_output_tokens(tokens)
        return tokens

    def add_main_input_tokens(self, count):
        """Add input tokens to main agent counter."""
        self.main_input_tokens += count

    def add_main_output_tokens(self, count):
        """Add output tokens to main agent counter."""
        self.main_output_tokens += count

    def add_reviewer_input_tokens(self, count):
        """Add input tokens to reviewer agent counter."""
        self.reviewer_input_tokens += count

    def add_reviewer_output_tokens(self, count):
        """Add output tokens to reviewer agent counter."""
        self.reviewer_output_tokens += count

    def get_stats_table(self):
        """Get token usage statistics as a Rich table."""
        from mcp_solver.client.client import format_token_count

        table = Table(title="Token Usage Statistics")
        table.add_column("Agent", style="cyan")
        table.add_column("Input", style="green")
        table.add_column("Output", style="magenta")
        table.add_column("Total", style="yellow")

        # Main agent row
        main_total = self.main_input_tokens + self.main_output_tokens
        table.add_row(
            "ReAct Agent",
            format_token_count(self.main_input_tokens),
            format_token_count(self.main_output_tokens),
            format_token_count(main_total),
        )

        # Reviewer agent row
        reviewer_total = self.reviewer_input_tokens + self.reviewer_output_tokens
        table.add_row(
            "Reviewer",
            format_token_count(self.reviewer_input_tokens),
            format_token_count(self.reviewer_output_tokens),
            format_token_count(reviewer_total),
        )

        # Total row
        grand_total = main_total + reviewer_total
        table.add_row(
            "COMBINED",
            format_token_count(self.main_input_tokens + self.reviewer_input_tokens),
            format_token_count(self.main_output_tokens + self.reviewer_output_tokens),
            format_token_count(grand_total),
            style="bold",
        )

        return table

    def print_stats(self):
        """Print token usage statistics."""
        table = self.get_stats_table()
        self.console.print("\n")
        self.console.print(table)

    def get_total_tokens(self):
        """Get the total number of tokens used."""
        main_total = self.main_input_tokens + self.main_output_tokens
        reviewer_total = self.reviewer_input_tokens + self.reviewer_output_tokens
        return main_total + reviewer_total
