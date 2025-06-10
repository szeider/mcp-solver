"""
Token counter for tracking token usage in LangGraph applications.

This module provides a class for tracking token usage during a client run,
separately for the main agent and reviewer agent.
"""

from typing import Any

from langchain_core.messages.utils import count_tokens_approximately
from rich.console import Console
from rich.table import Table


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
        self.reset()
        self.console = Console()

    def reset(self):
        """Reset all token counts."""
        self.main_input_tokens = 0
        self.main_output_tokens = 0
        self.reviewer_input_tokens = 0
        self.reviewer_output_tokens = 0
        self.main_is_exact = False
        self.reviewer_is_exact = False

    def count_main_input(self, messages):
        """Count input tokens for main agent (only if not exact)."""
        if not self.main_is_exact:
            self.main_input_tokens = count_tokens_approximately(messages)

    def count_main_output(self, messages):
        """Count output tokens for main agent (only if not exact)."""
        if not self.main_is_exact:
            self.main_output_tokens = count_tokens_approximately(messages)

    def count_reviewer_input(self, messages):
        """Count input tokens for reviewer (only if not exact)."""
        if not self.reviewer_is_exact:
            self.reviewer_input_tokens = count_tokens_approximately(messages)

    def count_reviewer_output(self, messages):
        """Count output tokens for reviewer (only if not exact)."""
        if not self.reviewer_is_exact:
            self.reviewer_output_tokens = count_tokens_approximately(messages)

    @property
    def total_main_tokens(self):
        """Get total tokens for main agent."""
        return self.main_input_tokens + self.main_output_tokens

    @property
    def total_reviewer_tokens(self):
        """Get total tokens for reviewer."""
        return self.reviewer_input_tokens + self.reviewer_output_tokens

    @property
    def total_tokens(self):
        """Get total tokens across all agents."""
        return self.total_main_tokens + self.total_reviewer_tokens

    def format_token_count(self, count: int) -> str:
        """Format token count with k/M suffixes."""
        if count >= 1_000_000:
            return f"{count / 1_000_000:.1f}M"
        elif count >= 1_000:
            return f"{count / 1_000:.1f}k"
        return str(count)

    def get_stats_table(self):
        """Get token usage statistics as a Rich table."""
        table = Table(title="Token Usage Statistics")
        table.add_column("Agent", style="cyan", no_wrap=True)
        table.add_column("Input", justify="right", style="green")
        table.add_column("Output", justify="right", style="blue")
        table.add_column("Total", justify="right", style="magenta")
        table.add_column("Type", justify="center", style="yellow")

        # Main agent row
        table.add_row(
            "ReAct Agent",
            self.format_token_count(self.main_input_tokens),
            self.format_token_count(self.main_output_tokens),
            self.format_token_count(self.total_main_tokens),
            "Exact" if self.main_is_exact else "Approx",
        )

        # Reviewer row
        table.add_row(
            "Reviewer",
            self.format_token_count(self.reviewer_input_tokens),
            self.format_token_count(self.reviewer_output_tokens),
            self.format_token_count(self.total_reviewer_tokens),
            "Exact" if self.reviewer_is_exact else "Approx",
        )

        # Combined row
        table.add_row(
            "COMBINED",
            self.format_token_count(
                self.main_input_tokens + self.reviewer_input_tokens
            ),
            self.format_token_count(
                self.main_output_tokens + self.reviewer_output_tokens
            ),
            self.format_token_count(self.total_tokens),
            "Mixed"
            if (self.main_is_exact or self.reviewer_is_exact)
            and not (self.main_is_exact and self.reviewer_is_exact)
            else ("Exact" if self.main_is_exact else "Approx"),
            style="bold",
        )

        return table

    def print_stats(self):
        """Print token usage statistics."""
        table = self.get_stats_table()
        self.console.print("\n")
        self.console.print(table)

    def display_token_usage(self):
        """Display token usage in a table."""
        table = self.get_stats_table()
        self.console.print("\n")
        self.console.print(table)
