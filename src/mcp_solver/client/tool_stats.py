"""
Tool Usage Statistics Tracking Module.

This module provides a class for tracking tool usage statistics during a client run.
"""

from collections import Counter

from rich.console import Console
from rich.table import Table


class ToolStats:
    """Class for tracking tool usage statistics."""

    _instance = None

    @classmethod
    def get_instance(cls):
        """Get the singleton instance of ToolStats."""
        if cls._instance is None:
            cls._instance = ToolStats()
        return cls._instance

    def __init__(self):
        """Initialize the tool statistics tracker."""
        self.tool_calls = Counter()
        self.total_calls = 0
        self.console = Console()
        self.enabled = True

    def record_tool_call(self, tool_name: str):
        """Record a tool call."""
        if not self.enabled:
            return

        self.tool_calls[tool_name] += 1
        self.total_calls += 1

    def print_stats(self):
        """Print the tool usage statistics."""
        if not self.enabled:
            return

        if self.total_calls == 0:
            self.console.print("[yellow]No tools were called during this run.[/yellow]")
            return

        table = Table(title="Tool Usage Statistics")
        table.add_column("Tool Name", style="cyan")
        table.add_column("Call Count", style="green")
        table.add_column("Percentage", style="magenta")

        # Sort tools by number of calls (descending)
        sorted_tools = sorted(self.tool_calls.items(), key=lambda x: x[1], reverse=True)

        for tool_name, count in sorted_tools:
            percentage = f"{(count / self.total_calls) * 100:.1f}%"
            table.add_row(tool_name, str(count), percentage)

        # Add a total row
        table.add_row("TOTAL", str(self.total_calls), "100.0%", style="bold")

        self.console.print("\n")
        self.console.print(table)
