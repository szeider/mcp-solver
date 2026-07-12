"""Unit tests for system-prompt assembly (files.system support)."""

from mcp_minion.agent import SYSTEM_PROMPT_BASE, build_system_prompt

TOOL_SECTIONS = "You have access to the following tools:\n\n## srv\n- `t`: do\n"


class TestBuildSystemPrompt:
    """Tests for build_system_prompt."""

    def test_default_base_template(self) -> None:
        """With no custom template, the base template is filled in."""
        result = build_system_prompt(None, TOOL_SECTIONS)
        assert TOOL_SECTIONS in result
        assert "helpful assistant" in result
        # {tool_sections} placeholder must be consumed.
        assert "{tool_sections}" not in result

    def test_default_base_no_tools(self) -> None:
        result = build_system_prompt(None, "")
        assert "{tool_sections}" not in result

    def test_custom_with_placeholder(self) -> None:
        """A custom template with the placeholder gets the tool list spliced in."""
        template = "SOLVER RULES\n\n{tool_sections}\nEnd."
        result = build_system_prompt(template, TOOL_SECTIONS)
        assert result == f"SOLVER RULES\n\n{TOOL_SECTIONS}\nEnd."
        assert "{tool_sections}" not in result

    def test_custom_with_placeholder_and_literal_braces(self) -> None:
        """Literal braces (e.g. ASP templates) must survive the splice."""
        template = (
            "ASP template:\n"
            "path(X,Y) :- edge(X,Y).\n"
            "reachable(X) :- start(X); { path(X,Y) : node(Y) }.\n"
            "{tool_sections}\n"
            "done"
        )
        result = build_system_prompt(template, TOOL_SECTIONS)
        # Braces from the ASP body are untouched.
        assert "{ path(X,Y) : node(Y) }" in result
        # Placeholder replaced with the tool sections.
        assert TOOL_SECTIONS in result
        assert "{tool_sections}" not in result

    def test_custom_without_placeholder_appends(self) -> None:
        """Without a placeholder, tool sections are appended after the file."""
        template = "SOLVER RULES with literal braces { a : b }"
        result = build_system_prompt(template, TOOL_SECTIONS)
        assert result == f"{template}\n\n{TOOL_SECTIONS}"
        # Literal braces preserved verbatim.
        assert "{ a : b }" in result

    def test_custom_without_placeholder_no_tools(self) -> None:
        """No placeholder and no tools returns the template unchanged."""
        template = "SOLVER RULES { a : b }"
        result = build_system_prompt(template, "")
        assert result == template

    def test_base_template_constant_has_placeholder(self) -> None:
        assert "{tool_sections}" in SYSTEM_PROMPT_BASE
