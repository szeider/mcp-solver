"""
Standalone test for custom agent components.

This module contains test functions that can be run independently 
without loading other dependencies.
"""

from typing import Literal

from langchain_core.messages import (
    AIMessage, 
    HumanMessage,
    ToolMessage
)


def should_continue(state) -> Literal["call_model", "call_tools", "end"]:
    """Determine the next node in the graph based on the current state.
    
    This function examines the current state and decides which node should be executed next:
    - If the last message is from a tool or human, it routes to the model node
    - If the last message is from the AI and contains tool calls, it routes to the tools node
    - Otherwise, it ends the execution
    
    Args:
        state: The current agent state
        
    Returns:
        The name of the next node to execute
    """
    messages = state.get("messages", [])
    
    # Safety check for empty messages
    if not messages:
        return "end"
    
    last_message = messages[-1]
    
    # If the last message is a tool message or human message, we should call the model
    if isinstance(last_message, (ToolMessage, HumanMessage)):
        return "call_model"
    
    # If the last message is an AI message with tool calls, we should execute the tools
    if isinstance(last_message, AIMessage):
        has_tool_calls = False
        
        # Check for tool calls
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            # Make sure at least one tool call is valid
            for tool_call in last_message.tool_calls:
                # For dict-style tool calls
                if isinstance(tool_call, dict) and tool_call.get("name"):
                    has_tool_calls = True
                    break
                    
                # For object-style tool calls
                if hasattr(tool_call, "name") and tool_call.name:
                    has_tool_calls = True
                    break
            
            if has_tool_calls:
                return "call_tools"
    
    # If no conditions match, we're done
    return "end"


def test_should_continue():
    """Test the should_continue function for graph routing."""
    print("Testing should_continue function...")
    
    # Test with human message (should route to model)
    state_human = {
        "messages": [HumanMessage(content="Hello, agent!")]
    }
    assert should_continue(state_human) == "call_model"
    
    # Test with tool message (should route to model)
    state_tool = {
        "messages": [
            HumanMessage(content="What's 5+3?"),
            AIMessage(content="I'll calculate that."),
            ToolMessage(content="8", name="calculator", tool_call_id="call1")
        ]
    }
    assert should_continue(state_tool) == "call_model"
    
    # Test with AI message with tool calls (dict-style, should route to tools)
    state_ai_tools_dict = {
        "messages": [
            HumanMessage(content="What's 5+3?"),
            AIMessage(content="I'll calculate that.", tool_calls=[
                {"name": "calculator", "id": "call1", "args": {"x": 5, "y": 3}}
            ])
        ]
    }
    assert should_continue(state_ai_tools_dict) == "call_tools"
    
    # Create a message with attached custom object-style tool calls
    # Instead of creating custom objects, we'll add tool_calls as an attribute to AIMessage
    message = AIMessage(content="I'll calculate that.")
    
    # Create a dict with an object instead of using MockToolCall directly
    class MockToolCall:
        def __init__(self, name, id, args):
            self.name = name
            self.id = id
            self.args = args
            
    tool_call = MockToolCall(name="calculator", id="call1", args={"x": 5, "y": 3})
    
    # Create a state with our own message and tool calls
    state_ai_tools_obj = {
        "messages": [
            HumanMessage(content="What's 5+3?"),
            message
        ]
    }
    
    # Manually set tool_calls attribute instead of passing it to constructor
    setattr(message, "tool_calls", [tool_call])
    
    assert should_continue(state_ai_tools_obj) == "call_tools"
    
    # Test with AI message without tool calls (should end)
    state_ai_no_tools = {
        "messages": [
            HumanMessage(content="Hello!"),
            AIMessage(content="Hi there!")
        ]
    }
    assert should_continue(state_ai_no_tools) == "end"
    
    # Test with empty messages (should end)
    state_empty = {
        "messages": []
    }
    assert should_continue(state_empty) == "end"
    
    print("✅ should_continue tests passed!")
    return True


if __name__ == "__main__":
    test_should_continue() 