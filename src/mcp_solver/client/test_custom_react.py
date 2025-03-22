"""
Test for custom React agent implementation.

This module tests the custom React agent without loading other dependencies.
"""

from typing import Annotated, Dict, List, Optional, Sequence, TypedDict, Any, Union, Literal

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage
)
from langchain_core.tools import BaseTool, tool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langchain_core.output_parsers.openai_tools import PydanticToolsParser

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages


# Agent state
class AgentState(TypedDict):
    """The state of our custom ReAct agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]


# Model call function
def call_model(state: AgentState, model: BaseChatModel, system_prompt: Optional[str] = None) -> Dict:
    """Call the language model with the current conversation state."""
    messages = list(state["messages"])
    
    if system_prompt and not any(isinstance(msg, SystemMessage) for msg in messages):
        messages.insert(0, SystemMessage(content=system_prompt))
    
    response = model.invoke(messages)
    
    return {"messages": messages + [response]}


# Tools call function
def call_tools(state: AgentState, tools_by_name: Dict[str, BaseTool]) -> Dict:
    """Execute tools based on the last message's tool calls."""
    messages = list(state["messages"])
    if not messages:
        return state
    
    last_message = messages[-1]
    
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return state
    
    tool_results = []
    for tool_call in last_message.tool_calls:
        tool_name = None
        tool_id = None
        tool_args = None
        
        if isinstance(tool_call, dict):
            # Dictionary format
            tool_name = tool_call.get("name")
            tool_id = tool_call.get("id")
            tool_args = tool_call.get("args")
        else:
            # Object format
            tool_name = getattr(tool_call, "name", None)
            tool_id = getattr(tool_call, "id", None)
            tool_args = getattr(tool_call, "args", None)
        
        if not tool_name or tool_name not in tools_by_name:
            continue
        
        tool = tools_by_name[tool_name]
        
        try:
            result = tool.invoke(tool_args)
            
            tool_message = ToolMessage(
                content=str(result),
                tool_call_id=tool_id,
                name=tool_name
            )
            
            tool_results.append(tool_message)
        except Exception as e:
            error_msg = f"Error executing tool {tool_name}: {str(e)}"
            tool_message = ToolMessage(
                content=error_msg,
                tool_call_id=tool_id,
                name=tool_name
            )
            tool_results.append(tool_message)
    
    return {"messages": tool_results}


# Routing logic
def router(state: AgentState) -> str:
    """Determine the next node in the graph based on the current state."""
    messages = state.get("messages", [])
    
    if not messages:
        return END
    
    last_message = messages[-1]
    
    if isinstance(last_message, (ToolMessage, HumanMessage)):
        return "call_model"
    
    if isinstance(last_message, AIMessage):
        has_tool_calls = False
        
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            for tool_call in last_message.tool_calls:
                if isinstance(tool_call, dict) and tool_call.get("name"):
                    has_tool_calls = True
                    break
                    
                if hasattr(tool_call, "name") and tool_call.name:
                    has_tool_calls = True
                    break
            
            if has_tool_calls:
                return "call_tools"
    
    return END


# Create the agent
def create_custom_react_agent(
    llm: BaseChatModel,
    tools: List[BaseTool],
    system_prompt: Optional[str] = None,
):
    """Create a custom ReAct agent from scratch using LangGraph."""
    tools_by_name = {tool.name: tool for tool in tools}
    
    llm_with_tools = llm.bind_tools(tools)
    
    # Define our graph
    workflow = StateGraph(AgentState)
    
    # Define the nodes for the model and tools
    workflow.add_node(
        "call_model", 
        lambda state: call_model(state, llm_with_tools, system_prompt)
    )
    
    workflow.add_node(
        "call_tools",
        lambda state: call_tools(state, tools_by_name)
    )
    
    # Set the entry point - always start with the model
    workflow.set_entry_point("call_model")
    
    # Add the router edges - this is the correct way to use conditional edges
    workflow.add_conditional_edges(
        "call_model",  # from node
        router,        # routing function
        {
            "call_tools": "call_tools",  # route condition -> target node
            END: END                     # end the graph
        }
    )
    
    workflow.add_conditional_edges(
        "call_tools",  # from node
        router,        # routing function
        {
            "call_model": "call_model",  # route condition -> target node
            END: END                     # end the graph
        }
    )
    
    # Compile the graph
    return workflow.compile()


# Function to run the agent
def run_agent(agent, message: str, config: Optional[RunnableConfig] = None):
    """Run the agent on a human input message."""
    initial_state = {
        "messages": [HumanMessage(content=message)]
    }
    
    return agent.invoke(initial_state, config)


# Create a custom mock LLM that doesn't rely on Pydantic fields
class MockToolUsingLLM(BaseChatModel):
    """Mock LLM that simulates tool use for testing."""
    
    def __init__(self):
        super().__init__()
        # Store counter as a separate attribute not managed by Pydantic
        self._counter = 0
    
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        # Required by BaseChatModel but not used in our test
        pass
    
    def invoke(self, messages, **kwargs):
        # Simple response pattern for testing:
        # 1. First response uses a tool
        # 2. Second response makes a normal reply
        self._counter += 1
        
        if self._counter == 1:
            # First call - use the calculator tool
            return AIMessage(
                content="I'll calculate 5 + 3 for you",
                tool_calls=[{
                    "name": "mock_calculator",
                    "id": "call_1",
                    "args": {"operation": "add", "x": 5, "y": 3}
                }]
            )
        else:
            # Second call - normal response
            return AIMessage(content="The result is 8. Is there anything else you need?")
    
    @property
    def _llm_type(self):
        return "mock_tool_using"
        
    def bind_tools(self, tools):
        # Just return self for testing
        return self


# Simple test function to make sure everything works
def test_custom_react():
    """Test the custom React agent implementation."""
    print("Testing custom React agent...")
    
    # Create mock calculator tool
    @tool
    def mock_calculator(operation: str, x: int, y: int) -> int:
        """A mock calculator tool."""
        if operation == "add":
            return x + y
        elif operation == "multiply":
            return x * y
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    # Create the agent
    mock_llm = MockToolUsingLLM()
    tools = [mock_calculator]
    system_prompt = "You are a helpful assistant."
    
    agent = create_custom_react_agent(mock_llm, tools, system_prompt)
    
    # Run the agent
    final_state = run_agent(agent, "What is 5 + 3?")
    
    # Print for debugging
    print("Final state message types:")
    for i, msg in enumerate(final_state["messages"]):
        print(f"  {i}: {type(msg).__name__} - {msg.content[:30]}...")
    
    # Verify the state
    assert "messages" in final_state
    
    # Check message count - we should have multiple messages
    # The exact count might vary but we need at least the human message, 
    # plus responses and tool results
    assert len(final_state["messages"]) >= 3
    
    # Find our key message types
    human_messages = [msg for msg in final_state["messages"] if isinstance(msg, HumanMessage)]
    ai_messages = [msg for msg in final_state["messages"] if isinstance(msg, AIMessage)]
    tool_messages = [msg for msg in final_state["messages"] if isinstance(msg, ToolMessage)]
    
    # There should be exactly one human message with our initial query
    assert len(human_messages) == 1
    assert human_messages[0].content == "What is 5 + 3?"
    
    # There should be at least one AI message 
    assert len(ai_messages) >= 1
    # First AI message should mention calculation
    assert "calculate" in ai_messages[0].content
    # Last AI message should mention the result
    assert "result is 8" in ai_messages[-1].content
    
    # There should be exactly one tool message with the calculation result
    assert len(tool_messages) == 1
    assert tool_messages[0].content == "8"
    assert tool_messages[0].name == "mock_calculator"
    
    print("✅ Custom React agent test passed!")
    return final_state


if __name__ == "__main__":
    test_custom_react() 