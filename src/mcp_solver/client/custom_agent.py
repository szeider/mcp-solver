"""
Custom ReAct Agent Implementation for MCP Solver

This module contains a true custom implementation of a ReAct agent built from scratch using LangGraph.
The implementation follows the canonical pattern for ReAct agents but is customized for
our specific use case.
"""

from typing import Annotated, Dict, List, Optional, Sequence, TypedDict, Any, Union, Literal, Callable
import sys
import asyncio
import json
import traceback
from copy import deepcopy

from langchain_core.messages import (
    AIMessage, 
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage
)
from langchain_core.tools import BaseTool, StructuredTool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

# Updated imports for langgraph 0.3.18
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


# Step 1: Define the agent state
class AgentState(TypedDict):
    """The state of the agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]


# Step 2: Define the model node function
def call_model(state: AgentState, model: BaseChatModel, system_prompt: Optional[str] = None) -> Dict:
    """Call the language model with the current conversation state.
    
    This function takes the current agent state, which includes the conversation history,
    and calls the language model to get a response. It may add a system prompt if provided.
    
    Args:
        state: The current agent state
        model: The language model to use
        system_prompt: Optional system prompt to add to the conversation
        
    Returns:
        Updated state with model response added
    """
    # Get the current messages
    messages = list(state["messages"])
    
    # Add system message at the beginning if provided and not already there
    if system_prompt and not any(isinstance(msg, SystemMessage) for msg in messages):
        messages.insert(0, SystemMessage(content=system_prompt))
    
    # Call the model with the messages
    response = model.invoke(messages)
    
    # Return updated state with model response added
    return {"messages": [response]}


# Step 3: Define the tools node function
def call_tools(state: AgentState, tools_by_name: Dict[str, BaseTool]) -> Dict:
    """Execute any tool calls in the latest AI message.
    
    Args:
        state: The current agent state
        tools_by_name: Dictionary mapping tool names to tools
        
    Returns:
        Updated state with tool messages added
    """
    messages = state["messages"]
    
    # Find the last AI message with tool calls
    tool_calls = []
    for message in reversed(messages):
        if not isinstance(message, AIMessage):
            continue
            
        # Look for tool_calls in the message
        if hasattr(message, "tool_calls") and message.tool_calls:
            tool_calls = message.tool_calls
            break
        
        # Also check additional message kwargs (for dict-style tool_calls)
        additional_kwargs = getattr(message, "additional_kwargs", {})
        if additional_kwargs and "tool_calls" in additional_kwargs:
            tool_calls = additional_kwargs["tool_calls"]
            break
    
    # If no tool calls were found, return the original state
    if not tool_calls:
        return {"messages": []}
    
    # Get or create event loop for async tool execution
    try:
        # Try to get the current event loop
        loop = asyncio.get_running_loop()
        
        # If we're in an event loop, apply nest_asyncio if available
        try:
            import nest_asyncio
            nest_asyncio.apply()
        except ImportError:
            print("Warning: nest_asyncio not available. Nested async execution may not work properly.", file=sys.stderr)
            print("To fix this, install nest_asyncio: `pip install nest_asyncio`", file=sys.stderr)
    
    except RuntimeError:
        # No event loop exists, create a new one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    # Process each tool call
    tool_results = []
    for tool_call in tool_calls:
        tool_name = None
        tool_id = None
        tool_args = None
        
        # Enhanced format detection for different tool call formats
        if isinstance(tool_call, dict):
            # Dictionary format (common in MCP and many API responses)
            tool_name = tool_call.get("name") or tool_call.get("function", {}).get("name")
            tool_id = tool_call.get("id") or tool_call.get("index")
            
            # Handle different args formats
            if "args" in tool_call:
                tool_args = tool_call["args"]
            elif "function" in tool_call and "arguments" in tool_call["function"]:
                # Handle OpenAI-style format
                args_str = tool_call["function"]["arguments"]
                try:
                    if isinstance(args_str, str):
                        tool_args = json.loads(args_str)
                    else:
                        tool_args = args_str
                except json.JSONDecodeError:
                    tool_args = {"raw_arguments": args_str}
        else:
            # Object format
            tool_name = getattr(tool_call, "name", None)
            if tool_name is None and hasattr(tool_call, "function") and hasattr(tool_call.function, "name"):
                tool_name = tool_call.function.name
                
            tool_id = getattr(tool_call, "id", None)
            if tool_id is None:
                tool_id = getattr(tool_call, "index", str(len(tool_results)))
                
            # Get args, handling different formats
            tool_args = getattr(tool_call, "args", None)
            if tool_args is None and hasattr(tool_call, "function"):
                args_str = getattr(tool_call.function, "arguments", "{}")
                try:
                    if isinstance(args_str, str):
                        tool_args = json.loads(args_str)
                    else:
                        tool_args = args_str
                except json.JSONDecodeError:
                    tool_args = {"raw_arguments": args_str}
        
        # Skip if tool name is missing or not in our tools
        if not tool_name:
            continue
            
        # Log tool execution for debugging
        print(f"Executing tool: {tool_name} with args: {tool_args}", file=sys.stderr)
        
        # Handle when tool isn't found but still process the request
        if tool_name not in tools_by_name:
            error_msg = f"Tool '{tool_name}' not found in available tools"
            print(f"Warning: {error_msg}", file=sys.stderr)
            tool_message = ToolMessage(
                content=f"Error: {error_msg}",
                tool_call_id=tool_id or f"unknown_tool_{len(tool_results)}",
                name=tool_name
            )
            tool_results.append(tool_message)
            continue
        
        # Get the tool
        tool = tools_by_name[tool_name]
        
        try:
            # Prepare the args for both sync and async invocation
            if tool_args is None:
                tool_args = {}
                
            # Prepare call_args for both formats tools might expect
            call_args = tool_args
            wrapped_args = {"args": tool_args}
            
            # Helper function to execute the tool with appropriate invocation method
            async def execute_tool():
                # Check if tool supports async invocation
                if hasattr(tool, "ainvoke") and callable(tool.ainvoke):
                    try:
                        # Try direct arg format first
                        return await tool.ainvoke(call_args)
                    except Exception as direct_error:
                        try:
                            # Try wrapped args format
                            return await tool.ainvoke(wrapped_args)
                        except Exception:
                            # If both fail, raise the original error
                            raise direct_error
                
                # If the tool doesn't support async invocation, raise an error
                raise ValueError(f"Tool {tool_name} doesn't support async invocation (ainvoke method)")
            
            # Execute the tool using the appropriate event loop
            try:
                if loop.is_running():
                    # If loop is already running, we need to use a different approach
                    # Create a future in the existing loop
                    future = asyncio.run_coroutine_threadsafe(execute_tool(), loop)
                    result = future.result(timeout=60)  # 60 second timeout
                else:
                    # If we're not in a running loop, run_until_complete works
                    result = loop.run_until_complete(execute_tool())
                
                # Format result for uniformity
                result_str = str(result) if result is not None else "Task completed successfully."
                
                # Create a tool message with the result
                tool_message = ToolMessage(
                    content=result_str,
                    tool_call_id=tool_id or f"tool_call_{len(tool_results)}",
                    name=tool_name
                )
                
                # Add the tool message to the results
                tool_results.append(tool_message)
                
            except Exception as e:
                # In case of error, create a tool message with detailed error info
                error_msg = f"Error executing tool {tool_name}: {str(e)}"
                print(f"Error: {error_msg}", file=sys.stderr)
                
                # Include traceback for better debugging
                tb_str = traceback.format_exc()
                detailed_error = f"{error_msg}\n\nTraceback:\n{tb_str}"
                print(detailed_error, file=sys.stderr)
                
                tool_message = ToolMessage(
                    content=error_msg,
                    tool_call_id=tool_id or f"error_{len(tool_results)}",
                    name=tool_name
                )
                tool_results.append(tool_message)
        
        except Exception as e:
            # In case of error, create a tool message with detailed error info
            error_msg = f"Error executing tool {tool_name}: {str(e)}"
            print(f"Error: {error_msg}", file=sys.stderr)
            
            # Include traceback for better debugging
            tb_str = traceback.format_exc()
            detailed_error = f"{error_msg}\n\nTraceback:\n{tb_str}"
            print(detailed_error, file=sys.stderr)
            
            tool_message = ToolMessage(
                content=error_msg,
                tool_call_id=tool_id or f"error_{len(tool_results)}",
                name=tool_name
            )
            tool_results.append(tool_message)
    
    # Return updated state with tool messages added
    return {"messages": tool_results}


# Step 4: Define the routing logic
def router(state: AgentState) -> Union[Literal["call_model", "call_tools"], Literal[END]]:
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
        return END
    
    last_message = messages[-1]
    
    # If the last message is a tool message or human message, we should call the model
    if isinstance(last_message, (ToolMessage, HumanMessage)):
        return "call_model"
    
    # If the last message is an AI message with tool calls, we should execute the tools
    if isinstance(last_message, AIMessage):
        # Check for tool_calls attribute
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "call_tools"
            
        # Also check additional_kwargs for tool_calls
        additional_kwargs = getattr(last_message, "additional_kwargs", {})
        if additional_kwargs and "tool_calls" in additional_kwargs:
            tool_calls = additional_kwargs["tool_calls"]
            if tool_calls and len(tool_calls) > 0:
                return "call_tools"
    
    # If no conditions match or AI message doesn't have tool calls, we're done
    return END


# Step 5: Define the complete graph
def create_custom_react_agent(
    llm: BaseChatModel,
    tools: List[BaseTool],
    system_prompt: Optional[str] = None,
):
    """
    Create a custom ReAct agent from scratch using LangGraph.
    
    This function creates an async agent but wraps it with a synchronous
    interface for compatibility with existing code.
    
    Args:
        llm: The language model to use
        tools: List of tools to provide to the agent
        system_prompt: Optional system prompt
        
    Returns:
        A compiled agent with a synchronous interface
    """
    # Create a tools by name dictionary for fast lookup
    tools_by_name = {tool.name: tool for tool in tools}
    
    # Prepare the LLM with the tools
    llm_with_tools = llm.bind_tools(tools)
    
    # Create a graph with the agent state
    workflow = StateGraph(AgentState)
    
    # Add nodes to the graph
    workflow.add_node("call_model", lambda state: call_model(state, llm_with_tools, system_prompt))
    
    # Attempt to use ToolNode with our tools
    try:
        tool_node = ToolNode(tools)
        workflow.add_node("call_tools", tool_node)
    except Exception as e:
        print(f"Warning: Error creating ToolNode with built-in tools: {str(e)}")
        print("Falling back to custom call_tools implementation")
        workflow.add_node("call_tools", lambda state: call_tools(state, tools_by_name))
    
    # Add conditional edges with our router
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
    
    # Set the entry point - always start with the model
    workflow.set_entry_point("call_model")
    
    # Compile the graph
    async_agent = workflow.compile()
    
    # Wrap the async agent with a synchronous interface
    return SyncCompatWrapper(async_agent)


# Function to run the agent on a human input
async def run_agent(agent, message: str, config: Optional[RunnableConfig] = None):
    """
    Async function for running an agent on a human input message.
    
    Args:
        agent: The agent to run
        message: The human input message
        config: Optional configuration
        
    Returns:
        The final state of the agent
    """
    # If it's our wrapped agent, unwrap it first to use the async version
    if isinstance(agent, SyncCompatWrapper):
        async_agent = agent.async_agent
    else:
        async_agent = agent
        
    # Create the initial state
    initial_state = {
        "messages": [HumanMessage(content=message)]
    }
    
    # Run the agent asynchronously
    final_state = await async_agent.ainvoke(initial_state, config)
    
    # Normalize the state to ensure consistent format
    return normalize_state(final_state)


def normalize_state(state) -> Dict:
    """
    Normalize state to a consistent dictionary format regardless of LangGraph version.
    
    This utility handles the differences in state formats between LangGraph versions.
    In newer versions, state might be an object with attributes, while in older versions
    it's a simple dictionary.
    
    Args:
        state: The state to normalize (could be dict, object, or other format)
        
    Returns:
        A normalized dictionary representation of the state
    """
    # Check if it's already a dict with the expected structure
    if isinstance(state, dict) and "messages" in state:
        return state
    
    # Handle object-style state (newer LangGraph versions)
    if hasattr(state, "messages"):
        return {"messages": state.messages}
    
    # Handle nested values dict pattern
    if hasattr(state, "values") and isinstance(state.values, dict):
        if "messages" in state.values:
            return {"messages": state.values["messages"]}
    
    # Handle the case where state itself might be the messages
    if isinstance(state, list) and all(isinstance(msg, BaseMessage) for msg in state):
        return {"messages": state}
    
    # Handle the ToolNode output style from langgraph 0.3.18+
    if isinstance(state, dict) and len(state) > 0:
        # Look for any node output that contains messages
        for node_name, node_output in state.items():
            if isinstance(node_output, dict) and "messages" in node_output:
                return {"messages": node_output["messages"]}
            elif isinstance(node_output, list) and all(isinstance(msg, BaseMessage) for msg in node_output):
                return {"messages": node_output}
    
    # Fallback: If we can't determine the format, return an empty dict with messages
    print(f"Warning: Unrecognized state format: {type(state)}", file=sys.stderr)
    
    # Try to extract any messages if possible, otherwise return empty list
    messages = []
    if isinstance(state, dict):
        # Look for any key that might contain messages
        for key, value in state.items():
            if isinstance(value, list) and value and isinstance(value[0], BaseMessage):
                messages = value
                break
    
    return {"messages": messages}


# Test function for the agent state
def test_agent_state():
    """Test the agent state structure."""
    print("Testing agent state...")
    
    # Create an initial state with a human message
    initial_state = {
        "messages": [HumanMessage(content="Hello, agent!")]
    }
    
    # Verify the state structure
    assert "messages" in initial_state
    assert len(initial_state["messages"]) == 1
    assert isinstance(initial_state["messages"][0], HumanMessage)
    
    # Test adding a message using the normal dictionary operation
    # Note: This doesn't use the add_messages reducer directly, just testing basic structure
    state_copy = initial_state.copy()
    messages_copy = list(state_copy["messages"])
    messages_copy.append(AIMessage(content="Hello, human!"))
    state_copy["messages"] = messages_copy
    
    assert len(state_copy["messages"]) == 2
    assert isinstance(state_copy["messages"][1], AIMessage)
    
    print("✅ Agent state tests passed!")
    return initial_state


# Test function for the model node
async def test_call_model():
    """Test the call_model function with a mock LLM."""
    print("Testing call_model function...")
    
    # Create a mock LLM
    class MockLLM(BaseChatModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            # This method is required but not directly used in our test
            pass
        
        def invoke(self, messages, **kwargs):
            # Direct implementation of invoke for testing
            return AIMessage(content="This is a mock response.")
            
        async def ainvoke(self, messages, **kwargs):
            # Async implementation
            return AIMessage(content="This is a mock async response.")
        
        @property
        def _llm_type(self):
            return "mock"
            
        def bind_tools(self, tools):
            # Just return self for testing
            return self
    
    # Create mock LLM instance
    mock_llm = MockLLM()
    
    # Create initial state
    initial_state = {
        "messages": [HumanMessage(content="Hello, agent!")]
    }
    
    # Test call_model
    result = await call_model(initial_state, mock_llm)
    assert "messages" in result
    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], AIMessage)
    assert result["messages"][0].content == "This is a mock async response."
    
    print("✅ call_model tests passed!")
    return result


# Test function for the tools node
async def test_call_tools():
    """Test the call_tools function with mock tools."""
    print("Testing call_tools function...")
    
    # Create a mock tool
    from langchain_core.tools import tool
    
    @tool
    def mock_calculator(operation: str, x: int, y: int) -> int:
        """A mock calculator tool."""
        if operation == "add":
            return x + y
        elif operation == "multiply":
            return x * y
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    # Create a tool that always fails
    @tool
    def failing_tool(input: str) -> str:
        """A tool that always fails."""
        raise Exception("This tool always fails")
    
    # Create a dictionary of tools
    tools = [mock_calculator, failing_tool]
    tools_by_name = {tool.name: tool for tool in tools}
    
    # Create a message with a tool call
    tool_calls = [
        {
            "id": "call_1",
            "name": "mock_calculator",
            "args": {"operation": "add", "x": 5, "y": 3}
        }
    ]
    message_with_tool_call = AIMessage(content="I'll calculate that for you.", tool_calls=tool_calls)
    
    # Create state with the message
    state = {
        "messages": [
            HumanMessage(content="What is 5 + 3?"),
            message_with_tool_call
        ]
    }
    
    # Test call_tools
    result = await call_tools(state, tools_by_name)
    assert "messages" in result
    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], ToolMessage)
    assert result["messages"][0].content == "8"
    assert result["messages"][0].name == "mock_calculator"
    
    # Test with multiple tool calls including one that fails
    tool_calls.append({
        "id": "call_2",
        "name": "failing_tool",
        "args": {"input": "test"}
    })
    message_with_multiple_calls = AIMessage(content="I'll try multiple tools.", tool_calls=tool_calls)
    
    state_multiple = {
        "messages": [
            HumanMessage(content="Try multiple tools"),
            message_with_multiple_calls
        ]
    }
    
    result_multiple = await call_tools(state_multiple, tools_by_name)
    assert len(result_multiple["messages"]) == 2
    assert "mock_calculator" in result_multiple["messages"][0].name
    assert "failing_tool" in result_multiple["messages"][1].name
    assert "Error" in result_multiple["messages"][1].content
    
    # Test with no tool calls
    state_no_tools = {
        "messages": [
            HumanMessage(content="Hello!"),
            AIMessage(content="Hi there!")
        ]
    }
    
    result_no_tools = await call_tools(state_no_tools, tools_by_name)
    assert len(result_no_tools["messages"]) == 0  # Fixed to match our new implementation
    
    print("✅ call_tools tests passed!")
    return result


# Test function for routing logic
def test_router():
    """Test the router function for graph routing."""
    print("Testing router function...")
    
    # Test with human message (should route to model)
    state_human = {
        "messages": [HumanMessage(content="Hello, agent!")]
    }
    assert router(state_human) == "call_model"
    
    # Test with tool message (should route to model)
    state_tool = {
        "messages": [
            HumanMessage(content="What's 5+3?"),
            AIMessage(content="I'll calculate that."),
            ToolMessage(content="8", name="calculator", tool_call_id="call1")
        ]
    }
    assert router(state_tool) == "call_model"
    
    # Test with AI message with tool calls (dict-style, should route to tools)
    state_ai_tools_dict = {
        "messages": [
            HumanMessage(content="What's 5+3?"),
            AIMessage(content="I'll calculate that.", tool_calls=[
                {"name": "calculator", "id": "call1", "args": {"x": 5, "y": 3}}
            ])
        ]
    }
    assert router(state_ai_tools_dict) == "call_tools"
    
    # Create a message with attached custom object-style tool calls
    message = AIMessage(content="I'll calculate that.")
    
    # Create a simple class with the necessary attributes
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
    
    # Manually set tool_calls attribute to avoid compatibility issues
    setattr(message, "tool_calls", [tool_call])
    
    assert router(state_ai_tools_obj) == "call_tools"
    
    # Test with AI message without tool calls (should end)
    state_ai_no_tools = {
        "messages": [
            HumanMessage(content="Hello!"),
            AIMessage(content="Hi there!")
        ]
    }
    assert router(state_ai_no_tools) == END
    
    # Test with empty messages (should end)
    state_empty = {
        "messages": []
    }
    assert router(state_empty) == END
    
    print("✅ router tests passed!")
    return True


# Test the full agent (basic functionality)
async def test_custom_agent():
    """Test the create_custom_react_agent function with mock tools and LLM."""
    print("Testing full custom agent...")
    
    # Create a mock LLM that simulates tool use
    class MockToolUsingLLM(BaseChatModel):
        def __init__(self):
            super().__init__()
            # Store counter as a separate attribute not managed by Pydantic
            self._counter = 0
        
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            # Required by BaseChatModel but not used in our test
            pass
        
        def invoke(self, messages, **kwargs):
            # Simple response pattern for testing
            return self._get_response()
            
        async def ainvoke(self, messages, **kwargs):
            # Async version
            return self._get_response()
            
        def _get_response(self):
            # Helper to get the appropriate response based on counter
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
    
    # Create mock tools
    from langchain_core.tools import tool
    
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
    final_state = await run_agent(agent, "What is 5 + 3?")
    
    # Print for debugging
    print("Final state message types:")
    for i, msg in enumerate(final_state["messages"]):
        print(f"  {i}: {type(msg).__name__} - {msg.content[:30]}...")
    
    # Verify the state has messages
    assert "messages" in final_state
    
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
    
    # There should be a tool message with the calculation result
    assert len(tool_messages) >= 1
    assert tool_messages[0].content == "8"
    assert tool_messages[0].name == "mock_calculator"
    
    print("✅ Full agent test passed!")
    return final_state


# Test the agent streaming with intermediate steps
@tool
def get_weather(location: str) -> str:
    """Get weather for a location."""
    return f"It's sunny in {location}."


# Test the agent streaming with intermediate steps
async def test_agent_intermediate_steps(
    llm: Optional[BaseChatModel] = None,
    tools: Optional[List] = None,
    query: Optional[str] = None,
    tool_executor: Optional[Callable] = None
):
    """
    Test the agent's intermediate steps during execution.
    
    Args:
        llm: A language model to use (if None, a mock LLM is created)
        tools: List of tools to use (if None, mock tools are created)
        query: The query to process (if None, a default query is used)
        tool_executor: A function to execute tools (if None, default execution is used)
        
    Returns:
        Tuple of (list of intermediate states, final state)
    """
    print("Testing agent intermediate steps...")
    
    # Use provided components or create mocks if not provided
    if llm is None or tools is None:
        # Create a mock LLM that simulates a more complex interaction with multiple turns
        class MockMultiTurnLLM(BaseChatModel):
            def __init__(self):
                super().__init__()
                self._counter = 0
            
            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                pass
            
            def invoke(self, messages, **kwargs):
                return self._get_response(messages)
                
            async def ainvoke(self, messages, **kwargs):
                return self._get_response(messages)
                
            def _get_response(self, messages):
                # Track non-system messages for more accurate response generation
                non_system_msgs = [m for m in messages if not isinstance(m, SystemMessage)]
                
                # Check where we are in the conversation to generate appropriate response
                human_content = non_system_msgs[0].content if non_system_msgs else ""
                
                # First turn: Ask for conversion from Celsius to Fahrenheit
                if "convert" in human_content.lower() and "celsius" in human_content.lower():
                    return AIMessage(
                        content="I'll convert Celsius to Fahrenheit for you.",
                        tool_calls=[{
                            "name": "temperature_converter",
                            "id": "temp_call_1",
                            "args": {"celsius": 25}
                        }]
                    )
                
                # After temperature conversion, check if they want the weather
                elif any(isinstance(m, ToolMessage) and m.name == "temperature_converter" for m in non_system_msgs):
                    return AIMessage(
                        content="The temperature 25°C is 77°F. Would you like to know the weather in a specific location?",
                        tool_calls=[{
                            "name": "get_weather",
                            "id": "weather_call_1",
                            "args": {"location": "San Francisco"}
                        }]
                    )
                
                # Final response after getting weather
                elif any(isinstance(m, ToolMessage) and m.name == "get_weather" for m in non_system_msgs):
                    return AIMessage(content="The weather in San Francisco is sunny with a temperature of 77°F (25°C). Anything else you'd like to know?")
                
                # Default response
                else:
                    return AIMessage(content="I'm not sure how to respond to that.")
            
            @property
            def _llm_type(self):
                return "mock_multi_turn"
                
            def bind_tools(self, tools):
                return self
        
        # Create mock tools if not provided
        mock_llm = MockMultiTurnLLM()
        
        @tool
        def temperature_converter(celsius: float) -> str:
            """Convert Celsius to Fahrenheit."""
            fahrenheit = (celsius * 9/5) + 32
            return f"{celsius}°C is equal to {fahrenheit}°F"
        
        @tool
        def get_weather(location: str) -> str:
            """Get weather for a location."""
            return f"It's sunny in {location}."
            
        # Use the mocks we created
        llm = mock_llm
        tools = [temperature_converter, get_weather]
        query = "Can you convert 25 degrees Celsius to Fahrenheit?"
    else:
        # Make sure tools are in the correct format
        if tools and isinstance(tools[0], dict):
            # Convert tool dictionaries to BaseTool objects if needed
            tool_objects = []
            for tool_dict in tools:
                @tool
                def dynamic_tool(**kwargs):
                    """Dynamically created tool."""
                    if tool_executor:
                        # Use the provided tool executor
                        return tool_executor(tool_dict["name"], kwargs)
                    return f"Tool {tool_dict['name']} called with args {kwargs}"
                
                # Set the tool's name and description
                dynamic_tool.name = tool_dict["name"]
                dynamic_tool.__doc__ = tool_dict.get("description", "A tool.")
                tool_objects.append(dynamic_tool)
            
            tools = tool_objects
    
    # Use the provided query or default
    query = query or "Can you convert 25 degrees Celsius to Fahrenheit?"
    
    # Create and prepare the agent
    system_prompt = "You are a helpful assistant."
    
    # Set up the tool executor function if provided
    if tool_executor:
        # Create a wrapper for the BaseTool objects to use our custom executor
        original_tools = tools
        tools = []
        
        for orig_tool in original_tools:
            @tool
            async def wrapped_tool(**kwargs):
                """Wrapped tool that uses the custom executor."""
                return await tool_executor(orig_tool.name, kwargs)
            
            # Copy name and description
            wrapped_tool.name = orig_tool.name
            wrapped_tool.__doc__ = orig_tool.__doc__
            tools.append(wrapped_tool)
    
    # Create the agent directly with the StateGraph to access streaming
    tools_by_name = {tool.name: tool for tool in tools}
    llm_with_tools = llm.bind_tools(tools)
    
    # Build the graph
    workflow = StateGraph(AgentState)
    workflow.add_node("call_model", lambda state: call_model(state, llm_with_tools, system_prompt))
    
    # Attempt to use ToolNode with our tools
    try:
        tool_node = ToolNode(tools)
        workflow.add_node("call_tools", tool_node)
    except Exception as e:
        print(f"Warning: Error creating ToolNode with built-in tools: {str(e)}")
        print("Falling back to custom call_tools implementation")
        workflow.add_node("call_tools", lambda state: call_tools(state, tools_by_name))
    
    workflow.add_conditional_edges(
        "call_model", router,
        {"call_tools": "call_tools", END: END}
    )
    
    workflow.add_conditional_edges(
        "call_tools", router,
        {"call_model": "call_model", END: END}
    )
    
    workflow.set_entry_point("call_model")
    
    # Compile the graph
    agent = workflow.compile()
    
    # Create initial state
    initial_state = {
        "messages": [HumanMessage(content=query)]
    }
    
    # Configure the run with a higher recursion limit
    config = {"recursion_limit": 10}
    
    # Capture all intermediate states
    states = []
    final_state = None
    
    try:
        async for state in agent.astream(initial_state, config=config):
            # Use our normalize_state function to handle different formats consistently
            normalized_state = normalize_state(state)
            states.append(normalized_state)
            
            # Update the final state
            final_state = normalized_state
            
            # Print the current state details for debugging
            messages = normalized_state.get("messages", [])
            if messages:
                last_msg = messages[-1]
                msg_type = type(last_msg).__name__
                content = getattr(last_msg, "content", "")
                msg_content = content[:50] + "..." if content and len(content) > 50 else content
                print(f"Intermediate state: {msg_type} - {msg_content}")
            else:
                print("Intermediate state: No messages")
    except Exception as e:
        print(f"Error during agent streaming: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Only verify test conditions when using our mock setup
    if llm.__class__.__name__ == "MockMultiTurnLLM" and isinstance(tools[0], BaseTool):
        # Extract messages from all states for analysis
        all_messages = []
        for state in states:
            all_messages.extend(state.get("messages", []))
        
        # Verify we have messages of different types
        message_types = [type(msg).__name__ for msg in all_messages]
        print("Message types in all states:", set(message_types))
        
        # Check if we have the expected types of messages
        assert "HumanMessage" in message_types, "Missing HumanMessage in states"
        assert "AIMessage" in message_types, "Missing AIMessage in states"
        
        print("✅ Agent intermediate steps test passed!")
    
    return states, final_state


# Update the run_tests function to include our new test
async def run_tests():
    """Run all async tests."""
    test_agent_state()
    await test_call_model()
    await test_call_tools()
    test_router()
    await test_custom_agent()
    await test_agent_intermediate_steps()


# Add a synchronous wrapper for client.py compatibility
class SyncCompatWrapper:
    """
    A wrapper class that provides a synchronous interface to our async agent.
    
    This allows the async agent to be used with code that expects a synchronous
    interface, like the client.py file. This implementation properly handles
    event loops to avoid conflicts.
    """
    def __init__(self, async_agent):
        """Initialize with an async agent."""
        self.async_agent = async_agent
    
    def invoke(self, inputs, config=None):
        """
        Provides a synchronous interface to the async agent that avoids event loop issues.
        
        Args:
            inputs: The inputs to pass to the agent
            config: Optional configuration
            
        Returns:
            The final state of the agent
        """
        # Check if we're already inside a running event loop
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                # We're in an event loop - run the task in a way that doesn't block
                # Create a new Future in this loop and run the coroutine in it
                future = asyncio.run_coroutine_threadsafe(
                    self.async_agent.ainvoke(inputs, config), loop
                )
                return future.result(timeout=600)  # 10-minute timeout for safety
        except RuntimeError:
            # No running event loop - we'll create one
            pass
        
        # If we're not in an event loop or future creation failed, use asyncio.run()
        return asyncio.run(self.async_agent.ainvoke(inputs, config))
            
    def stream(self, inputs, config=None):
        """
        Provides a synchronous streaming interface to the async agent.
        
        Args:
            inputs: The inputs to pass to the agent
            config: Optional configuration
            
        Returns:
            A generator yielding states from the agent
        """
        # Try to get the current event loop
        try:
            loop = asyncio.get_running_loop()
            in_event_loop = loop.is_running()
        except RuntimeError:
            in_event_loop = False
        
        if in_event_loop:
            # We're in a running event loop - use a threading approach
            async def astream_wrapper():
                async for item in self.async_agent.astream(inputs, config):
                    yield item
            
            # Create an async generator
            agen = astream_wrapper()
            
            # Yield items from the async generator using run_coroutine_threadsafe
            while True:
                try:
                    # Schedule anext() in the event loop
                    future = asyncio.run_coroutine_threadsafe(
                        agen.__anext__(), loop
                    )
                    # Wait for the result with a timeout
                    yield future.result(timeout=60)
                except StopAsyncIteration:
                    break
                except Exception as e:
                    print(f"Error in stream: {e}")
                    break
        else:
            # No running event loop - we can use a simpler approach
            # Get the async generator
            agen = self.async_agent.astream(inputs, config)
            
            def sync_generator():
                # Create a new event loop
                loop = asyncio.new_event_loop()
                try:
                    while True:
                        try:
                            # Run anext() to get the next item
                            next_item = loop.run_until_complete(agen.__anext__())
                            yield next_item
                        except StopAsyncIteration:
                            break
                finally:
                    loop.close()
            
            # Return a sync generator
            yield from sync_generator()


# Utility function to help diagnose MCP tool issues
def debug_mcp_tools(name, args, result):
    """Print debug information for MCP tool calls."""
    import sys
    print(f"MCP Tool Debug - {name}:", file=sys.stderr)
    print(f"  Input: {json.dumps(args, indent=2)}", file=sys.stderr)
    print(f"  Result: {result}", file=sys.stderr)
    
    # Special checks for MCP tools
    if name == "add_item" and result and "Model is empty" in str(result):
        print(f"  WARNING: Item added but model still showing as empty!", file=sys.stderr)
        print(f"  Content was: {args.get('content', '')[:50]}...", file=sys.stderr)
    
    if name == "get_model" and result and "Model is empty" in str(result):
        print(f"  WARNING: Model appears to be empty after adding items!", file=sys.stderr)
    
    return result


# Wrapper to add debug logs to tools
def create_debug_wrapper(tool_name, orig_func):
    """Create a debug wrapper for a tool function"""
    async def wrapped_func(args, **kwargs):
        result = await orig_func(args, **kwargs)
        return debug_mcp_tools(tool_name, args, result)
    return wrapped_func


if __name__ == "__main__":
    # Run the test functions
    import asyncio
    asyncio.run(run_tests()) 