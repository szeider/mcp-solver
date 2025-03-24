"""
ReAct Agent Implementation for MCP Solver

This module contains a ReAct (Reasoning+Acting) agent implementation built with LangGraph.
The implementation follows the canonical pattern for ReAct agents but is customized for
MCP Solver's specific use case.
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

# Updated imports for langgraph
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# Import tool stats tracking
try:
    from .tool_stats import ToolStats
except ImportError:
    # Fallback for when the module is imported directly
    ToolStats = None


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
    
    # If no tool calls were found, return empty messages
    if not tool_calls:
        return {"messages": []}
    
    # Set up async execution environment
    try:
        # Try to get the current event loop
        loop = asyncio.get_running_loop()
        
        # If we're in an event loop, apply nest_asyncio for nested async support
        try:
            import nest_asyncio
            nest_asyncio.apply()
        except ImportError:
            print("Warning: nest_asyncio not available. Nested async execution may not work properly.", file=sys.stderr)
    except RuntimeError:
        # No event loop exists, create a new one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    # Process each tool call
    tool_results = []
    for tool_call in tool_calls:
        # Extract tool information using a normalized approach
        tool_info = extract_tool_info(tool_call)
        
        if not tool_info["name"]:
            continue
            
        tool_name = tool_info["name"]
        tool_id = tool_info["id"] or f"tool_{len(tool_results)}"
        tool_args = tool_info["args"] or {}
            
        # Log tool execution for debugging
        print(f"Executing tool: {tool_name} with args: {json.dumps(tool_args, indent=2)}", file=sys.stderr)
        
        # Handle when tool isn't found
        if tool_name not in tools_by_name:
            error_msg = f"Tool '{tool_name}' not found in available tools"
            print(f"Warning: {error_msg}", file=sys.stderr)
            tool_message = ToolMessage(
                content=f"Error: {error_msg}",
                tool_call_id=tool_id,
                name=tool_name
            )
            tool_results.append(tool_message)
            continue
        
        # Get the tool and execute it
        tool = tools_by_name[tool_name]
        result = execute_tool_safely(tool, tool_name, tool_id, tool_args, loop)
        tool_results.append(result)
    
    # Return updated state with tool messages added
    return {"messages": tool_results}


def extract_tool_info(tool_call: Any) -> Dict[str, Any]:
    """Extract standardized tool information from different tool call formats.
    
    Args:
        tool_call: The tool call object or dictionary
        
    Returns:
        Dictionary with standardized name, id, and args
    """
    tool_name = None
    tool_id = None
    tool_args = None
    
    # Handle dictionary format
    if isinstance(tool_call, dict):
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
    # Handle object format
    else:
        tool_name = getattr(tool_call, "name", None)
        if tool_name is None and hasattr(tool_call, "function") and hasattr(tool_call.function, "name"):
            tool_name = tool_call.function.name
            
        tool_id = getattr(tool_call, "id", None)
        if tool_id is None:
            tool_id = getattr(tool_call, "index", None)
            
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
    
    return {
        "name": tool_name,
        "id": tool_id,
        "args": tool_args
    }


def execute_tool_safely(tool: BaseTool, tool_name: str, tool_id: str, tool_args: Dict[str, Any], loop: asyncio.AbstractEventLoop) -> ToolMessage:
    """Execute a tool safely, handling errors and timeouts.
    
    Args:
        tool: The tool to execute
        tool_name: The name of the tool
        tool_id: The ID of the tool call
        tool_args: The arguments to pass to the tool
        loop: The event loop to use for async execution
        
    Returns:
        A ToolMessage with the result or error
    """
    # Record tool usage statistics if ToolStats is available
    if ToolStats is not None:
        tool_stats = ToolStats.get_instance()
        tool_stats.record_tool_call(tool_name)
        
    # Prepare call arguments for both formats
    call_args = tool_args
    wrapped_args = {"args": tool_args}
    
    # Helper function to execute the tool with appropriate invocation method
    async def execute_tool_async():
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
        
        # Fall back to sync invoke if async is not available
        if hasattr(tool, "invoke") and callable(tool.invoke):
            try:
                # Try direct arg format first (sync in async)
                return tool.invoke(call_args)
            except Exception as direct_error:
                try:
                    # Try wrapped args format (sync in async)
                    return tool.invoke(wrapped_args)
                except Exception:
                    # If both fail, raise the original error
                    raise direct_error
                    
        # If no valid invocation method, raise error
        raise ValueError(f"Tool {tool_name} has no valid invocation method (ainvoke or invoke)")
    
    try:
        # Execute the tool with the appropriate method based on loop state
        if loop.is_running():
            # Loop is already running, use run_coroutine_threadsafe
            future = asyncio.run_coroutine_threadsafe(execute_tool_async(), loop)
            result = future.result(timeout=60)  # 60 second timeout
        else:
            # Not in a running loop, use run_until_complete
            result = loop.run_until_complete(execute_tool_async())
        
        # Format result for uniformity
        result_str = str(result) if result is not None else "Task completed successfully."
        
        # Create a tool message with the result
        return ToolMessage(
            content=result_str,
            tool_call_id=tool_id,
            name=tool_name
        )
        
    except Exception as e:
        # In case of error, create a tool message with error info
        error_msg = f"Error executing tool {tool_name}: {str(e)}"
        print(f"Error: {error_msg}", file=sys.stderr)
        
        # Include traceback for better debugging
        tb_str = traceback.format_exc()
        print(f"Traceback: {tb_str}", file=sys.stderr)
        
        return ToolMessage(
            content=error_msg,
            tool_call_id=tool_id,
            name=tool_name
        )


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
        if additional_kwargs and "tool_calls" in additional_kwargs and additional_kwargs["tool_calls"]:
            return "call_tools"
    
    # If no conditions match or AI message doesn't have tool calls, we're done
    return END


# Step 5: Define the complete graph
def create_react_agent(
    llm: BaseChatModel,
    tools: List[BaseTool],
    system_prompt: Optional[str] = None,
):
    """
    Create a ReAct agent using LangGraph.
    
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
    
    # Fallback: If we can't determine the format, log warning and return empty dict with messages
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


class SyncCompatWrapper:
    """
    A wrapper class that provides a synchronous interface to an async agent.
    
    This allows an async agent to be used with code that expects a synchronous
    interface. It properly handles event loops to avoid conflicts.
    """
    def __init__(self, async_agent):
        """Initialize with an async agent."""
        self.async_agent = async_agent
    
    def invoke(self, inputs, config=None):
        """
        Provides a synchronous interface to the async agent.
        
        Args:
            inputs: The inputs to pass to the agent
            config: Optional configuration
            
        Returns:
            The final state of the agent
        """
        try:
            # Check if we're already inside a running event loop
            loop = asyncio.get_running_loop()
            
            # We're in an event loop - use run_coroutine_threadsafe
            if loop.is_running():
                future = asyncio.run_coroutine_threadsafe(
                    self.async_agent.ainvoke(inputs, config), loop
                )
                return future.result(timeout=300)  # 5-minute timeout
                
        except RuntimeError:
            # No running event loop - use asyncio.run()
            return asyncio.run(self.async_agent.ainvoke(inputs, config))
            
        # Fallback if in loop but run_coroutine_threadsafe fails
        try:
            import nest_asyncio
            nest_asyncio.apply()
            return asyncio.run(self.async_agent.ainvoke(inputs, config))
        except Exception as e:
            print(f"Error in invoke: {e}", file=sys.stderr)
            raise
            
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
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        if in_event_loop:
            # We're in a running event loop
            async def astream_wrapper():
                async for item in self.async_agent.astream(inputs, config):
                    yield item
            
            # Create an async generator
            agen = astream_wrapper()
            
            # Yield items using run_coroutine_threadsafe
            try:
                import nest_asyncio
                nest_asyncio.apply()
                
                while True:
                    try:
                        future = asyncio.run_coroutine_threadsafe(
                            agen.__anext__(), loop
                        )
                        yield future.result(timeout=60)
                    except StopAsyncIteration:
                        break
                    except Exception as e:
                        print(f"Error in stream: {e}", file=sys.stderr)
                        break
            except ImportError:
                print("Warning: nest_asyncio not available. Streaming may not work properly.", file=sys.stderr)
                # Simplified fallback
                return asyncio.run(self._collect_stream_results(inputs, config))
        else:
            # No running event loop - use simpler approach
            for item in loop.run_until_complete(self._collect_stream_results(inputs, config)):
                yield item
    
    async def _collect_stream_results(self, inputs, config=None):
        """Helper to collect all stream results at once when streaming isn't possible."""
        results = []
        async for item in self.async_agent.astream(inputs, config):
            results.append(item)
        return results


def debug_mcp_tools(name, args, result):
    """Print debug information for MCP tool calls."""
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


def create_debug_wrapper(tool_name, orig_func):
    """Create a debug wrapper for a tool function."""
    async def wrapped_func(args, **kwargs):
        result = await orig_func(args, **kwargs)
        return debug_mcp_tools(tool_name, args, result)
    return wrapped_func


async def test_agent_intermediate_steps(
    llm: BaseChatModel,
    tools: List[Any],
    query: str,
    system_prompt: Optional[str] = None,
    tool_executor: Optional[Callable] = None,
):
    """
    Test function to capture intermediate steps during agent execution.
    
    This is primarily used for debugging and understanding the agent's behavior.
    
    Args:
        llm: The language model to use
        tools: List of tools to provide to the agent
        query: The query to process
        system_prompt: Optional system prompt
        tool_executor: Optional custom tool executor function
        
    Returns:
        Tuple of (intermediate_states, final_state)
    """
    # Convert tools to proper format if they're dictionaries
    processed_tools = []
    for tool in tools:
        if isinstance(tool, dict):
            # Create a simple tool from dict specification
            @tool(name=tool["name"], description=tool.get("description", ""))
            async def dynamic_tool(args: Dict[str, Any], tool_spec=tool):
                """Dynamic tool created from specification."""
                if tool_executor:
                    return await tool_executor(tool_spec["name"], args)
                return f"Called {tool_spec['name']} with {json.dumps(args)}"
                
            processed_tools.append(dynamic_tool)
        else:
            processed_tools.append(tool)
    
    # Create the agent
    agent = create_react_agent(llm, processed_tools, system_prompt)
    
    # Initialize state
    initial_state = {
        "messages": [HumanMessage(content=query)]
    }
    
    # Prepare to capture intermediate states
    intermediate_states = []
    
    # Create a test graph execution
    print("Starting test execution...")
    
    # Hack: Add a simple callback to the agent
    original_invoke = agent.async_agent.ainvoke
    
    async def invoke_with_capture(state, config=None):
        # Capture the input state
        intermediate_states.append(deepcopy(state))
        
        # Call the original
        result = await original_invoke(state, config)
        
        # Capture the output state
        intermediate_states.append(deepcopy(result))
        
        return result
    
    # Replace the invoke method temporarily
    agent.async_agent.ainvoke = invoke_with_capture
    
    # Run the agent
    try:
        print("Running agent...")
        final_state = await agent.async_agent.ainvoke(initial_state)
        print("Test execution completed successfully.")
    except Exception as e:
        print(f"Error during test execution: {str(e)}")
        traceback.print_exc()
        final_state = None
    finally:
        # Restore the original method
        agent.async_agent.ainvoke = original_invoke
    
    return intermediate_states, final_state


# For backward compatibility
create_custom_react_agent = create_react_agent

def create_mcp_react_agent(llm, server_command, system_message, verbose=False):
    """
    Create a ReAct agent for MCP server interaction.
    
    This is a simplified wrapper for client tests that makes it easier
    to create a ReAct agent with MCP tools.
    
    Args:
        llm: The language model to use
        server_command: The command to start the MCP server
        system_message: The system message to use
        verbose: Whether to enable verbose logging
        
    Returns:
        A function that takes a query and returns a response
    """
    import asyncio
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from langchain_mcp_adapters.tools import load_mcp_tools
    
    async def _run_agent_with_mcp(query):
        # Parse the server command
        parts = server_command.split()
        cmd = parts[0]
        args = parts[1:] if len(parts) > 1 else []
        
        # Set up server parameters
        server_params = StdioServerParameters(command=cmd, args=args)
        
        # Connect to the server and load tools
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize the session and load tools
                await session.initialize()
                raw_tools = await load_mcp_tools(session)
                
                # Create the agent
                agent = create_react_agent(
                    llm=llm,
                    tools=raw_tools,
                    system_prompt=system_message
                )
                
                # Run the agent
                state = await run_agent(agent, query)
                
                # Extract the final answer
                final_state = normalize_state(state)
                messages = final_state.get("messages", [])
                
                # Find the last AI message with content
                for msg in reversed(messages):
                    if isinstance(msg, AIMessage) and msg.content:
                        return msg.content
                
                # Fallback - return the last message or empty string
                if messages:
                    return str(messages[-1])
                return ""
    
    # Return a function that runs the agent
    def agent_fn(query):
        return asyncio.run(_run_agent_with_mcp(query))
    
    return agent_fn 