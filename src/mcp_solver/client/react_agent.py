"""
ReAct Agent Implementation for MCP Solver

This module contains a ReAct (Reasoning+Acting) agent implementation built with LangGraph.
The implementation follows the canonical pattern for ReAct agents but is customized for
MCP Solver's specific use case.
"""

from typing import Annotated, Dict, List, Optional, Sequence, TypedDict, Any, Union, Literal, Callable, Tuple
import sys
import asyncio
import json
import traceback
from copy import deepcopy
from string import Template

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

# Import token counter
from mcp_solver.client.token_counter import TokenCounter


# Module-level variables for solution tracking
mem_solution = "No solution generated yet"


# Step 1: Define the agent state
class AgentState(TypedDict):
    """The state of the agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    # Add these optional fields for the reviewer
    review_prompt: Optional[str] = None
    mem_problem: Optional[str] = None
    mem_model: Optional[str] = None
    mem_solution: Optional[str] = None
    review_result: Optional[Dict[str, Any]] = None
    # New field for token usage
    mem_tokens_used: Optional[int] = None
    # New field for review verdict
    mem_review_verdict: Optional[str] = None
    # New field for tool usage
    mem_tool_usage: Optional[Dict[str, int]] = None
    # New field for full review text
    mem_review_text: Optional[str] = None


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
    
    # Track input tokens using the TokenCounter
    token_counter = TokenCounter.get_instance()
    token_counter.count_main_input(messages)
    
    # Call the model with the messages
    response = model.invoke(messages)
    
    # Track output tokens using the TokenCounter
    token_counter.count_main_output(response.content)
    
    # Get total tokens for state
    total_tokens = token_counter.get_total_tokens()
    
    # Return updated state with model response added and token count
    return {"messages": [response], "mem_tokens_used": total_tokens}


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
    
    # Copy tool usage from ToolStats singleton at the beginning
    tool_usage = {}
    try:
        from mcp_solver.client.tool_stats import ToolStats
        tool_stats = ToolStats.get_instance()
        if hasattr(tool_stats, "tool_calls") and tool_stats.tool_calls:
            tool_usage = dict(tool_stats.tool_calls)
    except Exception as e:
        print(f"Warning: Could not access ToolStats: {e}", file=sys.stderr)
    
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
    
    # If no tool calls were found, return empty messages but preserve the tool usage
    if not tool_calls:
        return {"messages": [], "mem_tool_usage": tool_usage}
    
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
    combined_state_updates = {}
    
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
        
        # Execute the tool and get both the message and state updates
        tool_message, state_updates = execute_tool_safely(tool, tool_name, tool_id, tool_args, loop)
        tool_results.append(tool_message)
        
        # Merge any state updates
        combined_state_updates.update(state_updates)
    
    # Update tool usage again after execution
    try:
        tool_stats = ToolStats.get_instance()
        if hasattr(tool_stats, "tool_calls") and tool_stats.tool_calls:
            tool_usage = dict(tool_stats.tool_calls)
    except Exception:
        pass
    
    # Return updated state with tool messages, tool usage, and any other state updates
    result = {"messages": tool_results, "mem_tool_usage": tool_usage}
    
    # Add any state updates from tools
    if combined_state_updates:
        result.update(combined_state_updates)
        
    return result


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


def execute_tool_safely(tool: BaseTool, tool_name: str, tool_id: str, tool_args: Dict[str, Any], loop: asyncio.AbstractEventLoop) -> Tuple[ToolMessage, Dict[str, Any]]:
    """Execute a tool safely, handling errors and timeouts.
    
    Args:
        tool: The tool to execute
        tool_name: The name of the tool
        tool_id: The ID of the tool call
        tool_args: The arguments to pass to the tool
        loop: The event loop to use for async execution
        
    Returns:
        A tuple containing:
        - ToolMessage with the result or error
        - Dictionary with state updates
    """
    # Global declaration at the beginning of the function
    global mem_solution
    
    # Dictionary for state updates
    state_updates = {}
    
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
        
        # Special handling for Z3 results
        if "'values':" in result_str:
            # Try to extract and save the values directly
            import re
            values_match = re.search(r"'values':\s*(\{[^\}]+\})", result_str)
            if values_match:
                extracted_values = values_match.group(1)
                # Update the global mem_solution directly 
                mem_solution = extracted_values
                # Also add to state updates
                state_updates["mem_solution"] = extracted_values
        
        # Create a tool message with the result
        tool_message = ToolMessage(
            content=result_str,
            tool_call_id=tool_id,
            name=tool_name
        )
        
        # Special handling for solve_model tool
        if tool_name == "solve_model":
            # For solve_model, directly set the mem_solution right after the function returns
            # This will make it available immediately rather than waiting for the next tool call
            
            # Try to extract Z3 solution if it looks like one
            if "'values':" in result_str:
                import re
                values_match = re.search(r"'values':\s*(\{[^\}]+\})", result_str)
                if values_match:
                    mem_solution = values_match.group(1)
                    state_updates["mem_solution"] = values_match.group(1)
                else:
                    mem_solution = result_str
                    state_updates["mem_solution"] = result_str
            else:
                mem_solution = result_str
                state_updates["mem_solution"] = result_str
        
        # Special handling for get_model tool - store the raw model
        elif tool_name == "get_model":
            # Store the raw model result directly in state_updates
            state_updates["mem_model"] = result_str
            
            # Keep function attribute for backward compatibility
            if not hasattr(execute_tool_safely, "mem_solution"):
                execute_tool_safely.mem_solution = {}
            # This is the raw, unformatted Python model/solution
            execute_tool_safely.mem_solution[tool_id] = result
        
        # Store in the static variable too for backward compatibility
        if not hasattr(execute_tool_safely, "mem_solution"):
            execute_tool_safely.mem_solution = {}
            
        execute_tool_safely.mem_solution[tool_id] = result_str
        
        return tool_message, state_updates
        
    except Exception as e:
        # In case of error, create a tool message with error info
        error_msg = f"Error executing tool {tool_name}: {str(e)}"
        print(f"Error: {error_msg}", file=sys.stderr)
        
        # Include traceback for better debugging
        tb_str = traceback.format_exc()
        print(f"Traceback: {tb_str}", file=sys.stderr)
        
        tool_message = ToolMessage(
            content=error_msg,
            tool_call_id=tool_id,
            name=tool_name
        )
        
        return tool_message, state_updates


# Step 3.5: Define the reviewer node function
def call_reviewer(state: AgentState, model: BaseChatModel) -> Dict:
    """Review the solution using the review prompt and model/solution memory.
    
    Args:
        state: The current agent state with mem_problem, mem_model, mem_solution
        model: The language model to use
        
    Returns:
        Updated state with review results
    """
    # Get necessary data from state
    review_prompt = state.get("review_prompt", "")
    mem_problem = state.get("mem_problem", "")
    mem_model = state.get("mem_model", "")
    mem_solution = state.get("mem_solution", "")
    
    # Process the solution to make it more readable if it's in JSON format
    processed_solution = mem_solution
    try:
        if isinstance(mem_solution, str) and "solution" in mem_solution:
            # Try using ast.literal_eval which is safer than eval for parsing Python literals
            import ast
            try:
                # Convert the string representation of a dict to an actual dict
                solution_dict = ast.literal_eval(mem_solution)
                
                if isinstance(solution_dict, dict) and 'solution' in solution_dict:
                    solution_values = solution_dict['solution']
                    
                    processed_solution = "Solution values found:\n"
                    for var, value in solution_values.items():
                        processed_solution += f"{var} = {value}\n"
                    
                    # Add a clear statement about satisfiability
                    if solution_dict.get('satisfiable', False):
                        processed_solution += "\nThe model is satisfiable."
                    else:
                        processed_solution += "\nThe model is NOT satisfiable."
            except (SyntaxError, ValueError):
                # If literal_eval fails, try regex approach
                import re
                import json
                
                # Extract the solution part
                solution_match = re.search(r"'solution'\s*:\s*({[^}]+})", mem_solution)
                if solution_match:
                    try:
                        # Try to parse just the solution part
                        solution_part = solution_match.group(1).replace("'", '"')
                        solution_values = json.loads(solution_part)
                        processed_solution = "Solution values found:\n"
                        for var, value in solution_values.items():
                            processed_solution += f"{var} = {value}\n"
                        
                        # Check for satisfiability
                        if "'satisfiable': True" in mem_solution:
                            processed_solution += "\nThe model is satisfiable."
                    except json.JSONDecodeError:
                        # If JSON parsing fails, extract manually
                        value_matches = re.findall(r"'([^']+)':\s*(\d+)", solution_match.group(1))
                        if value_matches:
                            processed_solution = "Solution values found:\n"
                            for var, value in value_matches:
                                processed_solution += f"{var} = {value}\n"
    except Exception:
        # Silently handle errors in solution processing
        pass
    
    # Create reviewer prompt using Template
    reviewer_template = Template(review_prompt)
    reviewer_prompt = reviewer_template.substitute(
        PROBLEM=mem_problem,
        MODEL=mem_model,
        SOLUTION=processed_solution
    )
    
    # Request structured output
    structured_prompt = f"{reviewer_prompt}\n\nPlease provide your review in the following JSON format:\n{{\"correctness\": \"[correct|incorrect|unknown]\", \"explanation\": \"Your detailed explanation\"}}"
    
    try:
        # Track reviewer input tokens using the TokenCounter
        token_counter = TokenCounter.get_instance()
        token_counter.count_reviewer_input(structured_prompt)
        
        # Call the model with a HumanMessage
        review_message = HumanMessage(content=structured_prompt)
        response = model.invoke([review_message])
        
        # Track reviewer output tokens using the TokenCounter
        token_counter.count_reviewer_output(response.content)
        
        # Try to parse JSON from the response
        review_result = {"correctness": "unknown", "explanation": "Failed to parse review"}
        try:
            # Look for JSON in the response
            import re
            import json
            
            # Try to find JSON pattern in the text
            json_match = re.search(r'({.*})', response.content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                parsed = json.loads(json_str)
                if isinstance(parsed, dict) and "correctness" in parsed:
                    review_result = parsed
        except Exception:
            # Silently handle parsing errors
            pass
        
        # Extract verdict from review result
        mem_review_verdict = review_result.get("correctness", "unknown")
        
        # Store full review content for review_text
        mem_review_text = response.content
        
        # Return both the review result and keep existing messages
        return {
            "review_result": review_result, 
            "mem_review_verdict": mem_review_verdict,
            "mem_review_text": mem_review_text,
            "messages": state.get("messages", [])  # Preserve existing messages
        }
    except Exception as e:
        # Return a default review result and keep existing messages
        return {
            "review_result": {
                "correctness": "unknown", 
                "explanation": f"Error running reviewer: {str(e)}"
            },
            "mem_review_verdict": "unknown",
            "mem_review_text": f"Error running reviewer: {str(e)}",
            "messages": state.get("messages", [])  # Preserve existing messages
        }


# Step 4: Define the routing logic
def router(state: AgentState) -> Union[Literal["call_model", "call_tools", "call_reviewer"], Literal[END]]:
    """Determine the next node in the graph based on the current state.
    
    This function examines the current state and decides which node should be executed next:
    - If the last message is from a tool or human, it routes to the model node
    - If the last message is from the AI and contains tool calls, it routes to the tools node
    - If the agent has completed its task, route to reviewer before ending
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
    
    # If no conditions match or AI message doesn't have tool calls, go to reviewer if we haven't already
    if not state.get("review_result"):  # Only go to reviewer if we haven't already
        return "call_reviewer"
    
    # If we've already been to the reviewer, end the graph
    return END


# Step 5: Define the complete graph
def create_react_agent(
    llm: BaseChatModel,
    tools: List[BaseTool],
    system_prompt: Optional[str] = None,
    review_prompt: Optional[str] = None,
):
    """
    Create a ReAct agent using LangGraph.
    
    Args:
        llm: The language model to use
        tools: List of tools to provide to the agent
        system_prompt: Optional system prompt
        review_prompt: Optional review prompt for solution evaluation
        
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
    
    # Add reviewer node (using the same LLM but without tools)
    workflow.add_node("call_reviewer", lambda state: call_reviewer(state, llm))
    
    # Add conditional edges with our router
    workflow.add_conditional_edges(
        "call_model",  # from node
        router,        # routing function
        {
            "call_tools": "call_tools",  # route condition -> target node
            "call_reviewer": "call_reviewer",  # route to reviewer
            END: END                     # end the graph
        }
    )
    
    workflow.add_conditional_edges(
        "call_tools",  # from node
        router,        # routing function
        {
            "call_model": "call_model",  # route condition -> target node
            "call_reviewer": "call_reviewer",  # route to reviewer
            END: END                     # end the graph
        }
    )
    
    workflow.add_conditional_edges(
        "call_reviewer",  # from node
        router,           # routing function
        {
            END: END      # after reviewer, always end
        }
    )
    
    # Set the entry point - always start with the model
    workflow.set_entry_point("call_model")
    
    # Compile the graph
    async_agent = workflow.compile()
    
    # Wrap the async agent with a synchronous interface
    return SyncCompatWrapper(async_agent)


async def run_agent(agent, message: str, config: Optional[RunnableConfig] = None, review_prompt: Optional[str] = None):
    """
    Async function for running an agent on a human input message.
    
    Args:
        agent: The agent to run
        message: The human input message
        config: Optional configuration
        review_prompt: Optional review prompt for solution evaluation
        
    Returns:
        The final state of the agent
    """
    # Global declaration at the beginning of the function
    global mem_solution
    
    # Reset global mem_solution at the start of each run
    mem_solution = "No solution generated yet"
    
    # If it's our wrapped agent, unwrap it first to use the async version
    if isinstance(agent, SyncCompatWrapper):
        async_agent = agent.async_agent
    else:
        async_agent = agent
        
    # Create the initial state
    initial_state = {
        "messages": [HumanMessage(content=message)],
        "mem_solution": "No solution generated yet",  # Initialize mem_solution
        "mem_model": "No model captured yet",         # Initialize mem_model
        "mem_problem": message                        # Store the problem statement
    }
    
    # Add review_prompt if provided
    if review_prompt:
        initial_state["review_prompt"] = review_prompt
    
    # Run the agent asynchronously
    final_state = await async_agent.ainvoke(initial_state, config)
    
    # Normalize the state to ensure consistent format
    normalized_state = normalize_state(final_state)
    
    # Try to extract Z3 solution from messages directly
    for message in normalized_state.get("messages", []):
        if isinstance(message, ToolMessage) and "solve_model" in message.name and "'values':" in message.content:
            try:
                import re
                values_match = re.search(r"'values':\s*(\{[^\}]+\})", message.content)
                if values_match:
                    extracted_solution = values_match.group(1)
                    mem_solution = extracted_solution
                    normalized_state["mem_solution"] = extracted_solution
            except Exception:
                pass
    
    # If the module-level solution has been updated, use it
    if mem_solution != "No solution generated yet":
        normalized_state["mem_solution"] = mem_solution
    # Otherwise fall back to the execute_tool_safely.mem_solution
    elif hasattr(execute_tool_safely, "mem_solution") and execute_tool_safely.mem_solution:
        # Get the latest solution (using the last tool_id if there were multiple calls)
        latest_solution = None
        
        # Check if we have a specifically extracted values field for Z3
        if "values" in execute_tool_safely.mem_solution:
            latest_solution = execute_tool_safely.mem_solution["values"]
        else:
            # Otherwise get the last solution added by a tool call
            tool_ids = [k for k in execute_tool_safely.mem_solution.keys() if k != "values"]
            if tool_ids:
                latest_solution = execute_tool_safely.mem_solution[tool_ids[-1]]
        
        if latest_solution:
            normalized_state["mem_solution"] = latest_solution
    
    # Ensure token usage is in the normalized state
    if "mem_tokens_used" not in normalized_state:
        try:
            token_counter = TokenCounter.get_instance()
            normalized_state["mem_tokens_used"] = token_counter.get_total_tokens()
        except Exception as e:
            print(f"Warning: Could not access TokenCounter: {e}", file=sys.stderr)
    
    # Ensure tool usage is in the normalized state
    try:
        from mcp_solver.client.tool_stats import ToolStats
        tool_stats = ToolStats.get_instance()
        if hasattr(tool_stats, "tool_calls") and tool_stats.tool_calls:
            normalized_state["mem_tool_usage"] = dict(tool_stats.tool_calls)
    except Exception as e:
        print(f"Warning: Could not access ToolStats in run_agent: {e}", file=sys.stderr)
    
    # If review_result exists but mem_review_text doesn't, store review_result explanation in mem_review_text
    if "review_result" in normalized_state and normalized_state["review_result"]:
        if "mem_review_text" not in normalized_state or not normalized_state["mem_review_text"]:
            if isinstance(normalized_state["review_result"], dict):
                # Store the explanation as review text
                normalized_state["mem_review_text"] = normalized_state["review_result"].get("explanation", 
                                                   json.dumps(normalized_state["review_result"]))
            else:
                # Store the entire review result
                normalized_state["mem_review_text"] = str(normalized_state["review_result"])
    
    # Ensure review verdict is in the normalized state
    if "mem_review_verdict" not in normalized_state and "review_result" in normalized_state:
        if isinstance(normalized_state["review_result"], dict) and "correctness" in normalized_state["review_result"]:
            normalized_state["mem_review_verdict"] = normalized_state["review_result"]["correctness"]
    
    # Print all state variables consistently for run_test.py to capture
    # These prints should be the single source of truth for run_test.py
    if "mem_tokens_used" in normalized_state:
        print(f"mem_tokens_used: {normalized_state['mem_tokens_used']}")
    
    if "mem_review_verdict" in normalized_state:
        print(f"mem_review_verdict: {normalized_state['mem_review_verdict']}")
    
    if "mem_review_text" in normalized_state and normalized_state["mem_review_text"]:
        print(f"mem_review_text: {normalized_state['mem_review_text']}")
    
    if "mem_solution" in normalized_state and normalized_state["mem_solution"]:
        # Print solution to allow run_test.py to determine satisfiability
        print(f"mem_solution: {normalized_state['mem_solution']}")
    
    if "mem_tool_usage" in normalized_state and normalized_state["mem_tool_usage"]:
        print(f"mem_tool_usage: {json.dumps(normalized_state['mem_tool_usage'])}")
    
    return normalized_state


def normalize_state(state) -> Dict:
    """
    Normalize state to a consistent dictionary format regardless of LangGraph version.
    
    This utility handles the differences in state formats between LangGraph versions.
    
    Args:
        state: The state to normalize (could be dict, object, or other format)
        
    Returns:
        A normalized dictionary representation of the state
    """
    # First convert the state to a normalized dictionary format
    if isinstance(state, dict) and "messages" in state:
        # Copy all keys from the original state
        normalized_state = dict(state)
    elif hasattr(state, "messages"):
        # Try to convert all attributes to a dict
        normalized_state = {"messages": state.messages}
        # Copy other fields if they exist
        for field in ["mem_problem", "mem_model", "mem_solution", "mem_tokens_used", 
                     "mem_review_verdict", "mem_tool_usage", "mem_review_text", "review_result", "review_prompt"]:
            if hasattr(state, field):
                normalized_state[field] = getattr(state, field)
    elif hasattr(state, "values") and isinstance(state.values, dict):
        # Start with messages
        normalized_state = {}
        if "messages" in state.values:
            normalized_state["messages"] = state.values["messages"]
        
        # Copy other fields if they exist
        for field in ["mem_problem", "mem_model", "mem_solution", "mem_tokens_used", 
                     "mem_review_verdict", "mem_tool_usage", "mem_review_text", "review_result", "review_prompt"]:
            if field in state.values:
                normalized_state[field] = state.values[field]
    elif isinstance(state, list) and all(isinstance(msg, BaseMessage) for msg in state):
        normalized_state = {"messages": state}
    elif isinstance(state, dict) and len(state) > 0:
        # Look for any node output that contains messages
        normalized_state = {"messages": []}
        for node_name, node_output in state.items():
            if isinstance(node_output, dict) and "messages" in node_output:
                normalized_state = {"messages": node_output["messages"]}
                break
            elif isinstance(node_output, list) and all(isinstance(msg, BaseMessage) for msg in node_output):
                normalized_state = {"messages": node_output}
                break
    else:
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
        
        normalized_state = {"messages": messages}
    
    # Now ensure all required fields have default values if missing
    
    # Handle solution - prefer state value, but fall back to global if needed
    if "mem_solution" not in normalized_state or not normalized_state.get("mem_solution"):
        # Get the global mem_solution variable
        global mem_solution
        
        # If the global solution has been set, use it
        if mem_solution != "No solution generated yet":
            normalized_state["mem_solution"] = mem_solution
        # Otherwise check if the function attribute has been set
        elif hasattr(execute_tool_safely, "mem_solution") and execute_tool_safely.mem_solution:
            # Get the latest solution (using the last tool_id if there were multiple calls)
            latest_solution = None
            
            # Check if we have a specifically extracted values field for Z3
            if "values" in execute_tool_safely.mem_solution:
                latest_solution = execute_tool_safely.mem_solution["values"]
            else:
                # Otherwise get the last solution added by a tool call
                tool_ids = [k for k in execute_tool_safely.mem_solution.keys() if k != "values"]
                if tool_ids:
                    latest_solution = execute_tool_safely.mem_solution[tool_ids[-1]]
            
            if latest_solution:
                normalized_state["mem_solution"] = latest_solution
        else:
            # Default value if no solution is found anywhere
            normalized_state["mem_solution"] = "No solution generated yet"
    
    # Ensure token usage is in the normalized state
    if "mem_tokens_used" not in normalized_state:
        try:
            token_counter = TokenCounter.get_instance()
            normalized_state["mem_tokens_used"] = token_counter.get_total_tokens()
        except Exception as e:
            print(f"Warning: Could not access TokenCounter: {e}", file=sys.stderr)
    
    # Ensure tool usage is in the normalized state
    try:
        from mcp_solver.client.tool_stats import ToolStats
        tool_stats = ToolStats.get_instance()
        if hasattr(tool_stats, "tool_calls") and tool_stats.tool_calls:
            normalized_state["mem_tool_usage"] = dict(tool_stats.tool_calls)
    except Exception as e:
        print(f"Warning: Could not access ToolStats in run_agent: {e}", file=sys.stderr)
    
    # If review_result exists but mem_review_text doesn't, store review_result explanation in mem_review_text
    if normalized_state.get("review_result") and "mem_review_text" not in normalized_state:
        if isinstance(normalized_state["review_result"], dict):
            # Store the explanation as review text
            normalized_state["mem_review_text"] = normalized_state["review_result"].get("explanation", 
                                               json.dumps(normalized_state["review_result"]))
        else:
            # Store the entire review result
            normalized_state["mem_review_text"] = str(normalized_state["review_result"])
    
    # Ensure review verdict is in the normalized state
    if "mem_review_verdict" not in normalized_state and normalized_state.get("review_result"):
        if isinstance(normalized_state["review_result"], dict) and "correctness" in normalized_state["review_result"]:
            normalized_state["mem_review_verdict"] = normalized_state["review_result"]["correctness"]
    
    # Print all state variables consistently for run_test.py to capture
    # These prints should be the single source of truth for run_test.py
    if "mem_tokens_used" in normalized_state:
        print(f"mem_tokens_used: {normalized_state['mem_tokens_used']}")
    
    if "mem_review_verdict" in normalized_state:
        print(f"mem_review_verdict: {normalized_state['mem_review_verdict']}")
    
    if normalized_state.get("mem_review_text"):
        print(f"mem_review_text: {normalized_state['mem_review_text']}")
    
    if normalized_state.get("mem_solution"):
        # Print solution to allow run_test.py to determine satisfiability
        print(f"mem_solution: {normalized_state['mem_solution']}")
    
    if normalized_state.get("mem_tool_usage"):
        print(f"mem_tool_usage: {json.dumps(normalized_state['mem_tool_usage'])}")
    
    return normalized_state


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


# For backward compatibility
create_custom_react_agent = create_react_agent 