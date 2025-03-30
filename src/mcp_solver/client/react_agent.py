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
import logging
import os
import re

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
    state_update = {"messages": messages + [response], "mem_tokens_used": total_tokens}
    
    # Also preserve the token counts separately for reporting
    state_update["mem_main_input_tokens"] = token_counter.main_input_tokens
    state_update["mem_main_output_tokens"] = token_counter.main_output_tokens
    
    return state_update


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
    """Call a reviewer to assess the solution.
    
    This function takes the current agent state, which includes the solution,
    and calls a reviewer to assess its correctness.
    
    Args:
        state: The current agent state
        model: The language model to use
        
    Returns:
        Updated state with review result added
    """
    global mem_solution
    
    print("Starting solution review process...", flush=True)

    # Extract necessary information from state
    mem_problem = state.get("mem_problem", "No problem statement provided")
    mem_model = state.get("mem_model", "No model code provided")  
    solution = state.get("mem_solution", mem_solution)

    # Process the solution to make it more readable if it's in JSON format
    processed_solution = solution
    try:
        if isinstance(solution, str) and "solution" in solution:
            # Try using ast.literal_eval which is safer than eval for parsing Python literals
            import ast
            try:
                # Convert the string representation of a dict to an actual dict
                solution_dict = ast.literal_eval(solution)
                
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
                solution_match = re.search(r"'solution'\s*:\s*({[^}]+})", solution)
                if solution_match:
                    try:
                        # Try to parse just the solution part
                        solution_part = solution_match.group(1).replace("'", '"')
                        solution_values = json.loads(solution_part)
                        processed_solution = "Solution values found:\n"
                        for var, value in solution_values.items():
                            processed_solution += f"{var} = {value}\n"
                        
                        # Check for satisfiability
                        if "'satisfiable': True" in solution:
                            processed_solution += "\nThe model is satisfiable."
                    except json.JSONDecodeError:
                        # If JSON parsing fails, extract manually
                        value_matches = re.findall(r"'([^']+)':\s*(\d+)", solution_match.group(1))
                        if value_matches:
                            processed_solution = "Solution values found:\n"
                            for var, value in value_matches:
                                processed_solution += f"{var} = {value}\n"
    except Exception as e:
        # Log errors in solution processing but continue
        print(f"Note: Solution preprocessing encountered an issue: {str(e)}", flush=True)
        pass
    
    # Create reviewer prompt using Template
    reviewer_template = Template(state.get("review_prompt", ""))
    reviewer_prompt = reviewer_template.substitute(
        PROBLEM=mem_problem,
        MODEL=mem_model,
        SOLUTION=processed_solution
    )
    
    # For debug, show just a preview of the prompt
    print(f"Preparing review prompt ({len(reviewer_prompt)} characters)...", flush=True)
    
    # Request structured output with clearer formatting instructions
    structured_prompt = f"""{reviewer_prompt}

Return your assessment in JSON format with these fields:
- correctness: must be exactly one of "correct", "incorrect", or "unknown"
- explanation: your detailed justification
"""
    
    try:
        # Track reviewer input tokens using the TokenCounter
        token_counter = TokenCounter.get_instance()
        token_counter.count_reviewer_input(structured_prompt)
        
        # Create a human message with the prompt
        review_message = HumanMessage(content=structured_prompt)
        
        # Define the expected output structure
        from typing import Literal
        from pydantic import BaseModel
        
        class ReviewResult(BaseModel):
            explanation: str
            correctness: Literal["correct", "incorrect", "unknown"]
        
        try:
            # Use LangChain's built-in structured output method
            print("Using model's structured output capabilities...", flush=True)
            structured_model = model.with_structured_output(ReviewResult)
            result = structured_model.invoke(review_message)
            
            # Convert the structured output to a dict
            review_result = {
                "correctness": result.correctness,
                "explanation": result.explanation
            }
            
            # Keep the original response for review_text
            response = model.invoke([review_message])
            mem_review_text = response.content
            
            # Track reviewer output tokens using the TokenCounter
            token_counter.count_reviewer_output(response.content)
        
        except (TypeError, ValueError, AttributeError, ImportError) as e:
            # Fallback for models that don't support structured output
            print(f"Using standard invocation (structured output not available)", flush=True)
            
            # Invoke the model normally
            response = model.invoke([review_message])
            
            # Track reviewer output tokens using the TokenCounter
            token_counter.count_reviewer_output(response.content)
            
            # Try to parse the JSON from the response
            from langchain_core.output_parsers import JsonOutputParser
            import json
            import re
            
            review_result = {"correctness": "unknown", "explanation": "Failed to parse review"}
            
            # Try direct JSON parsing
            try:
                json_parser = JsonOutputParser()
                review_result = json_parser.parse(response.content)
                print("Successfully parsed JSON response", flush=True)
            except Exception:
                # If direct parsing fails, try looking for JSON in code blocks
                content = response.content
                try:
                    # Look for ```json blocks
                    json_block_match = re.search(r'```(?:json)?\s*(.*?)\s*```', content, re.DOTALL)
                    if json_block_match:
                        json_content = json_block_match.group(1).strip()
                        print("Extracted JSON from code block", flush=True)
                        review_result = json.loads(json_content)
                    else:
                        # If no code block, look for any JSON object in the response
                        json_match = re.search(r'(\{.*\})', content, re.DOTALL)
                        if json_match:
                            json_content = json_match.group(1).strip()
                            print("Extracted JSON from text", flush=True)
                            review_result = json.loads(json_content)
                except Exception as e:
                    print(f"JSON parsing failed, using defaults", flush=True)
            
            # Store full review content for review_text
            mem_review_text = response.content
        
        # Extract verdict from review result
        mem_review_verdict = review_result.get("correctness", "unknown")
        
        # Check for consistency between verdict and explanation
        explanation = review_result.get("explanation", "")
        
        # Look for conclusive statements at the end of the explanation
        conclusion_check = explanation.lower().strip()
        last_lines = conclusion_check.split('\n')[-3:]  # Get last 3 lines
        conclusion_text = ' '.join(last_lines)
        
        # Check if there's a mismatch between verdict and conclusion
        if "solution is correct" in conclusion_text and mem_review_verdict != "correct":
            print(f"Fixing verdict mismatch: changed from '{mem_review_verdict}' to 'correct'", flush=True)
            mem_review_verdict = "correct"
            review_result["correctness"] = "correct"
        elif "solution is incorrect" in conclusion_text and mem_review_verdict != "incorrect":
            print(f"Fixing verdict mismatch: changed from '{mem_review_verdict}' to 'incorrect'", flush=True)
            mem_review_verdict = "incorrect"
            review_result["correctness"] = "incorrect"
        
        print(f"Review complete: verdict is '{mem_review_verdict}'", flush=True)
        
        # Return both the review result and keep existing messages
        return {
            "review_result": review_result, 
            "mem_review_verdict": mem_review_verdict,
            "mem_review_text": mem_review_text,
            "messages": state.get("messages", [])  # Preserve existing messages
        }
    except Exception as e:
        print(f"Review process error: {str(e)}", flush=True)
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
def create_react_agent(model: BaseChatModel, tools: List[BaseTool], system_prompt: Optional[str] = None) -> Any:
    """Create a ReAct agent with the given tools and model.
    
    This function creates a ReAct agent with the given tools and model. It wires the agent
    up to perform the ReAct loop (act, think, observe, repeat).
    
    Args:
        model: The language model to use
        tools: The tools to give the agent
        system_prompt: Optional system prompt to use
        
    Returns:
        A runnable ReAct agent
    """
    # Create a mapping of tool name to tool
    tools_by_name = {tool.name: tool for tool in tools}
    
    # Create the agent state graph
    workflow = StateGraph(AgentState)
    
    # Define a partial call_model function that binds the model and system_prompt
    def call_model_with_system_prompt(state):
        return call_model(state, model, system_prompt)
    
    # Define a partial call_tools function that binds the tools
    def call_tools_with_tools(state):
        return call_tools(state, tools_by_name)
    
    # Define the review function that binds the model
    def call_review_with_model(state):
        return call_reviewer(state, model)
    
    # Add the model node
    workflow.add_node("model", call_model_with_system_prompt)
    
    # Add the tools node
    workflow.add_node("tools", call_tools_with_tools)
    
    # Add the reviewer node 
    workflow.add_node("reviewer", call_review_with_model)
    
    # Tool node for handling tool calls
    workflow.add_node("tool_node", ToolNode(tools))
    
    # Add edges between the nodes
    workflow.add_edge("model", "tools")
    workflow.add_edge("tools", "model")
    
    # Define conditional edge for ending after model step
    def should_continue(state):
        """Check if there are any tool calls in the last AI message."""
        # Get the most recent message
        messages = state["messages"]
        if not messages:
            return "model"
            
        # Get the last message from the AI
        last_message = messages[-1]
        if not isinstance(last_message, AIMessage):
            return "model"
            
        # Check if there are any tool calls
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            # There are explicit tool calls
            return "tools"
            
        # Check for tool calls in additional kwargs
        additional_kwargs = getattr(last_message, "additional_kwargs", {})
        if additional_kwargs and "tool_calls" in additional_kwargs:
            # There are tool calls in the additional kwargs
            return "tools"
            
        # No tool calls, check for review prompt
        if state.get("review_prompt") and state.get("mem_model") and state.get("mem_solution"):
            # We have everything needed for a review
            return "reviewer"
            
        # If we get here, we're done
        return END
    
    # Add conditional edge out of model
    workflow.add_conditional_edges(
        "model",
        should_continue
    )
    
    # Define conditional edge for completion after tools step
    def should_end_after_tools(state):
        """Check if we should end after executing tools."""
        if state.get("review_prompt") and state.get("mem_model") and state.get("mem_solution"):
            # We have everything needed for a review
            return "reviewer"
        return "model"
    
    # Add conditional edge out of tools
    workflow.add_conditional_edges(
        "tools",
        should_end_after_tools
    )
    
    # Add the edge from reviewer to END
    workflow.add_edge("reviewer", END)
    
    # Compile the workflow
    app = workflow.compile()
    
    return app 