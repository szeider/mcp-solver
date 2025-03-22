"""
Custom ReAct Agent Implementation for MCP Solver

This module contains a custom implementation of a ReAct agent based on LangGraph
that follows the canonical implementation pattern from the LangGraph documentation.
"""

from typing import Annotated, Sequence, List, Optional, TypedDict, Any, Dict
import asyncio
import inspect
import sys
import json

from langgraph.graph import StateGraph, END
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.graph.message import add_messages

from langchain_core.messages import (
    AIMessage, 
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage
)
from langchain_core.tools import BaseTool
from langchain_core.language_models.chat_models import BaseChatModel


class AgentState(TypedDict):
    """The state of the agent."""
    # add_messages is a reducer for message lists
    messages: Annotated[Sequence[BaseMessage], add_messages]


def create_custom_react_agent(
    llm: BaseChatModel,
    tools: List[BaseTool],
    system_prompt: Optional[str] = None,
):
    """
    Create a custom React agent using LangGraph.
    
    Args:
        llm: The language model to use
        tools: List of tools to provide to the agent
        system_prompt: Optional system prompt to override default
        
    Returns:
        A compiled StateGraph that can be used for agent interactions
    """
    # Create a model with bound tools
    model = llm.bind_tools(tools)
    
    # Set up system prompt with tools info if not provided
    if system_prompt is None:
        tool_descs = "\n".join([
            f"- {tool.name}: {tool.description}" for tool in tools
        ])
        system_prompt = f"""You are a helpful AI assistant that can use tools.

Available tools:
{tool_descs}

Use these tools to help answer the user's question.
If the user's request doesn't require tools, simply respond to the best of your ability.
"""
    
    # Simple agent state with just messages
    workflow = StateGraph(AgentState)
    
    # Define model node function
    def call_model(state: AgentState) -> Dict:
        """Call the model with the current state."""
        messages = state["messages"]
        
        # Add system message first if it doesn't exist yet
        if not any(isinstance(msg, SystemMessage) for msg in messages):
            messages = [SystemMessage(content=system_prompt)] + list(messages)
        
        # Invoke the model and return updated messages
        response = model.invoke(messages)
        return {"messages": messages + [response]}
    
    # Function to determine if we should continue or end
    def should_continue(state: AgentState) -> str:
        """Determine whether to continue or end based on the current state."""
        messages = state["messages"]
        last_message = messages[-1]
        
        # If last message has tool calls, we should continue to the tool node
        if (
            isinstance(last_message, AIMessage) and 
            hasattr(last_message, "tool_calls") and 
            last_message.tool_calls and 
            len(last_message.tool_calls) > 0
        ):
            return "tools"
        
        # Otherwise we're done
        return END
    
    # Add the model node
    workflow.add_node("model", call_model)
    
    # Add the tool node using the built-in ToolNode
    tool_node = ToolNode(tools)
    workflow.add_node("tools", tool_node)
    
    # Add edges
    workflow.add_edge("model", "tools")
    workflow.add_edge("tools", "model")
    
    # Set condition to check if tools are needed
    workflow.add_conditional_edges("model", should_continue)
    
    # Set entry point
    workflow.set_entry_point("model")
    
    # Compile and return
    return workflow.compile()


# Simple testing utility
def test_agent_with_multiply():
    """Test the custom agent with a simple multiply tool."""
    from langchain_core.tools import tool
    from mcp_solver.client.llm_factory import LLMFactory
    import os
    
    @tool
    def multiply(a: int, b: int) -> int:
        """Multiply two numbers together."""
        return a * b
    
    # Get an LLM from the factory
    llm = LLMFactory.create_model("claude-3-5-sonnet", temperature=0)
    
    # Create the agent
    agent = create_custom_react_agent(llm, [multiply])
    
    # Run a test
    result = agent.invoke(
        {"messages": [HumanMessage(content="What is 12 multiplied by 15?")]}
    )
    
    # Print the result
    print("\nFinal messages:")
    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            print(f"AI: {msg.content}")
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"  Tool Call: {tc.name}({tc.args})")
        elif isinstance(msg, HumanMessage):
            print(f"Human: {msg.content}")
        elif isinstance(msg, ToolMessage):
            print(f"Tool ({msg.name}): {msg.content}")
        elif isinstance(msg, SystemMessage):
            print(f"System: {msg.content}")
    
    return result


if __name__ == "__main__":
    # Run a test of the agent if executed directly
    test_agent_with_multiply() 