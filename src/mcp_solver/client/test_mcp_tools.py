#!/usr/bin/env python
"""
Test script to demonstrate using MCP tools with the custom agent.
This shows how to integrate the MCP solver tools with our custom ReAct agent.
"""

import asyncio
import sys
from typing import List, Dict, Any

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.language_models import ChatOpenAI

from mcp_solver.client.custom_agent import create_custom_react_agent
from langchain_mcp_adapters.mcp import ClientSession
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_mcp_adapters.stdio import stdio_client, StdioServerParameters

# Function to demonstrate using MCP tools with the custom agent
async def test_mcp_tools_with_agent(mcp_command: str):
    """
    Demonstrate using MCP tools with our custom agent.
    
    Args:
        mcp_command: The command to start the MCP solver server
    """
    print("Testing MCP tools with custom agent...")
    
    # Set up server parameters for stdio connection
    server_params = StdioServerParameters(command=mcp_command, args=[])
    
    try:
        # Create a direct client session and initialize MCP tools
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                try:
                    # Initialize the connection and get tools
                    await session.initialize()
                    raw_tools = await load_mcp_tools(session)
                    
                    # Print tools for debugging
                    print("\n📋 Available MCP tools:")
                    for i, tool in enumerate(raw_tools):
                        print(f"  {i+1}. {tool.name}: {tool.description}")
                    
                    # Create an LLM (this would be replaced with your actual LLM)
                    # Note: This is just a placeholder, in practice you would use your specific LLM
                    llm = ChatOpenAI(model="gpt-3.5-turbo")
                    
                    # Create the custom agent with MCP tools
                    print("\nCreating custom ReAct agent with MCP tools...")
                    agent = create_custom_react_agent(
                        llm=llm,
                        tools=raw_tools,
                        system_prompt="You are a helpful assistant for Mathematical Constraint Programming."
                    )
                    
                    # Create a human message for the agent to respond to
                    # In practice, this would come from user input
                    human_message = "Create a simple model with two variables x and y, " \
                                    "where x + y = 10 and x - y = 2, then solve it."
                    
                    print(f"\nExecuting agent with prompt: {human_message}")
                    
                    # Here we would normally run the agent, but for this test
                    # we're just showing the setup since we don't have a real MCP server
                    print("\nIn a real implementation, the agent would now:")
                    print("1. Parse the human request")
                    print("2. Create a model using the clear_model and add_item tools")
                    print("3. Set up constraints for x + y = 10 and x - y = 2")
                    print("4. Solve the model and return the results")
                    print("5. Format the solution for the user")
                    
                except Exception as e:
                    print(f"Error initializing MCP tools: {e}")
    except Exception as e:
        print(f"Error connecting to MCP server: {e}")

if __name__ == "__main__":
    print("Running MCP tools demonstration with custom agent...")
    print("="*80)
    
    # Example MCP command - in practice, this would be your actual command
    # This is just a placeholder for the test
    MCP_COMMAND = "python -m mcp_solver.server"
    
    try:
        # Run the demonstration
        print("NOTE: This is a demonstration only. It won't connect to a real MCP server.")
        print("To run with a real server, replace MCP_COMMAND with your actual server command.")
        print("="*80)
        
        # For demonstration purposes, we'll just show the setup without connecting
        asyncio.run(test_mcp_tools_with_agent(MCP_COMMAND))
    except Exception as e:
        print(f"Error running demonstration: {e}")
    
    print("="*80)
    print("Demonstration completed.") 