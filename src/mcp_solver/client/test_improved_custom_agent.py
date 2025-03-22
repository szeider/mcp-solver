#!/usr/bin/env python
"""
Test script to verify the improved custom agent implementation with MCP tools.
"""
import asyncio
import os
import sys
from typing import Dict, Any, List

# Import the MCP client components
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools

# Import our custom agent implementation
from mcp_solver.client.custom_agent import create_custom_react_agent, normalize_state
from langchain_core.messages import HumanMessage

# Import a simple LLM for testing
from langchain_openai import ChatOpenAI

# Test configuration
DEFAULT_SERVER_COMMAND = "uv"
DEFAULT_SERVER_ARGS = ["run", "mcp-solver-mzn"]  # Use the MiniZinc solver for testing

async def test_custom_agent_with_mcp():
    """
    Test the improved custom agent implementation with MCP tools.
    
    This test initializes an MCP server, loads its tools, creates a custom agent,
    and runs a simple test with it.
    """
    print("\n=== Testing Improved Custom Agent with MCP Tools ===\n")
    
    # Verify API key is set
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable is not set.")
        print("Please set it to run this test.")
        return False
    
    # Set up server parameters
    print("Setting up MCP server parameters...")
    server_params = StdioServerParameters(
        command=DEFAULT_SERVER_COMMAND,
        args=DEFAULT_SERVER_ARGS
    )
    
    try:
        # Connect to the MCP server and load tools
        print("Connecting to MCP server...")
        
        async with stdio_client(server_params) as (read, write):
            print("Connected successfully.")
            
            async with ClientSession(read, write) as session:
                # Initialize the connection
                print("Initializing MCP session...")
                await session.initialize()
                print("Session initialized.")
                
                # Load the MCP tools
                print("Loading MCP tools...")
                mcp_tools = await load_mcp_tools(session)
                print(f"Loaded {len(mcp_tools)} MCP tools.")
                
                # List available tools
                for i, tool in enumerate(mcp_tools):
                    print(f"  {i+1}. {tool.name}: {tool.description[:50]}..." if len(tool.description) > 50 else tool.description)
                
                # Create an LLM (using a smaller model for testing)
                print("\nCreating LLM...")
                llm = ChatOpenAI(model="gpt-3.5-turbo")
                
                # Create the custom agent
                print("Creating custom agent...")
                agent = create_custom_react_agent(
                    llm=llm,
                    tools=mcp_tools,
                    system_prompt="You are a helpful assistant specialized in constraint programming."
                )
                
                # Test query
                test_query = "Create a simple model with a variable x between 1 and 10 that must be greater than 5."
                print(f"\nRunning test query: '{test_query}'")
                
                # Run the agent with this query
                human_message = HumanMessage(content=test_query)
                initial_state = {"messages": [human_message]}
                
                # Execute the agent
                print("Executing agent...")
                final_state = agent.invoke(initial_state)
                
                # Normalize and print the result
                normalized_state = normalize_state(final_state)
                
                print("\n=== Agent Execution Complete ===\n")
                print("Message count:", len(normalized_state["messages"]))
                
                # Print messages in a user-friendly format
                for i, msg in enumerate(normalized_state["messages"]):
                    msg_type = type(msg).__name__
                    print(f"\n--- Message {i+1} ({msg_type}) ---")
                    
                    if msg_type == "HumanMessage":
                        print(f"Human: {msg.content}")
                    elif msg_type == "AIMessage":
                        print(f"AI: {msg.content[:200]}..." if len(msg.content) > 200 else msg.content)
                        
                        # Print tool calls if any
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            print("\nTool Calls:")
                            for tc in msg.tool_calls:
                                tc_name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "unknown")
                                print(f"  - {tc_name}")
                    elif msg_type == "ToolMessage":
                        print(f"Tool ({msg.name}): {msg.content[:100]}..." if len(msg.content) > 100 else msg.content)
                
                print("\n=== Test Complete ===")
                return True
                
    except Exception as e:
        import traceback
        print(f"Error during test: {str(e)}")
        print(traceback.format_exc())
        return False

def main():
    """Main entry point for the test script."""
    try:
        result = asyncio.run(test_custom_agent_with_mcp())
        return 0 if result else 1
    except KeyboardInterrupt:
        print("\nTest interrupted.")
        return 130
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 