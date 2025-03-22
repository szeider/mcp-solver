#!/usr/bin/env python
"""
Test script for specifically testing the intermediate steps functionality
of the improved custom agent implementation.
"""
import asyncio
import sys
import os
from pathlib import Path
import traceback

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the required functionality
from mcp_solver.client.custom_agent import test_agent_intermediate_steps, normalize_state
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

async def test_intermediate_steps():
    """Test the custom agent's intermediate steps functionality."""
    print(f"\n{'='*60}")
    print("Testing Agent Intermediate Steps Functionality")
    print(f"{'='*60}\n")
    
    # Check if OpenAI API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please set your OpenAI API key to run this test.")
        return False
    
    try:
        # Create a simple query for testing
        query = "Create a variable x that is between 1 and 10, and make sure it's greater than 5."
        
        # Create a simple LLM
        llm = ChatOpenAI(model="gpt-3.5-turbo")
        
        # Create test tools
        tools = [
            {
                "name": "create_variable",
                "description": "Creates a variable with specified constraints",
                "parameters": {
                    "name": {"type": "string", "description": "Name of the variable"},
                    "min_value": {"type": "integer", "description": "Minimum value"},
                    "max_value": {"type": "integer", "description": "Maximum value"},
                }
            },
            {
                "name": "add_constraint",
                "description": "Adds a constraint to a variable",
                "parameters": {
                    "variable": {"type": "string", "description": "Variable name"},
                    "constraint": {"type": "string", "description": "Constraint expression (e.g., '> 5')"},
                }
            }
        ]
        
        # Mock tool execution
        async def mock_tool_executor(tool_name, tool_args):
            if tool_name == "create_variable":
                return f"Variable {tool_args.get('name')} created with range {tool_args.get('min_value')} to {tool_args.get('max_value')}"
            elif tool_name == "add_constraint":
                return f"Constraint {tool_args.get('constraint')} added to variable {tool_args.get('variable')}"
            else:
                return "Unknown tool called"
        
        print("Starting test with query:", query)
        print("Using tools:", [t["name"] for t in tools])
        
        # Run the test_agent_intermediate_steps function
        states, final_state = await test_agent_intermediate_steps(
            llm=llm,
            tools=tools,
            query=query,
            tool_executor=mock_tool_executor
        )
        
        # Check if we have intermediate states
        if states:
            print(f"\nIntermediate States Captured: {len(states)}")
            
            # Display information about each state
            for i, state in enumerate(states):
                print(f"\nState {i+1}:")
                
                # Normalize the state
                normalized_state = normalize_state(state)
                
                # Check if normalization worked
                if "messages" in normalized_state:
                    print(f"  Messages: {len(normalized_state['messages'])}")
                    
                    # Check if any tool calls were made
                    last_message = normalized_state["messages"][-1]
                    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                        print(f"  Tool Calls: {len(last_message.tool_calls)}")
                        for tc in last_message.tool_calls:
                            print(f"    - Tool: {tc.get('name', 'unknown')}")
                else:
                    print("  Normalization failed to produce expected format")
            
            print("\nFinal state:", "OK" if final_state else "Missing")
            return True
        else:
            print("\nTest failed: No intermediate states were captured.")
            return False
    
    except Exception as e:
        print(f"Error during test execution: {str(e)}")
        print(traceback.format_exc())
        return False

def main():
    """Main entry point for the test script."""
    try:
        result = asyncio.run(test_intermediate_steps())
        return 0 if result else 1
    except KeyboardInterrupt:
        print("\nTest interrupted.")
        return 130
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 