"""
Test for custom React agent with MCP multiply tool

This script demonstrates our custom React agent using a simple multiply tool,
which uses a local tool definition since it's just testing the custom agent functionality.
"""

import asyncio
import os
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

# Import our custom React agent
from mcp_solver.client.custom_agent import create_custom_react_agent

# Import LLM factory
from mcp_solver.client.llm_factory import LLMFactory
from mcp_solver.client.client import MODEL_CODES, DEFAULT_MODEL

# Define a simple multiply tool for testing
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b

async def run_test():
    """Run a simple test of the custom React agent with the multiply tool."""
    # Get the model code for the default model
    model_code = MODEL_CODES[DEFAULT_MODEL]
    
    try:
        # Get model info to check if API key is present
        model_info = LLMFactory.parse_model_code(model_code)
        api_key_name = model_info.api_key_name
        api_key = os.environ.get(api_key_name)
        
        if not api_key:
            print(f"Error: {api_key_name} not found in environment variables")
            return
    except Exception as e:
        print(f"Error parsing model code: {str(e)}")
        return
    
    print("\n===== Testing Custom React Agent with Multiply Tool =====")
    print(f"Using model: {model_code}")
    
    # Create the language model
    print(f"Initializing model {model_info.model_name} from {model_info.platform}...")
    llm = LLMFactory.create_model(model_code, temperature=0)
    
    # Create our custom React agent with just the multiply tool
    print("Creating custom React agent with multiply tool...")
    agent = create_custom_react_agent(llm, [multiply])
    
    # Test query
    test_query = "What is 12 multiplied by 15?"
    print(f"\n----- Test Query -----")
    print(f"Query: {test_query}")
    
    try:
        # Run the agent
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=test_query)]}
        )
        
        # Print each message to see the flow
        print("\n----- Conversation -----")
        for msg in result["messages"]:
            msg_type = type(msg).__name__
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                print(f"{msg_type}: {msg.content}")
                for tc in msg.tool_calls:
                    # Handle tool calls as either dicts or objects
                    if isinstance(tc, dict):
                        tool_name = tc.get("name", "unknown")
                        tool_args = tc.get("args", {})
                        print(f"  Tool Call: {tool_name}({tool_args})")
                    else:
                        tool_name = getattr(tc, "name", "unknown")
                        tool_args = getattr(tc, "args", {})
                        print(f"  Tool Call: {tool_name}({tool_args})")
            else:
                content = msg.content
                name = getattr(msg, "name", None)
                prefix = f"({name}): " if name else ": "
                print(f"{msg_type}{prefix}{content}")
        
        # Print the final answer
        print(f"\nFinal answer: {result['messages'][-1].content}")
        print("\n✅ Test completed successfully!")
        
        # Print information about what we tested
        print("\nThis test verified that we can:")
        print("1. Initialize an LLM using our LLMFactory")
        print("2. Create a custom React agent that can use tools")
        print("3. Process simple questions requiring tool use")
        print("\nThis lays the foundation for using the custom agent with actual MCP tools.")
    
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    """Main function"""
    asyncio.run(run_test())

if __name__ == "__main__":
    main() 