#!/usr/bin/env python
"""
Test script to demonstrate observing the agent state in a custom agent.
This shows how to examine what happens during the ReAct process.
"""

import asyncio
from typing import List, Dict, Any, ClassVar, Optional
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

from mcp_solver.client.custom_agent import (
    AgentState, 
    create_custom_react_agent,
    call_model,
    call_tools,
    router
)

# Function to demonstrate a step-by-step execution
def manual_step_execution():
    """Run the agent step by step manually to observe intermediate states."""
    print("Testing manual step-by-step execution...")
    
    # 1. Define a simple calculator tool
    from langchain_core.tools import tool
    
    @tool
    def calculator(expression: str) -> str:
        """Calculate the result of a mathematical expression."""
        try:
            return str(eval(expression))
        except Exception as e:
            return f"Error: {str(e)}"
    
    # 2. Create a custom mock LLM
    from langchain_core.language_models.chat_models import BaseChatModel
    from pydantic import Field, PrivateAttr
    
    class CustomMockLLM(BaseChatModel):
        """A custom mock LLM that returns predetermined responses."""
        
        # Private attribute for state tracking
        _call_count: int = PrivateAttr(default=0)
        
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            pass
            
        async def ainvoke(self, messages, **kwargs):
            return self._get_response(messages)
            
        def invoke(self, messages, **kwargs):
            return self._get_response(messages)
        
        def _get_response(self, messages):
            self._call_count += 1
            
            # First call - return a message with a tool call
            if self._call_count == 1:
                return AIMessage(
                    content="I'll help you with that calculation.", 
                    tool_calls=[{
                        "name": "calculator", 
                        "id": "calc1", 
                        "args": {"expression": "5+3"}
                    }]
                )
            # Second call - return a final response
            else:
                return AIMessage(content="The result is 8.")
                
        @property
        def _llm_type(self) -> str:
            return "custom_mock"
        
        def bind_tools(self, tools):
            return self
    
    # 3. Create our agent components
    mock_llm = CustomMockLLM()
    tools = [calculator]
    system_prompt = "You are a helpful assistant."
    
    # 4. Create an initial state with a human message
    state = AgentState(messages=[HumanMessage(content="What is 5 + 3?")])
    print(f"\nINITIAL STATE: {state}")
    
    # 5. Step 1: Call the model
    print("\nSTEP 1: Calling the model")
    state = call_model(state, mock_llm.bind_tools(tools), system_prompt)
    print(f"After model call: {state}")
    print(f"Last message: {state['messages'][-1]}")
    
    # 6. Check where to go next
    next_step = router(state)
    print(f"Next step: {next_step}")
    
    # 7. Step 2: Call the tools
    print("\nSTEP 2: Calling the tools")
    tools_by_name = {tool.name: tool for tool in tools}
    state = call_tools(state, tools_by_name)
    print(f"After tool call: {state}")
    print(f"Last message: {state['messages'][-1]}")
    
    # 8. Check where to go next
    next_step = router(state)
    print(f"Next step: {next_step}")
    
    # 9. Step 3: Call the model again to generate final response
    print("\nSTEP 3: Final model call")
    state = call_model(state, mock_llm.bind_tools(tools), system_prompt)
    print(f"Final state: {state}")
    print(f"Last message: {state['messages'][-1]}")
    
    # 10. Check if we're done
    next_step = router(state)
    print(f"Final step: {next_step}")
    
    # 11. Extract full conversation for review
    print("\nFULL CONVERSATION:")
    for i, msg in enumerate(state["messages"]):
        msg_type = type(msg).__name__
        print(f"{i+1}. {msg_type}: {msg.content}")
    
    return state

if __name__ == "__main__":
    print("Running intermediate state observation for custom agent...")
    print("="*80)
    
    # Run the manual step test
    manual_step_execution()
    
    print("="*80)
    print("Test completed successfully!") 