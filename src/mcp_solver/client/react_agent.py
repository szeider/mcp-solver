"""
Custom ReAct agent implementation for MCP Solver.

This module provides a ReAct agent pattern that works with MCP protocol tools
while maintaining compatibility with MCP error handling requirements.
"""

import json
import asyncio
from typing import Any, Dict, List, Optional, Callable

from anthropic import Anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPReactAgent:
    """
    A ReAct agent implementation for MCP tools.
    
    This agent follows the ReAct pattern:
    1. Observe - Get a query or state
    2. Think - Process with LLM
    3. Act - Call tools based on LLM decision
    4. Repeat until satisfied
    """
    
    def __init__(
        self, 
        llm: Any,
        server_command: str,
        system_message: Optional[str] = None,
        verbose: bool = False
    ):
        """Initialize the agent with model and session."""
        self.llm = llm
        self.server_command = server_command
        self.session = None
        self.tools = {}
        self.system_message = system_message or """You are an expert constraint solver, trained to help users solve constraint programming problems.
Use the available tools to interact with a constraint solver and solve the problem described in the user's query.
Think step by step and use the tools effectively to build and solve the constraint model."""
        self.verbose = verbose
        self.messages = []
        self._initialize_messages()
    
    def _initialize_messages(self):
        """Initialize the message history with system message."""
        self.messages = [
            {"role": "system", "content": self.system_message}
        ]
    
    async def connect(self):
        """Establish connection to the MCP server and load tools."""
        if self.session is not None:
            return
            
        # Create server parameters for stdio connection
        server_params = StdioServerParameters(command=self.server_command)
        
        # Create client session
        self.log_system(f"Connecting to MCP server with command: {self.server_command}")
        
        # Set up the async connection
        try:
            # Open the stdio connection - this returns a context manager
            ctx = stdio_client(server_params)
            # Enter the context manager to get read/write pair
            read, write = await ctx.__aenter__()
            # Create a session with the read/write pair
            session_ctx = ClientSession(read, write)
            # Enter the session context
            self.session = await session_ctx.__aenter__()
            # Initialize the session
            await self.session.initialize()
            
            # Load available tools
            self.tools = await self._load_and_wrap_tools()
            self.log_system(f"Loaded {len(self.tools)} tools: {', '.join(self.tools.keys())}")
        except Exception as e:
            self.log_system(f"Error connecting to MCP server: {str(e)}")
            raise
    
    async def disconnect(self):
        """Close the connection to the MCP server."""
        if self.session:
            await self.session.__aexit__(None, None, None)
            self.session = None
    
    def run_async(self, coro):
        """Helper method to run async code from sync context."""
        return asyncio.run(coro)
    
    def connect_sync(self):
        """Synchronous wrapper for connect()."""
        return self.run_async(self.connect())
    
    async def _load_and_wrap_tools(self):
        """Load tools from the MCP session and wrap them with tracking."""
        if not self.session:
            raise ValueError("Session not initialized. Call connect() first.")
        
        # Get available tools from session
        tools_response = await self.session.list_tools()
        raw_tools = {}
        
        # Create a function for each tool
        for tool_info in tools_response.tools:
            name = tool_info.name
            
            async def make_tool_fn(tool_name):
                async def tool_fn(args):
                    tool_args = args.get("args", {})
                    result = await self.session.call_tool(tool_name, tool_args)
                    return result
                return tool_fn
            
            raw_tools[name] = await make_tool_fn(name)
        
        # Wrap tools with tracking and error handling
        wrapped_tools = {}
        for name, tool in raw_tools.items():
            wrapped_tools[name] = self._wrap_tool(tool, name)
            
        return wrapped_tools
    
    def _wrap_tool(self, tool_fn, tool_name: str):
        """Wrap a tool function with tracking and error handling."""
        async def wrapped_tool(args):
            try:
                # Log the tool call
                args_str = json.dumps(args) if isinstance(args, dict) else str(args)
                self.log_system(f"TOOL: {tool_name}: {tool_name} called with args: {args_str}")
                
                # Call the tool
                result = await tool_fn(args)
                
                # Log the result
                result_str = str(result)
                if len(result_str) > 100:
                    result_str = result_str[:97] + "..."
                self.log_system(f"TOOL: {tool_name}: {tool_name} output: {result_str}")
                
                return result
            except Exception as e:
                error_msg = f"Error calling tool {tool_name}: {str(e)}"
                self.log_system(error_msg)
                return {"error": error_msg}
        
        return wrapped_tool
    
    def log_system(self, message: str):
        """Log a system message."""
        if self.verbose:
            print(message)
    
    async def invoke(self, query: str):
        """
        Process a query using the ReAct pattern.
        
        Args:
            query: The user query to process
            
        Returns:
            The final response from the agent
        """
        # Ensure connection is established
        await self.connect()
        
        # Reset message history with system message
        self._initialize_messages()
        
        # Add the user query
        self.messages.append({"role": "user", "content": query})
        
        # Begin the ReAct loop
        max_iterations = 15  # Safety limit
        iterations = 0
        
        while iterations < max_iterations:
            iterations += 1
            
            # Get response from LLM
            self.log_system("Sending request to LLM...")
            llm_response = await self._get_llm_response()
            
            # Extract potential tool calls
            tool_calls = llm_response.get("tool_calls", [])
            
            # If no tool calls, we're done
            if not tool_calls:
                return llm_response.get("content", "No response generated.")
            
            # Process each tool call
            for tool_call in tool_calls:
                tool_name = tool_call.get("name")
                if tool_name not in self.tools:
                    self.log_system(f"Unknown tool requested: {tool_name}")
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.get("id"),
                        "name": tool_name,
                        "content": f"Error: Tool '{tool_name}' not found. Available tools: {', '.join(self.tools.keys())}"
                    })
                    continue
                
                # Call the tool
                tool_result = await self.tools[tool_name](tool_call)
                
                # Add the result to the messages
                self.messages.append({
                    "role": "tool", 
                    "tool_call_id": tool_call.get("id"),
                    "name": tool_name,
                    "content": str(tool_result)
                })
        
        # If we reach here, we hit the iteration limit
        return "Maximum number of iterations reached without a final answer."
    
    def invoke_sync(self, query: str):
        """Synchronous wrapper for invoke()."""
        return self.run_async(self.invoke(query))
    
    async def _get_llm_response(self):
        """Get a response from the LLM."""
        # Use Anthropic-compatible tool format
        tools_for_llm = []
        for name in self.tools:
            tools_for_llm.append({
                "name": name,
                "description": f"Call the {name} tool to interact with the constraint solver."
            })
        
        # Get response from LLM
        response = await self.llm.bind_tools(tools_for_llm).ainvoke(self.messages)
        
        # Add the response to the messages
        self.messages.append(response)
        
        return response

def create_mcp_react_agent(llm, server_command, system_message=None, verbose=False):
    """
    Create a ReAct agent for MCP tools.
    
    Args:
        llm: The language model to use
        server_command: The command to start the MCP server
        system_message: Optional system message for the agent
        verbose: Whether to log verbose output
        
    Returns:
        A function that takes a query and returns a response
    """
    agent = MCPReactAgent(llm, server_command, system_message, verbose)
    
    def invoke_agent(query):
        """Invoke the agent synchronously."""
        return agent.invoke_sync(query)
    
    return invoke_agent 