"""
Custom ReAct agent implementation for MCP Solver.

This module provides a ReAct agent pattern that works with MCP protocol tools
while maintaining compatibility with MCP error handling requirements.
"""

import json
from typing import Any, Dict, List, Optional, Callable

from anthropic import Anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPReactAgent:
    """
    A ReAct agent implementation optimized for MCP constraint solving.
    
    This agent follows the ReAct pattern (Reasoning + Acting) to solve problems
    using MCP tools. It manages the interaction loop between the LLM and tools,
    handling tool calls, error processing, and response formatting.
    """
    
    def __init__(
        self, 
        llm: Any,
        server_command: str,
        system_message: Optional[str] = None,
        verbose: bool = False
    ):
        """
        Initialize the MCP ReAct agent.
        
        Args:
            llm: The language model to use (typically Claude)
            server_command: The command to start the MCP server
            system_message: Optional system message to include in prompts
            verbose: Whether to enable verbose logging
        """
        self.llm = llm
        self.server_command = server_command
        self.system_message = system_message
        self.verbose = verbose
        self.session = None
        self.tools = {}
        
        # Set up logging functions
        self.log_system = print if verbose else lambda *args, **kwargs: None
        self.set_system_title = lambda *args, **kwargs: None
    
    def connect(self):
        """Establish connection to the MCP server and load tools."""
        if self.session is not None:
            return
            
        # Create server parameters for stdio connection
        server_params = StdioServerParameters(command=self.server_command)
        
        # Create client session
        self.log_system(f"Connecting to MCP server with command: {self.server_command}")
        self.session = stdio_client(server_params)
        
        # Load available tools
        self.tools = self._load_and_wrap_tools()
        self.log_system(f"Loaded {len(self.tools)} tools: {', '.join(self.tools.keys())}")
    
    def _load_and_wrap_tools(self):
        """Load tools from the MCP session and wrap them with tracking."""
        if not self.session:
            raise ValueError("Session not initialized. Call connect() first.")
            
        raw_tools = self.session.load_mcp_tools()
        
        # Wrap tools with tracking and error handling
        wrapped_tools = {}
        for name, tool in raw_tools.items():
            wrapped_tools[name] = self._wrap_tool(tool, name)
            
        return wrapped_tools
    
    def _wrap_tool(self, tool, tool_name):
        """Wrap a tool with logging and error handling."""
        orig_invoke = tool.invoke
        
        def new_invoke(call_args, config=None):
            args_only = call_args.get("args", {})
            
            args_str = json.dumps(args_only, indent=2).strip()
            if args_str.startswith("{") and args_str.endswith("}"):
                args_str = args_str[1:-1].strip()
            
            self.set_system_title(f"tool: {tool_name}")
            self.log_system(f"{tool_name} called with args: {args_str}")
            
            try:
                result = orig_invoke(call_args, config)
                formatted = self._format_tool_output(result)
                self.log_system(f"{tool_name} output: {formatted}")
                return result
            except Exception as e:
                error_msg = f"Tool execution failed: {str(e)}"
                self.log_system(f"Error: {error_msg}")
                
                # Return MCP-compliant error format
                return {
                    "isError": True,
                    "content": [
                        {
                            "type": "text",
                            "text": f"Error: {error_msg}"
                        }
                    ]
                }
        
        return new_invoke
    
    def _format_tool_output(self, result):
        """Format tool output for logging."""
        if isinstance(result, dict):
            return json.dumps(result, indent=2)
        return str(result)
    
    def _format_as_tool_message(self, result, tool_call_id):
        """Format a tool result as a message for the LLM."""
        content = result.get("content", [{"type": "text", "text": str(result)}])
        is_error = result.get("isError", False)
        
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
            "is_error": is_error
        }
    
    def invoke(self, query: str) -> str:
        """
        Execute the ReAct agent on a user query.
        
        Args:
            query: The user's natural language query
            
        Returns:
            The final response from the LLM
        """
        self.connect()  # Ensure connection is established
        
        # Initialize messages with user query
        messages = []
        
        # Add system message if provided
        if self.system_message:
            messages.append({"role": "system", "content": self.system_message})
        
        # Add user query
        messages.append({"role": "user", "content": query})
        
        # Start ReAct loop
        while True:
            # Get LLM response
            self.log_system("Sending request to LLM...")
            response = self.llm.invoke(messages)
            
            # Add response to conversation
            messages.append(response)
            
            # Check if response contains tool calls
            if not hasattr(response, 'tool_calls') or not response.tool_calls:
                self.log_system("No tool calls, returning final response")
                return response.content
                
            # Process each tool call
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_call_id = tool_call["id"]
                
                if tool_name not in self.tools:
                    error_message = f"Unknown tool: {tool_name}"
                    self.log_system(f"Error: {error_message}")
                    
                    tool_result = {
                        "isError": True,
                        "content": [{"type": "text", "text": f"Error: {error_message}"}]
                    }
                else:
                    # Execute the tool
                    tool_result = self.tools[tool_name](tool_call)
                
                # Format and add tool result to messages
                tool_message = self._format_as_tool_message(tool_result, tool_call_id)
                messages.append(tool_message)
                
                self.log_system(f"Tool {tool_name} executed, added result to conversation")
        
        # This point should not be reached as we return when no tool calls are made
        return "Error: ReAct loop terminated unexpectedly"


def create_mcp_react_agent(
    llm, 
    server_command, 
    system_message=None, 
    verbose=False
):
    """
    Factory function to create an MCP ReAct agent.
    
    Args:
        llm: Language model to use
        server_command: Command to start the MCP server
        system_message: Optional system message to include
        verbose: Whether to enable verbose output
        
    Returns:
        A function that takes a query and returns a response
    """
    agent = MCPReactAgent(llm, server_command, system_message, verbose)
    
    def invoke_agent(query):
        return agent.invoke(query)
    
    return invoke_agent 