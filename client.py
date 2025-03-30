import asyncio
import sys
from typing import Optional, List, Dict, Any
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters, Tool
from mcp.client.stdio import stdio_client

from anthropic import Anthropic
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

class MCPClient:
    """Model Context Protocol client for connecting to MCP servers and using them with Claude."""
    
    def __init__(self, model: str = None):
        """Initialize the MCP client.
        
        Args:
            model: The Claude model to use for generating responses
        """
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        
        # Check for API key
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found. Please add it to your .env file.")
        
        self.anthropic = Anthropic(api_key=api_key)
        
        # Get model from .env or use default
        self.model = model or os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-20241022")
        self.available_tools: List[Dict[str, Any]] = []
    
    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server.
        
        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")
        
        # Create correct command based on file type
        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )
        
        # Set up connection to server
        try:
            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
            self.stdio, self.write = stdio_transport
            self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
            
            # Initialize the session
            await self.session.initialize()
            
            # List available tools
            response = await self.session.list_tools()
            tools = response.tools
            self.available_tools = [{
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema
            } for tool in tools]
            
            print("\nConnected to server with tools:", [tool["name"] for tool in self.available_tools])
        except Exception as e:
            print(f"Error connecting to server: {str(e)}")
            await self.cleanup()
            sys.exit(1)
    
    async def process_query(self, query: str) -> str:
        """Process a query using Claude and available tools.
        
        Args:
            query: The user's query text
            
        Returns:
            The response from Claude after processing tools
        """
        if not self.session:
            raise ValueError("Not connected to a server. Call connect_to_server first.")
        
        messages = [
            {
                "role": "user",
                "content": query
            }
        ]
        
        # Initial Claude API call
        try:
            response = self.anthropic.messages.create(
                model=self.model,
                max_tokens=1000,
                messages=messages,
                tools=self.available_tools
            )
        except Exception as e:
            return f"Error getting response from Claude: {str(e)}"
        
        # Process response and handle tool calls
        final_text = []
        
        # Process all tool calls and responses in a loop until no more tool calls
        while True:
            has_tool_calls = False
            assistant_message_content = []
            
            for content in response.content:
                if content.type == 'text':
                    final_text.append(content.text)
                    assistant_message_content.append(content)
                elif content.type == 'tool_use':
                    has_tool_calls = True
                    tool_name = content.name
                    tool_args = content.input
                    
                    try:
                        # Execute tool call
                        print(f"\nCalling tool {tool_name} with args {tool_args}...")
                        result = await self.session.call_tool(tool_name, tool_args)
                        print(f"Tool response received: {result.content[:100]}..." if len(result.content) > 100 else f"Tool response received: {result.content}")
                        
                        final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")
                        
                        assistant_message_content.append(content)
                        messages.append({
                            "role": "assistant",
                            "content": assistant_message_content
                        })
                        messages.append({
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": content.id,
                                    "content": result.content
                                }
                            ]
                        })
                        
                        # Get next response from Claude
                        print("Getting Claude's response to tool result...")
                        response = self.anthropic.messages.create(
                            model=self.model,
                            max_tokens=1000,
                            messages=messages,
                            tools=self.available_tools
                        )
                        
                        # Break the inner loop to process the new response
                        break
                    except Exception as e:
                        final_text.append(f"Error executing tool {tool_name}: {str(e)}")
            
            # If no tool calls were made in this iteration, we're done
            if not has_tool_calls:
                break
        
        return "\n".join(final_text)
    
    async def chat_loop(self):
        """Run an interactive chat loop."""
        if not self.session:
            raise ValueError("Not connected to a server. Call connect_to_server first.")
        
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")
        
        while True:
            try:
                query = input("\nQuery: ").strip()
                
                if query.lower() in ('quit', 'exit', 'q'):
                    break
                
                response = await self.process_query(query)
                print("\n" + response)
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"\nError: {str(e)}")
    
    async def cleanup(self):
        """Clean up resources."""
        await self.exit_stack.aclose()
        print("\nClient resources cleaned up.")

async def main():
    """Main entry point for the MCP client."""
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        print("Example: python client.py ./weather_server.py")
        sys.exit(1)
    
    server_path = sys.argv[1]
    
    # Optional model parameter from command line (overrides .env)
    model = sys.argv[2] if len(sys.argv) > 2 else None
    
    client = MCPClient(model=model)
    try:
        await client.connect_to_server(server_path)
        await client.chat_loop()
    except KeyboardInterrupt:
        print("\nInterrupted by user. Cleaning up...")
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())