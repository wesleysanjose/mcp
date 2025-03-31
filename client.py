import asyncio
import sys
from typing import Optional, List, Dict, Any
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters, Tool
from mcp.client.stdio import stdio_client

from llm_interface import LLMInterface
from claude_llm import ClaudeLLM
from openai_llm import OpenAILLM

from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

class MCPClient:
    """Model Context Protocol client for connecting to MCP servers and using them with LLMs."""
    
    def __init__(self, llm_provider: str = "claude", model: str = None):
        """Initialize the MCP client.
        
        Args:
            llm_provider: The LLM provider to use ('claude' or 'openai')
            model: The model to use for generating responses
        """
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        
        # Set up LLM provider
        self.llm_provider = llm_provider.lower()
        
        if self.llm_provider == "claude":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not found. Please add it to your .env file.")
            
            model = model or os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-20241022")
            self.llm = ClaudeLLM(api_key, model)
            
        elif self.llm_provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY", "dummy-api-key")  # Allow dummy key for local servers
            base_url = os.getenv("OPENAI_BASE_URL")
            
            model = model or os.getenv("OPENAI_MODEL", "gpt-4o")
            self.llm = OpenAILLM(api_key, model, base_url)
            
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}. Use 'claude' or 'openai'.")
        
        self.available_tools: List[Dict[str, Any]] = []
        self.raw_tools: List[Tool] = []
    
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
            self.raw_tools = response.tools
            self.available_tools = self.llm.format_tools(self.raw_tools)
            
            print("\nConnected to server with tools:", [tool.name for tool in self.raw_tools])
        except Exception as e:
            print(f"Error connecting to server: {str(e)}")
            await self.cleanup()
            sys.exit(1)
    
    async def process_query(self, query: str) -> str:
        """Process a query using the LLM and available tools.
        
        Args:
            query: The user's query text
            
        Returns:
            The response from the LLM after processing tools
        """
        if not self.session:
            raise ValueError("Not connected to a server. Call connect_to_server first.")
        
        messages = [
            {
                "role": "user",
                "content": query
            }
        ]
        
        # Initial LLM API call
        try:
            response = await self.llm.generate_response(messages, self.available_tools)
        except Exception as e:
            return f"Error getting response from LLM: {str(e)}"
        
        # Process response and handle tool calls
        final_text = []
        
        # Process all tool calls and responses in a loop until no more tool calls
        while True:
            processed_response = self.llm.process_response(response)
            
            if processed_response['text']:
                final_text.append(processed_response['text'])
            
            if not processed_response['tool_calls']:
                # No more tool calls, we're done
                break
            
            # Process tool calls
            for tool_call in processed_response['tool_calls']:
                tool_name = tool_call['name']
                tool_args = tool_call['args']
                
                try:
                    # Execute tool call
                    print(f"\nCalling tool {tool_name} with args {tool_args}...")
                    result = await self.session.call_tool(tool_name, tool_args)
                    print(f"Tool response received: {result.content[:100]}..." if len(result.content) > 100 else f"Tool response received: {result.content}")
                    
                    final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")
                    
                    # Add assistant message to conversation
                    if self.llm_provider == "claude":
                        messages.append({
                            "role": "assistant",
                            "content": processed_response['raw_response'].content
                        })
                    else:  # openai
                        messages.append(processed_response['message'].model_dump())
                    
                    # Add tool result to conversation
                    messages.append(self.llm.format_tool_result(tool_call, result.content))
                    
                    # Get next response from LLM
                    print(f"Getting {self.llm_provider}'s response to tool result...")
                    response = await self.llm.generate_response(messages, self.available_tools)
                    
                    # Break to process the new response
                    break
                except Exception as e:
                    final_text.append(f"Error executing tool {tool_name}: {str(e)}")
        
        return "\n".join(final_text)
    
    async def chat_loop(self):
        """Run an interactive chat loop."""
        if not self.session:
            raise ValueError("Not connected to a server. Call connect_to_server first.")
        
        print(f"\nMCP Client Started with {self.llm_provider.capitalize()} LLM!")
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
        print("Usage: python client.py <path_to_server_script> [llm_provider] [model]")
        print("Example: python client.py ./weather_server.py claude")
        print("Example: python client.py ./weather_server.py openai gpt-4o")
        sys.exit(1)
    
    server_path = sys.argv[1]
    
    # Optional LLM provider parameter (defaults to claude)
    llm_provider = sys.argv[2] if len(sys.argv) > 2 else "claude"
    
    # Optional model parameter (overrides .env)
    model = sys.argv[3] if len(sys.argv) > 3 else None
    
    client = MCPClient(llm_provider=llm_provider, model=model)
    try:
        await client.connect_to_server(server_path)
        await client.chat_loop()
    except KeyboardInterrupt:
        print("\nInterrupted by user. Cleaning up...")
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())