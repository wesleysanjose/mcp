from typing import List, Dict, Any
from openai import OpenAI
from mcp import Tool
from llm_interface import LLMInterface
import json

class OpenAILLM(LLMInterface):
    """OpenAI LLM provider implementation."""
    
    def __init__(self, api_key: str, model: str, base_url: str = None):
        """Initialize the OpenAI LLM provider.
        
        Args:
            api_key: OpenAI API key
            model: OpenAI model name
            base_url: Base URL for local OpenAI-compatible API server
        """
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        
        self.client = OpenAI(**client_kwargs)
        self.model = model
        # Check if we're using a Qwen model
        self.is_qwen = "qwen" in model.lower()
    
    def format_tools(self, tools: List[Tool]) -> List[Dict[str, Any]]:
        """Format tools for OpenAI.
        
        Args:
            tools: List of MCP tools
            
        Returns:
            Formatted tools for OpenAI
        """
        return [{
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema
            }
        } for tool in tools]
    
    async def generate_response(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a response from OpenAI.
        
        Args:
            messages: List of conversation messages
            tools: List of available tools
            
        Returns:
            Response from OpenAI
        """
        # For Qwen models, we need to be more explicit about tool usage
        if self.is_qwen:
            # Add a system message to encourage tool usage
            has_system = any(msg.get("role") == "system" for msg in messages)
            if not has_system:
                system_message = {
                    "role": "system",
                    "content": "You are a helpful assistant with access to tools. When a user asks about stocks or financial information, USE THE TOOLS provided to get accurate information. DO NOT make up information. Always call the appropriate tool when asked about stock data."
                }
                messages = [system_message] + messages
            
            # Try with more explicit tool_choice
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    temperature=0.2  # Lower temperature to make tool usage more likely
                )
                return response
            except Exception as e:
                print(f"Error with tool_choice='auto': {e}")
                # Fall back to a simpler approach if the above fails
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools
                )
                return response
        else:
            # Standard OpenAI approach
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )
            return response
    
    def process_response(self, response: Any) -> Dict[str, Any]:
        """Process the response from OpenAI.
        
        Args:
            response: Response from OpenAI
            
        Returns:
            Processed response with text and tool calls
        """
        message = response.choices[0].message
        tool_calls = []
        
        # Check if the model returned tool calls
        if hasattr(message, 'tool_calls') and message.tool_calls:
            for tool_call in message.tool_calls:
                try:
                    args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    # Handle case where arguments aren't valid JSON
                    args = {"error": "Invalid JSON in arguments", "raw": tool_call.function.arguments}
                
                tool_calls.append({
                    'id': tool_call.id,
                    'name': tool_call.function.name,
                    'args': args
                })
        else:
            # For Qwen models, we might need to parse the content for tool calls
            if self.is_qwen and message.content:
                # Try to detect if the model is trying to use a tool in its text
                content = message.content.lower()
                if any(keyword in content for keyword in ["tool", "function", "api", "get_stock", "stock data"]):
                    # Extract potential tool calls from text
                    for tool_name in ["get_stock_data", "get_stock_analysis"]:
                        if tool_name.lower() in content:
                            # Look for ticker symbols in the content
                            import re
                            ticker_match = re.search(r'\b[A-Z]{1,5}\b', message.content)
                            ticker = ticker_match.group(0) if ticker_match else "PYPL"
                            
                            print(f"Detected implicit tool call to {tool_name} for ticker {ticker}")
                            
                            # Create a synthetic tool call
                            tool_calls.append({
                                'id': f"synthetic-{tool_name}",
                                'name': tool_name,
                                'args': {"ticker": ticker}
                            })
        
        return {
            'text': message.content or '',
            'tool_calls': tool_calls,
            'raw_response': response,
            'message': message
        }
    
    def format_tool_result(self, tool_call: Dict[str, Any], result: str) -> Dict[str, Any]:
        """Format a tool result for OpenAI.
        
        Args:
            tool_call: Tool call from OpenAI
            result: Result from the tool
            
        Returns:
            Formatted tool result for OpenAI
        """
        return {
            "role": "tool",
            "tool_call_id": tool_call['id'],
            "name": tool_call['name'],
            "content": result
        }