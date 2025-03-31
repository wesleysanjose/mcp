from typing import List, Dict, Any
from anthropic import Anthropic
from mcp import Tool
from llm_interface import LLMInterface

class ClaudeLLM(LLMInterface):
    """Claude LLM provider implementation."""
    
    def __init__(self, api_key: str, model: str):
        """Initialize the Claude LLM provider.
        
        Args:
            api_key: Anthropic API key
            model: Claude model name
        """
        self.client = Anthropic(api_key=api_key)
        self.model = model
    
    def format_tools(self, tools: List[Tool]) -> List[Dict[str, Any]]:
        """Format tools for Claude.
        
        Args:
            tools: List of MCP tools
            
        Returns:
            Formatted tools for Claude
        """
        return [{
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.inputSchema
        } for tool in tools]
    
    async def generate_response(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a response from Claude.
        
        Args:
            messages: List of conversation messages
            tools: List of available tools
            
        Returns:
            Response from Claude
        """
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            messages=messages,
            tools=tools
        )
        return response
    
    def process_response(self, response: Any) -> Dict[str, Any]:
        """Process the response from Claude.
        
        Args:
            response: Response from Claude
            
        Returns:
            Processed response with text and tool calls
        """
        text_content = []
        tool_calls = []
        
        for content in response.content:
            if content.type == 'text':
                text_content.append(content.text)
            elif content.type == 'tool_use':
                tool_calls.append({
                    'id': content.id,
                    'name': content.name,
                    'args': content.input
                })
        
        return {
            'text': '\n'.join(text_content),
            'tool_calls': tool_calls,
            'raw_response': response
        }
    
    def format_tool_result(self, tool_call: Dict[str, Any], result: str) -> Dict[str, Any]:
        """Format a tool result for Claude.
        
        Args:
            tool_call: Tool call from Claude
            result: Result from the tool
            
        Returns:
            Formatted tool result for Claude
        """
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_call['id'],
                    "content": result
                }
            ]
        }