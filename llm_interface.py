import abc
from typing import List, Dict, Any
from mcp import Tool

class LLMInterface(abc.ABC):
    """Interface for LLM providers."""
    
    @abc.abstractmethod
    def __init__(self, api_key: str, model: str):
        """Initialize the LLM provider.
        
        Args:
            api_key: API key for the LLM provider
            model: Model name to use
        """
        pass
    
    @abc.abstractmethod
    def format_tools(self, tools: List[Tool]) -> List[Dict[str, Any]]:
        """Format tools for the specific LLM provider.
        
        Args:
            tools: List of MCP tools
            
        Returns:
            Formatted tools for the LLM provider
        """
        pass
    
    @abc.abstractmethod
    async def generate_response(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a response from the LLM.
        
        Args:
            messages: List of conversation messages
            tools: List of available tools
            
        Returns:
            Response from the LLM
        """
        pass
    
    @abc.abstractmethod
    def process_response(self, response: Any) -> Dict[str, Any]:
        """Process the response from the LLM.
        
        Args:
            response: Response from the LLM
            
        Returns:
            Processed response with text and tool calls
        """
        pass
    
    @abc.abstractmethod
    def format_tool_result(self, tool_call: Any, result: str) -> Dict[str, Any]:
        """Format a tool result for the LLM.
        
        Args:
            tool_call: Tool call from the LLM
            result: Result from the tool
            
        Returns:
            Formatted tool result for the LLM
        """
        pass