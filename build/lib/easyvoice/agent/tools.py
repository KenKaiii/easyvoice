"""Tool system for function calling without langchain dependency"""

import asyncio
import logging
import psutil
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable, Union
import inspect

from easyvoice.config.settings import Settings

logger = logging.getLogger(__name__)


class Tool:
    """Individual tool definition"""
    
    def __init__(self, name: str, func: Callable, description: str = "", timeout: int = 30):
        """Initialize tool
        
        Args:
            name: Tool name
            func: Function to execute
            description: Tool description
            timeout: Execution timeout in seconds
        """
        self.name = name
        self.func = func
        self.description = description
        self.timeout = timeout
        self.is_async = asyncio.iscoroutinefunction(func)
        
        # Extract function signature for parameter validation
        self.signature = inspect.signature(func)
    
    async def execute(self, **kwargs) -> Any:
        """Execute the tool with given parameters
        
        Args:
            **kwargs: Tool parameters
            
        Returns:
            Tool execution result
        """
        try:
            # Validate parameters against signature
            bound_args = self.signature.bind(**kwargs)
            bound_args.apply_defaults()
            
            if self.is_async:
                return await asyncio.wait_for(
                    self.func(**bound_args.arguments),
                    timeout=self.timeout
                )
            else:
                # Run sync function in thread pool
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None,
                    lambda: self.func(**bound_args.arguments)
                )
                
        except asyncio.TimeoutError:
            logger.error(f"Tool '{self.name}' timed out after {self.timeout}s")
            raise
        except Exception as e:
            logger.error(f"Tool '{self.name}' failed: {e}")
            raise


class ToolsManager:
    """Manages tool registration and execution"""
    
    def __init__(self, settings: Settings):
        """Initialize tools manager
        
        Args:
            settings: EasyVoice configuration settings
        """
        self.settings = settings
        self.tools: Dict[str, Tool] = {}
        self.healthy = True
        
        # Register built-in tools
        self._register_builtin_tools()
        
        logger.info("Tools manager initialized")
    
    def _register_builtin_tools(self) -> None:
        """Register built-in tools"""
        # Time tool
        self.register_tool(
            "time",
            self._get_current_time,
            "Get current date and time"
        )
        
        # System info tool
        self.register_tool(
            "system_info",
            self._get_system_info,
            "Get system information (memory, CPU, disk)"
        )
        
        # Weather tool (mock for now)
        self.register_tool(
            "weather",
            self._get_weather,
            "Get weather information for a location"
        )
    
    def register_tool(self, name: str, func: Callable, description: str = "", timeout: int = 30) -> None:
        """Register a new tool
        
        Args:
            name: Tool name (unique identifier)
            func: Function to execute
            description: Tool description
            timeout: Execution timeout in seconds
        """
        if name in self.tools:
            logger.warning(f"Tool '{name}' already exists, overwriting")
        
        self.tools[name] = Tool(name, func, description, timeout)
        logger.info(f"Registered tool: {name}")
    
    async def execute_tool(self, name: str, parameters: Optional[Dict[str, Any]] = None, timeout: Optional[int] = None) -> Any:
        """Execute a tool by name
        
        Args:
            name: Tool name
            parameters: Tool parameters
            timeout: Override timeout
            
        Returns:
            Tool execution result
            
        Raises:
            KeyError: If tool doesn't exist
            Exception: Tool execution errors
        """
        if name not in self.tools:
            raise KeyError(f"Tool '{name}' not found")
        
        tool = self.tools[name]
        if timeout:
            tool.timeout = timeout
        
        params = parameters or {}
        
        logger.info(f"Executing tool '{name}' with params: {params}")
        
        try:
            result = await tool.execute(**params)
            logger.info(f"Tool '{name}' completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Tool '{name}' failed: {e}")
            raise
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tool names
        
        Returns:
            List of tool names
        """
        return list(self.tools.keys())
    
    def list_tools(self) -> List[Dict[str, str]]:
        """Get detailed list of available tools
        
        Returns:
            List of tool dictionaries with name, description
        """
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "timeout": tool.timeout,
                "is_async": tool.is_async
            }
            for tool in self.tools.values()
        ]
    
    def is_healthy(self) -> bool:
        """Check if tools manager is healthy
        
        Returns:
            True if healthy
        """
        return self.healthy
    
    # Built-in tool implementations
    def _get_current_time(self) -> str:
        """Get current time"""
        now = datetime.now()
        return now.strftime("%I:%M %p on %B %d, %Y")
    
    def _get_system_info(self, info_type: str = "memory") -> str:
        """Get system information
        
        Args:
            info_type: Type of info (memory, cpu, disk, performance)
            
        Returns:
            System information string
        """
        try:
            if info_type.lower() == "memory":
                memory = psutil.virtual_memory()
                return f"Memory usage: {memory.percent}% ({memory.used // (1024**3):.1f}GB / {memory.total // (1024**3):.1f}GB)"
            
            elif info_type.lower() == "cpu":
                cpu_percent = psutil.cpu_percent(interval=1)
                cpu_count = psutil.cpu_count()
                return f"CPU usage: {cpu_percent}% ({cpu_count} cores)"
            
            elif info_type.lower() == "disk":
                disk = psutil.disk_usage('/')
                return f"Disk usage: {disk.percent}% ({disk.used // (1024**3):.1f}GB / {disk.total // (1024**3):.1f}GB)"
            
            elif info_type.lower() == "performance":
                memory = psutil.virtual_memory()
                cpu_percent = psutil.cpu_percent(interval=0.5)
                return f"System performance - CPU: {cpu_percent}%, Memory: {memory.percent}%"
            
            else:
                return f"Unknown info type: {info_type}. Available: memory, cpu, disk, performance"
                
        except Exception as e:
            return f"Error getting system info: {e}"
    
    def _get_weather(self, location: str = "current location") -> str:
        """Get weather information (mock implementation)
        
        Args:
            location: Location to get weather for
            
        Returns:
            Weather information string
        """
        # Mock weather data - in real implementation would call weather API
        mock_weather_data = {
            "paris": "Weather in Paris: 18°C, partly cloudy with light rain expected",
            "london": "Weather in London: 15°C, overcast with occasional drizzle",
            "new york": "Weather in New York: 22°C, sunny with light breeze",
            "tokyo": "Weather in Tokyo: 25°C, humid with chance of thunderstorms"
        }
        
        location_key = location.lower()
        if location_key in mock_weather_data:
            return mock_weather_data[location_key]
        else:
            return f"Weather information for {location}: 20°C, partly cloudy (mock data - API integration needed)"


def create_tool_executor(tools_manager: ToolsManager) -> Callable[[str, str], str]:
    """Create tool executor function for LLM integration
    
    Args:
        tools_manager: Configured tools manager
        
    Returns:
        Function that can be called by LLM to execute tools
    """
    async def execute_tool_for_llm(tool_name: str, parameters_str: str = "{}") -> str:
        """Execute tool for LLM
        
        Args:
            tool_name: Name of tool to execute
            parameters_str: JSON string of parameters
            
        Returns:
            Tool result as string
        """
        try:
            import json
            parameters = json.loads(parameters_str) if parameters_str.strip() else {}
            
            result = await tools_manager.execute_tool(tool_name, parameters)
            return str(result)
            
        except Exception as e:
            return f"Error executing tool '{tool_name}': {e}"
    
    return execute_tool_for_llm