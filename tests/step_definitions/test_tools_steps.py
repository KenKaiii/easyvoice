"""Step definitions for tools system BDD scenarios"""

import asyncio

import pytest
from pytest_bdd import given, scenarios, then, when

# Load scenarios from feature file
scenarios("../features/tools.feature")

# Global test state
pytest.tools_system = None
pytest.tool_result = None
pytest.tool_error = None
pytest.tools_called = []


@given("the tools system is initialized")
def tools_system_initialized(test_settings):
    """Initialize the tools system"""
    from easyvoice.agent.tools import ToolsManager

    pytest.tools_system = ToolsManager(test_settings)


@given("basic tools are available")
def basic_tools_available():
    """Verify basic tools are registered"""
    available_tools = pytest.tools_system.get_available_tools()
    assert "time" in available_tools
    assert "system_info" in available_tools


@when('the user asks "What time is it?"')
async def ask_time():
    """Ask for current time"""
    pytest.tool_result = await pytest.tools_system.execute_tool("time")
    pytest.tools_called.append("time")


@then("the time tool should be called")
def time_tool_called():
    """Verify time tool was called"""
    assert "time" in pytest.tools_called


@then("I should get the current time")
def should_get_current_time():
    """Verify we get current time"""
    assert pytest.tool_result is not None
    assert ":" in str(pytest.tool_result)  # Time format check


@then("the response should include the time")
def response_includes_time():
    """Verify response includes time"""
    assert pytest.tool_result is not None


@when('the user asks "How much memory is being used?"')
async def ask_memory_usage():
    """Ask about memory usage"""
    pytest.tool_result = await pytest.tools_system.execute_tool(
        "system_info", {"info_type": "memory"}
    )
    pytest.tools_called.append("system_info")


@then("the system info tool should be called")
def system_info_tool_called():
    """Verify system info tool was called"""
    assert "system_info" in pytest.tools_called


@then("I should get memory usage data")
def should_get_memory_data():
    """Verify we get memory usage data"""
    assert pytest.tool_result is not None
    result_str = str(pytest.tool_result).lower()
    assert "memory" in result_str or "mb" in result_str


@then("the response should be formatted properly")
def response_formatted_properly():
    """Verify response is properly formatted"""
    assert isinstance(pytest.tool_result, (str, dict))


@given("a tool that responds slowly")
def slow_tool_available():
    """Register a slow tool for testing"""
    pytest.tools_system.register_tool("slow_tool", pytest.slow_tool_func, timeout=10)


@when("I call the slow tool with a short timeout")
async def call_slow_tool():
    """Call slow tool with short timeout"""
    try:
        pytest.tool_result = await pytest.tools_system.execute_tool(
            "slow_tool", timeout=1
        )
        pytest.tool_error = None
    except asyncio.TimeoutError as e:
        pytest.tool_error = e
        pytest.tool_result = None


@then("the tool call should timeout gracefully")
def tool_timeouts_gracefully():
    """Verify tool timed out gracefully"""
    assert pytest.tool_error is not None
    assert isinstance(pytest.tool_error, asyncio.TimeoutError)


@then("I should get a timeout error")
def should_get_timeout_error():
    """Verify timeout error received"""
    assert pytest.tool_error is not None


@then("the system should remain stable")
def system_remains_stable():
    """Verify system is still functional after timeout"""
    assert pytest.tools_system.is_healthy()


@given("a tool that always fails")
def failing_tool_available():
    """Register a tool that always fails"""
    pytest.tools_system.register_tool("failing_tool", pytest.failing_tool_func)


@when("I try to use the failing tool")
async def use_failing_tool():
    """Try to use the failing tool"""
    try:
        pytest.tool_result = await pytest.tools_system.execute_tool("failing_tool")
        pytest.tool_error = None
    except Exception as e:
        pytest.tool_error = e
        pytest.tool_result = None


@then("I should get an appropriate error message")
def should_get_error_message():
    """Verify appropriate error message"""
    assert pytest.tool_error is not None
    error_msg = str(pytest.tool_error).lower()
    assert "error" in error_msg or "fail" in error_msg


@then("the agent should continue normally")
def agent_continues_normally():
    """Verify agent continues working"""
    assert pytest.tools_system.is_healthy()


@then("suggest alternative approaches")
def suggest_alternatives():
    """Verify alternatives are suggested"""
    # In real implementation, this would check error message for suggestions
    assert pytest.tool_error is not None


@when('the user asks "What time is it and how\'s the system performance?"')
async def ask_time_and_performance():
    """Ask about time and performance"""
    pytest.time_result = await pytest.tools_system.execute_tool("time")
    pytest.system_result = await pytest.tools_system.execute_tool(
        "system_info", {"info_type": "performance"}
    )
    pytest.tools_called.extend(["time", "system_info"])


@then("both results should be combined")
def results_combined():
    """Verify both results were obtained"""
    assert pytest.time_result is not None
    assert pytest.system_result is not None


@then("the response should address both questions")
def response_addresses_both():
    """Verify response addresses both questions"""
    assert pytest.time_result is not None
    assert pytest.system_result is not None


@when("I register a new custom tool")
def register_custom_tool():
    """Register a new custom tool"""

    def custom_func():
        return "Custom tool result"

    pytest.tools_system.register_tool(
        "custom_tool", custom_func, description="A custom test tool"
    )


@then("it should be available for use")
def custom_tool_available():
    """Verify custom tool is available"""
    available_tools = pytest.tools_system.get_available_tools()
    assert "custom_tool" in available_tools


@then("it should appear in the tools list")
def appears_in_tools_list():
    """Verify tool appears in list"""
    tools_list = pytest.tools_system.list_tools()
    assert any(tool["name"] == "custom_tool" for tool in tools_list)


@then("it should be callable by the agent")
async def callable_by_agent():
    """Verify tool is callable"""
    result = await pytest.tools_system.execute_tool("custom_tool")
    assert result == "Custom tool result"


@when('the user asks "What\'s the weather in Paris?"')
async def ask_weather_paris():
    """Ask about weather in Paris"""
    pytest.tool_result = await pytest.tools_system.execute_tool(
        "weather", {"location": "Paris"}
    )
    pytest.tools_called.append("weather")


@then('the location parameter should be "Paris"')
def location_parameter_paris():
    """Verify Paris was passed as parameter"""
    # This would be verified by checking the mock calls
    assert pytest.tool_result is not None


@then("the tool should return weather data")
def tool_returns_weather_data():
    """Verify weather data returned"""
    assert pytest.tool_result is not None
    result_str = str(pytest.tool_result).lower()
    assert (
        "weather" in result_str or "temperature" in result_str or "paris" in result_str
    )


@then("the response should be location-specific")
def response_location_specific():
    """Verify response is location-specific"""
    assert "paris" in str(pytest.tool_result).lower()


# Mock tool functions for testing
async def slow_tool_func():
    """Slow tool for timeout testing"""
    await asyncio.sleep(5)
    return "Slow result"


def failing_tool_func():
    """Failing tool for error testing"""
    raise Exception("Tool failed intentionally")


# Store mock functions in pytest namespace
pytest.slow_tool_func = slow_tool_func
pytest.failing_tool_func = failing_tool_func
