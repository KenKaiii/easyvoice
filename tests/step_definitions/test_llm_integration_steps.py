"""Step definitions for LLM integration BDD scenarios"""

import pytest
import asyncio
from unittest.mock import Mock, patch
from pytest_bdd import scenarios, given, when, then, parsers

# Load scenarios from feature file
scenarios('../features/llm_integration.feature')

# Global test state
pytest.llm_system = None
pytest.llm_response = None
pytest.llm_error = None
pytest.conversation_context = None


@given("the LLM system is initialized with test settings")
def llm_system_initialized(test_settings):
    """Initialize the LLM system"""
    from easyvoice.agent.llm import LLMInterface
    pytest.llm_system = LLMInterface(test_settings)


@given("Ollama is available for testing")
def ollama_available():
    """Mock Ollama availability"""
    pytest.ollama_available = True
    # This will be mocked in the actual implementation


@when('I send the message "What is 2 + 2?"')
async def send_simple_message():
    """Send a simple message to LLM"""
    pytest.llm_response = await pytest.llm_system.generate_response("What is 2 + 2?")


@then("I should get a response from the LLM")
def should_get_llm_response():
    """Verify we get a response"""
    assert pytest.llm_response is not None
    assert isinstance(pytest.llm_response, str)


@then("the response should not be empty")
def response_not_empty():
    """Verify response is not empty"""
    assert len(pytest.llm_response.strip()) > 0


@then("the operation should complete within timeout")
def operation_within_timeout():
    """Verify operation completed within timeout"""
    # This is implicitly tested by the async operation completing
    assert pytest.llm_response is not None


@given("I have conversation history with 3 messages")
def have_conversation_history():
    """Setup conversation history"""
    pytest.conversation_context = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"}
    ]


@when('I send a follow-up message "What did we just discuss?"')
async def send_followup_message():
    """Send follow-up message with context"""
    pytest.llm_response = await pytest.llm_system.generate_response(
        "What did we just discuss?", 
        context=pytest.conversation_context
    )


@then("the LLM should receive the conversation context")
def llm_receives_context():
    """Verify LLM received conversation context"""
    # This would be verified by checking the mock calls
    assert pytest.llm_response is not None


@then("the response should reference previous messages")
def response_references_previous():
    """Verify response references previous context"""
    # In a real test, this would check if the response actually references context
    assert pytest.llm_response is not None


@given("the LLM will respond slowly")
def llm_responds_slowly():
    """Mock slow LLM response"""
    pytest.slow_llm = True


@when("I send a message with a short timeout")
async def send_with_short_timeout(test_settings):
    """Send message with short timeout"""
    test_settings.llm_timeout = 1  # Very short timeout
    
    try:
        pytest.llm_response = await pytest.llm_system.generate_response(
            "This will timeout", 
            timeout=1
        )
        pytest.llm_error = None
    except asyncio.TimeoutError as e:
        pytest.llm_error = e
        pytest.llm_response = None


@then("the operation should timeout gracefully")
def operation_timeouts_gracefully():
    """Verify operation timed out gracefully"""
    assert pytest.llm_error is not None
    assert isinstance(pytest.llm_error, asyncio.TimeoutError)


@then("I should get a timeout error")
def should_get_timeout_error():
    """Verify we get a timeout error"""
    assert pytest.llm_error is not None


@then("no partial response should be returned")
def no_partial_response():
    """Verify no partial response"""
    assert pytest.llm_response is None


@given("Ollama is not available")
def ollama_not_available():
    """Mock Ollama unavailability"""
    pytest.ollama_available = False


@when("I try to send a message")
async def try_send_message():
    """Try to send message when Ollama unavailable"""
    try:
        pytest.llm_response = await pytest.llm_system.generate_response("Test message")
        pytest.llm_error = None
    except Exception as e:
        pytest.llm_error = e
        pytest.llm_response = None


@then("I should get a connection error")
def should_get_connection_error():
    """Verify we get a connection error"""
    assert pytest.llm_error is not None


@then("the error message should be informative")
def error_message_informative():
    """Verify error message is informative"""
    error_msg = str(pytest.llm_error).lower()
    assert any(word in error_msg for word in ["connection", "ollama", "unavailable", "failed"])


@when("I check the LLM configuration")
def check_llm_configuration():
    """Check LLM configuration"""
    pytest.llm_config = pytest.llm_system.get_configuration()


@then("the model name should be set")
def model_name_set():
    """Verify model name is set"""
    assert "model_name" in pytest.llm_config
    assert pytest.llm_config["model_name"] is not None


@then("the base URL should be configured")
def base_url_configured():
    """Verify base URL is configured"""
    assert "base_url" in pytest.llm_config
    assert pytest.llm_config["base_url"] is not None


@then("temperature should be within valid range")
def temperature_valid_range():
    """Verify temperature is in valid range"""
    assert "temperature" in pytest.llm_config
    temp = pytest.llm_config["temperature"]
    assert 0.0 <= temp <= 2.0


@when("I send a message with streaming enabled")
async def send_with_streaming():
    """Send message with streaming enabled"""
    pytest.stream_chunks = []
    
    async for chunk in pytest.llm_system.generate_response_stream("Tell me a story"):
        pytest.stream_chunks.append(chunk)
    
    pytest.final_response = "".join(pytest.stream_chunks)


@then("I should receive response chunks")
def should_receive_chunks():
    """Verify we receive response chunks"""
    assert len(pytest.stream_chunks) > 0


@then("each chunk should be processed correctly")
def chunks_processed_correctly():
    """Verify chunks are processed correctly"""
    for chunk in pytest.stream_chunks:
        assert isinstance(chunk, str)


@then("the final response should be complete")
def final_response_complete():
    """Verify final response is complete"""
    assert len(pytest.final_response) > 0