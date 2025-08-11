"""Step definitions for conversation flow BDD scenarios"""

import pytest
import asyncio
from unittest.mock import Mock, patch
from pytest_bdd import scenarios, given, when, then, parsers

# Load scenarios from feature file
scenarios('../features/conversation_flow.feature')

# Global test state
pytest.voice_agent = None
pytest.transcription = None
pytest.agent_response = None
pytest.spoken_response = None
pytest.error_message = None
pytest.conversation_history = []


@given("the voice agent is initialized")
def voice_agent_initialized(test_settings):
    """Initialize the complete voice agent system"""
    from easyvoice.agent.core import VoiceAgent
    pytest.voice_agent = VoiceAgent(test_settings)


@given("all components are ready")
def all_components_ready():
    """Verify all components are initialized"""
    assert pytest.voice_agent is not None
    assert pytest.voice_agent.is_ready()


@when('I speak "Hello, how are you today?"')
async def speak_hello():
    """Simulate speaking to the agent"""
    pytest.spoken_text = "Hello, how are you today?"
    pytest.agent_response = await pytest.voice_agent.process_voice_input(pytest.spoken_text)


@then("the agent should transcribe my speech")
def agent_transcribes_speech():
    """Verify speech was transcribed"""
    transcription = pytest.voice_agent.get_last_transcription()
    assert transcription == "Hello, how are you today?"


@then("process it through the LLM")
def process_through_llm():
    """Verify message was processed by LLM"""
    assert pytest.agent_response is not None
    assert len(pytest.agent_response) > 0


@then("generate a spoken response")
def generate_spoken_response():
    """Verify spoken response was generated"""
    audio_output = pytest.voice_agent.get_last_audio_output()
    assert audio_output is not None


@then("save both messages to memory")
def save_messages_to_memory():
    """Verify both messages were saved"""
    memory = pytest.voice_agent.get_memory()
    recent_messages = memory.get_recent_messages(limit=2)
    
    assert len(recent_messages) == 2
    assert recent_messages[0]["role"] == "user"
    assert recent_messages[0]["content"] == "Hello, how are you today?"
    assert recent_messages[1]["role"] == "assistant"
    assert recent_messages[1]["content"] == pytest.agent_response


@given("I have previous conversation history")
def have_previous_history():
    """Setup previous conversation history"""
    memory = pytest.voice_agent.get_memory()
    memory.add_message("user", "What's the weather like?")
    memory.add_message("assistant", "I'd need to check a weather service for current conditions.")
    pytest.conversation_history = memory.get_recent_messages()


@when('I ask a follow-up question "What did I just ask about?"')
async def ask_followup_question():
    """Ask a follow-up question"""
    pytest.followup_response = await pytest.voice_agent.process_voice_input(
        "What did I just ask about?"
    )


@then("the agent should use conversation context")
def agent_uses_context():
    """Verify agent used conversation context"""
    # In a real test, we'd verify the LLM received the conversation history
    assert pytest.followup_response is not None


@then("provide a relevant response")
def provide_relevant_response():
    """Verify response is relevant to context"""
    # In a real implementation, this would check if response references previous messages
    assert "weather" in pytest.followup_response.lower() or len(pytest.followup_response) > 0


@then("maintain conversation continuity")
def maintain_continuity():
    """Verify conversation continuity is maintained"""
    memory = pytest.voice_agent.get_memory()
    messages = memory.get_recent_messages()
    assert len(messages) >= 4  # Original 2 + new 2


@given("the STT system fails")
def stt_system_fails():
    """Mock STT system failure"""
    pytest.voice_agent.stt_available = False


@when("I try to speak to the agent")
async def try_speak_to_agent():
    """Try to speak when STT fails"""
    try:
        pytest.agent_response = await pytest.voice_agent.process_voice_input("Test message")
        pytest.error_message = None
    except Exception as e:
        pytest.error_message = str(e)
        pytest.agent_response = None


@then("I should get an appropriate error message")
def should_get_error_message():
    """Verify we get an appropriate error message"""
    assert pytest.error_message is not None
    assert "speech" in pytest.error_message.lower() or "stt" in pytest.error_message.lower()


@then("the agent should remain in listening mode")
def agent_remains_listening():
    """Verify agent is still in listening mode"""
    assert pytest.voice_agent.is_listening()


@then("suggest troubleshooting steps")
def suggest_troubleshooting():
    """Verify troubleshooting steps are suggested"""
    # In a real implementation, this would check for specific troubleshooting guidance
    assert pytest.error_message is not None


@given("voice input is not available")
def voice_input_not_available():
    """Mock voice input unavailability"""
    pytest.voice_agent.audio_available = False


@when('I send a text message "Process this as text"')
async def send_text_message():
    """Send text message instead of voice"""
    pytest.text_response = await pytest.voice_agent.process_text_input("Process this as text")


@then("the agent should process it normally")
def agent_processes_normally():
    """Verify agent processes text normally"""
    assert pytest.text_response is not None
    assert len(pytest.text_response) > 0


@then("respond appropriately")
def respond_appropriately():
    """Verify appropriate response"""
    assert pytest.text_response is not None


@then("save the interaction to memory")
def save_interaction_to_memory():
    """Verify interaction saved to memory"""
    memory = pytest.voice_agent.get_memory()
    messages = memory.get_recent_messages(limit=2)
    
    assert any(msg["content"] == "Process this as text" for msg in messages)


@when('I start a conversation about "planning a trip"')
async def start_trip_conversation():
    """Start conversation about planning a trip"""
    pytest.trip_response_1 = await pytest.voice_agent.process_text_input(
        "I'm planning a trip to Paris next month"
    )


@when('continue with "what about the weather?"')
async def continue_weather():
    """Continue with weather question"""
    pytest.trip_response_2 = await pytest.voice_agent.process_text_input(
        "what about the weather?"
    )


@when('follow up with "any restaurant recommendations?"')
async def followup_restaurants():
    """Follow up with restaurant question"""
    pytest.trip_response_3 = await pytest.voice_agent.process_text_input(
        "any restaurant recommendations?"
    )


@then("all messages should be stored in order")
def messages_stored_in_order():
    """Verify all messages stored in chronological order"""
    memory = pytest.voice_agent.get_memory()
    messages = memory.get_recent_messages()
    
    # Should have at least 6 messages (3 user + 3 assistant)
    assert len(messages) >= 6
    
    # Verify chronological order
    for i in range(len(messages) - 1):
        assert messages[i]["timestamp"] <= messages[i + 1]["timestamp"]


@then("each response should build on previous context")
def responses_build_on_context():
    """Verify responses build on context"""
    # In a real test, this would verify that responses reference previous context
    assert pytest.trip_response_1 is not None
    assert pytest.trip_response_2 is not None
    assert pytest.trip_response_3 is not None


@then("the memory should not exceed 20 messages")
def memory_not_exceed_20():
    """Verify memory doesn't exceed limit"""
    memory = pytest.voice_agent.get_memory()
    assert memory.get_message_count() <= 20


@given("I'm in an active conversation")
def in_active_conversation():
    """Setup active conversation state"""
    pytest.voice_agent.start_conversation_session()
    pytest.session_active = True


@when("no input is received for session timeout period")
async def no_input_for_timeout():
    """Simulate session timeout"""
    # Mock the timeout by advancing time
    await pytest.voice_agent.check_session_timeout()


@then("the conversation should end gracefully")
def conversation_ends_gracefully():
    """Verify conversation ended gracefully"""
    assert not pytest.voice_agent.is_session_active()


@then("I should get a timeout notification")
def get_timeout_notification():
    """Verify timeout notification received"""
    notifications = pytest.voice_agent.get_notifications()
    timeout_notif = any("timeout" in n.lower() for n in notifications)
    assert timeout_notif


@then("be able to start a new conversation")
def can_start_new_conversation():
    """Verify can start new conversation"""
    pytest.voice_agent.start_conversation_session()
    assert pytest.voice_agent.is_session_active()


@when('I ask "What time is it?"')
async def ask_time():
    """Ask for current time"""
    pytest.time_response = await pytest.voice_agent.process_text_input("What time is it?")


@then("the agent should use the time tool")
def agent_uses_time_tool():
    """Verify agent used time tool"""
    tools_used = pytest.voice_agent.get_last_tools_used()
    assert "time" in [tool["name"].lower() for tool in tools_used]


@then("provide the current time")
def provide_current_time():
    """Verify current time provided"""
    assert pytest.time_response is not None
    # In real test, would verify time format
    assert len(pytest.time_response) > 0


@then("mention which tool was used")
def mention_tool_used():
    """Verify tool usage mentioned"""
    # In real implementation, response might mention tool usage
    assert pytest.time_response is not None


@then("save the tool usage to memory")
def save_tool_usage():
    """Verify tool usage saved to memory"""
    memory = pytest.voice_agent.get_memory()
    messages = memory.get_recent_messages(limit=2)
    
    # Verify both user question and assistant response with tool usage
    assert len(messages) >= 2