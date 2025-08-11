"""Step definitions for memory BDD scenarios"""

import pytest
from pytest_bdd import given, scenarios, then, when

# Load scenarios from feature file
scenarios("../features/memory.feature")

# Global test state
pytest.memory_system = None
pytest.stored_messages = []


@given("the memory system is initialized")
def memory_system_initialized(test_settings):
    """Initialize the memory system"""
    from easyvoice.agent.memory import ConversationMemory

    pytest.memory_system = ConversationMemory(test_settings)


@given("the database is empty")
def database_is_empty():
    """Ensure database starts empty"""
    pytest.memory_system.clear_all()
    assert pytest.memory_system.get_message_count() == 0


@when('I add a user message "Hello agent"')
def add_user_message():
    """Add a user message to memory"""
    pytest.memory_system.add_message("user", "Hello agent")


@then("the message should be stored in memory")
def message_stored_in_memory():
    """Verify message is stored"""
    messages = pytest.memory_system.get_recent_messages(limit=1)
    assert len(messages) == 1
    assert messages[0]["content"] == "Hello agent"


@then("I should be able to retrieve it")
def should_retrieve_message():
    """Verify message can be retrieved"""
    messages = pytest.memory_system.get_recent_messages(limit=1)
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Hello agent"


@then("the message count should be 1")
def message_count_should_be_1():
    """Verify message count is 1"""
    assert pytest.memory_system.get_message_count() == 1


@given("I have 20 messages in memory")
def have_20_messages():
    """Add 20 messages to memory"""
    pytest.memory_system.clear_all()
    for i in range(20):
        role = "user" if i % 2 == 0 else "assistant"
        pytest.memory_system.add_message(role, f"Message {i+1}")

    assert pytest.memory_system.get_message_count() == 20


@when('I add a new user message "This is message 21"')
def add_21st_message():
    """Add the 21st message"""
    pytest.memory_system.add_message("user", "This is message 21")


@then("the oldest message should be removed")
def oldest_message_removed():
    """Verify oldest message was removed"""
    messages = pytest.memory_system.get_all_messages()
    # The first message should no longer be "Message 1"
    assert messages[0]["content"] != "Message 1"


@then("the message count should still be 20")
def message_count_still_20():
    """Verify count is still 20"""
    assert pytest.memory_system.get_message_count() == 20


@then("the newest message should be retrievable")
def newest_message_retrievable():
    """Verify newest message can be retrieved"""
    messages = pytest.memory_system.get_recent_messages(limit=1)
    assert messages[0]["content"] == "This is message 21"


@given("I have a conversation with 5 exchanges")
def have_5_exchanges():
    """Add 5 exchanges (10 messages) to memory"""
    pytest.memory_system.clear_all()
    for i in range(5):
        pytest.memory_system.add_message("user", f"User message {i+1}")
        pytest.memory_system.add_message("assistant", f"Assistant response {i+1}")

    assert pytest.memory_system.get_message_count() == 10


@when("I request the conversation history")
def request_conversation_history():
    """Request conversation history"""
    pytest.conversation_history = pytest.memory_system.get_recent_messages(limit=20)


@then("I should get all 10 messages in chronological order")
def get_10_messages_chronological():
    """Verify we get 10 messages in order"""
    assert len(pytest.conversation_history) == 10

    # Check chronological order (first message should be oldest)
    for i in range(len(pytest.conversation_history) - 1):
        current_time = pytest.conversation_history[i]["timestamp"]
        next_time = pytest.conversation_history[i + 1]["timestamp"]
        assert current_time <= next_time


@then("each message should have role, content, and timestamp")
def each_message_has_required_fields():
    """Verify each message has required fields"""
    for message in pytest.conversation_history:
        assert "role" in message
        assert "content" in message
        assert "timestamp" in message
        assert message["role"] in ["user", "assistant"]


@given("I have 10 messages in memory")
def have_10_messages():
    """Add 10 messages to memory"""
    pytest.memory_system.clear_all()
    for i in range(10):
        role = "user" if i % 2 == 0 else "assistant"
        pytest.memory_system.add_message(role, f"Message {i+1}")

    assert pytest.memory_system.get_message_count() == 10


@when("I clear the memory")
def clear_memory():
    """Clear all memory"""
    pytest.memory_system.clear_all()


@then("the message count should be 0")
def message_count_should_be_0():
    """Verify message count is 0"""
    assert pytest.memory_system.get_message_count() == 0


@then("retrieving messages should return empty list")
def retrieving_messages_empty():
    """Verify retrieving messages returns empty list"""
    messages = pytest.memory_system.get_recent_messages()
    assert len(messages) == 0
