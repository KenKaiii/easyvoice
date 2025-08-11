Feature: Conversation Flow
  As a user of EasyVoice CLI
  I want to have natural conversations with the agent
  So that I can interact through voice seamlessly

  Background:
    Given the voice agent is initialized
    And all components are ready

  Scenario: Complete voice conversation cycle
    When I speak "Hello, how are you today?"
    Then the agent should transcribe my speech
    And process it through the LLM
    And generate a spoken response
    And save both messages to memory

  Scenario: Contextual conversation
    Given I have previous conversation history
    When I ask a follow-up question "What did I just ask about?"
    Then the agent should use conversation context
    And provide a relevant response
    And maintain conversation continuity

  Scenario: Handle processing errors gracefully
    Given the STT system fails
    When I try to speak to the agent
    Then I should get an appropriate error message
    And the agent should remain in listening mode
    And suggest troubleshooting steps

  Scenario: Process text input when voice fails
    Given voice input is not available
    When I send a text message "Process this as text"
    Then the agent should process it normally
    And respond appropriately
    And save the interaction to memory

  Scenario: Multi-turn conversation with memory
    When I start a conversation about "planning a trip"
    And continue with "what about the weather?"
    And follow up with "any restaurant recommendations?"
    Then all messages should be stored in order
    And each response should build on previous context
    And the memory should not exceed 20 messages

  Scenario: Conversation timeout and recovery
    Given I'm in an active conversation
    When no input is received for session timeout period
    Then the conversation should end gracefully
    And I should get a timeout notification
    And be able to start a new conversation

  Scenario: Agent response with tool usage
    When I ask "What time is it?"
    Then the agent should use the time tool
    And provide the current time
    And mention which tool was used
    And save the tool usage to memory