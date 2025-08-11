Feature: Conversation Memory
  As a user of EasyVoice CLI
  I want the agent to remember our conversation
  So that I can have contextual discussions

  Background:
    Given the memory system is initialized
    And the database is empty

  Scenario: Store and retrieve a single message
    When I add a user message "Hello agent"
    Then the message should be stored in memory
    And I should be able to retrieve it
    And the message count should be 1

  Scenario: Enforce 20 message limit with sliding window
    Given I have 20 messages in memory
    When I add a new user message "This is message 21"
    Then the oldest message should be removed
    And the message count should still be 20
    And the newest message should be retrievable

  Scenario: Get conversation history for context
    Given I have a conversation with 5 exchanges
    When I request the conversation history
    Then I should get all 10 messages in chronological order
    And each message should have role, content, and timestamp

  Scenario: Clear conversation memory
    Given I have 10 messages in memory
    When I clear the memory
    Then the message count should be 0
    And retrieving messages should return empty list