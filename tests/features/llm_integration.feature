Feature: LLM Integration
  As a voice agent system
  I want to process user messages through an LLM
  So that I can provide intelligent responses

  Background:
    Given the LLM system is initialized with test settings
    And Ollama is available for testing

  Scenario: Generate response to simple question
    When I send the message "What is 2 + 2?"
    Then I should get a response from the LLM
    And the response should not be empty
    And the operation should complete within timeout

  Scenario: Handle conversation context
    Given I have conversation history with 3 messages
    When I send a follow-up message "What did we just discuss?"
    Then the LLM should receive the conversation context
    And the response should reference previous messages

  Scenario: Handle LLM timeout gracefully
    Given the LLM will respond slowly
    When I send a message with a short timeout
    Then the operation should timeout gracefully
    And I should get a timeout error
    And no partial response should be returned

  Scenario: Handle LLM connection failure
    Given Ollama is not available
    When I try to send a message
    Then I should get a connection error
    And the error message should be informative

  Scenario: Validate LLM configuration
    When I check the LLM configuration
    Then the model name should be set
    And the base URL should be configured
    And temperature should be within valid range

  Scenario: Stream LLM response
    When I send a message with streaming enabled
    Then I should receive response chunks
    And each chunk should be processed correctly
    And the final response should be complete