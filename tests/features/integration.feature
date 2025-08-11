Feature: Complete System Integration
  As a user of EasyVoice CLI
  I want all components to work together seamlessly
  So that I can have a complete voice agent experience

  Background:
    Given the complete system is initialized
    And all components are healthy

  Scenario: End-to-end conversation with tools
    When I ask "What time is it and how's the system performance?"
    Then the system should use both time and system tools
    And provide a comprehensive response
    And save the conversation to memory
    And all operations should complete within timeout

  Scenario: Memory persistence across sessions
    Given I have a previous conversation
    When I start a new session
    And ask about previous conversation
    Then the agent should remember past context
    And provide relevant responses

  Scenario: Tool chaining and context
    When I ask "What's the current time?"
    And follow up with "How about system memory usage?"
    And then ask "Can you summarize what I just asked?"
    Then each response should build on the previous
    And the summary should reference both queries

  Scenario: Error recovery and graceful degradation
    Given one component fails temporarily
    When I interact with the system
    Then it should continue functioning
    And provide appropriate error messages
    And recover when the component is restored

  Scenario: Performance and timeout handling
    Given the system is under load
    When I make multiple concurrent requests
    Then all requests should be handled
    And timeouts should be respected
    And the system should remain responsive