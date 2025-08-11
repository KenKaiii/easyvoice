Feature: Tool System
  As a voice agent
  I want to use external tools to answer questions
  So that I can provide accurate, real-time information

  Background:
    Given the tools system is initialized
    And basic tools are available

  Scenario: Get current time
    When the user asks "What time is it?"
    Then the time tool should be called
    And I should get the current time
    And the response should include the time

  Scenario: Get system information
    When the user asks "How much memory is being used?"
    Then the system info tool should be called
    And I should get memory usage data
    And the response should be formatted properly

  Scenario: Handle tool timeout
    Given a tool that responds slowly
    When I call the slow tool with a short timeout
    Then the tool call should timeout gracefully
    And I should get a timeout error
    And the system should remain stable

  Scenario: Handle tool failure
    Given a tool that always fails
    When I try to use the failing tool
    Then I should get an appropriate error message
    And the agent should continue normally
    And suggest alternative approaches

  Scenario: Chain multiple tools
    When the user asks "What time is it and how's the system performance?"
    Then the time tool should be called
    And the system info tool should be called
    And both results should be combined
    And the response should address both questions

  Scenario: Tool registration and discovery
    When I register a new custom tool
    Then it should be available for use
    And it should appear in the tools list
    And it should be callable by the agent

  Scenario: Tool with parameters
    When the user asks "What's the weather in Paris?"
    Then the weather tool should be called
    And the location parameter should be "Paris"
    And the tool should return weather data
    And the response should be location-specific