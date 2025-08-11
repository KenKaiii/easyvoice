Feature: CLI Commands
  As a user of EasyVoice CLI
  I want to interact with the voice agent through command line
  So that I can use the system effectively

  Background:
    Given the EasyVoice CLI is installed
    And the configuration is set to test mode

  Scenario: Display version information
    When I run "easyvoice --version"
    Then I should see the version number
    And the exit code should be 0

  Scenario: Display help information
    When I run "easyvoice --help"
    Then I should see the usage information
    And I should see available commands:
      | command    | description                        |
      | listen     | Start interactive voice conversation |
      | ask        | Ask a single question              |
      | history    | Show conversation history          |
      | test-audio | Test audio pipeline               |

  Scenario: Test audio pipeline command
    Given the audio system is mocked
    When I run "easyvoice test-audio"
    Then the microphone test should run
    And the STT test should run
    And the TTS test should run
    And I should see success messages for each test

  Scenario: Test audio with verbose output
    Given the audio system is mocked
    When I run "easyvoice test-audio --verbose"
    Then I should see detailed output for each test
    And timing information should be displayed
    And technical details should be shown

  Scenario: Ask command with simple question
    Given the agent system is mocked
    When I run "easyvoice ask 'What time is it?'"
    Then I should get a response from the agent
    And the response should be displayed
    And the exit code should be 0

  Scenario: Ask command with voice output
    Given the agent and TTS systems are mocked
    When I run "easyvoice ask 'Hello' --voice"
    Then the response should be synthesized to speech
    And audio should be played
    And text should also be displayed

  Scenario: Ask command with history saving
    Given the agent and memory systems are mocked
    When I run "easyvoice ask 'Remember this' --save"
    Then the question and response should be saved to history
    And the memory count should increase

  Scenario: Show conversation history
    Given there is conversation history in the database
    When I run "easyvoice history"
    Then I should see recent conversations in table format
    And conversations should be ordered by time
    And both user and assistant messages should be shown

  Scenario: Show history with custom limit
    Given there are 15 messages in conversation history
    When I run "easyvoice history --limit 5"
    Then I should see exactly 5 messages
    And they should be the most recent ones

  Scenario: Show history in different formats
    Given there is conversation history in the database
    When I run "easyvoice history --format json"
    Then the output should be valid JSON
    When I run "easyvoice history --format plain"
    Then the output should be plain text without formatting

  Scenario: Handle empty history gracefully
    Given there is no conversation history
    When I run "easyvoice history"
    Then I should see a message indicating no history found
    And the exit code should be 0

  Scenario: Listen command setup
    Given the voice agent system is mocked
    When I run "easyvoice listen" in test mode
    Then the voice agent should be initialized
    And listening mode should start
    And I should see the status banner

  Scenario: Configuration loading
    Given a custom config file exists at "test_config.json"
    When I run "easyvoice --config test_config.json ask 'test'"
    Then the custom configuration should be loaded
    And the settings should reflect the config values

  Scenario: Handle invalid commands gracefully
    When I run "easyvoice invalid-command"
    Then I should see an error message
    And suggested commands should be displayed
    And the exit code should be non-zero

  Scenario: Handle missing arguments
    When I run "easyvoice ask"
    Then I should see an error about missing question
    And usage help should be displayed
    And the exit code should be non-zero

  Scenario: Development commands (hidden)
    When I run "easyvoice dev reset-memory"
    Then the memory database should be reset
    And I should see a confirmation message

  Scenario: Show current configuration
    When I run "easyvoice dev show-config"
    Then I should see all current configuration values
    And the format should be readable