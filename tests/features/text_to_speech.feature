Feature: Text to Speech
  As a user of EasyVoice CLI
  I want the system to convert text responses to natural speech
  So that I can hear the voice agent's responses

  Background:
    Given the KittenTTS system is initialized with test settings
    And the TTS model loading is mocked for testing

  Scenario: Load KittenTTS model successfully
    When I load the KittenTTS model
    Then the model should be loaded successfully
    And the model status should be "loaded"
    And no errors should occur

  Scenario: Synthesize simple text
    Given the KittenTTS model is loaded
    When I synthesize the text "Hello world"
    Then audio data should be generated
    And the audio data should not be empty
    And the operation should complete within the timeout

  Scenario: Handle empty text input
    Given the KittenTTS model is loaded
    When I try to synthesize empty text
    Then the synthesis should return None
    And an appropriate warning should be logged

  Scenario: Voice selection
    Given the KittenTTS model is loaded
    When I synthesize text with different voices
      | voice_id | voice_type    | expected_result |
      | 0        | Male Voice 1  | success         |
      | 4        | Female Voice 1| success         |
      | 7        | Female Voice 4| success         |
    Then each voice should generate unique audio
    And all syntheses should complete successfully

  Scenario: Invalid voice handling
    Given the KittenTTS model is loaded
    When I try to synthesize with voice ID 10
    Then an error should be raised
    And the error should mention invalid voice ID

  Scenario: Audio playback
    Given the KittenTTS model is loaded
    And I have synthesized audio data
    When I play the audio
    Then the playback should start successfully
    And no audio errors should occur

  Scenario: Speed adjustment
    Given the KittenTTS model is loaded
    And speed adjustment is set to 1.5x
    When I synthesize text "Testing speed"
    Then the generated audio should be faster than normal
    And the audio duration should be shorter

  Scenario: Save synthesized audio
    Given the KittenTTS model is loaded
    And I have synthesized audio data
    When I save the audio to "test_output.wav"
    Then the file should be created successfully
    And the file should contain valid audio data

  Scenario: TTS timeout handling
    Given the KittenTTS model is loaded
    And synthesis will take longer than timeout
    When I try to synthesize with a short timeout
    Then the operation should timeout gracefully
    And a timeout error should be logged
    And the result should be None

  Scenario: Available voices information
    Given the KittenTTS model is loaded
    When I request available voices information
    Then I should get a list of 8 voices
    And each voice should have an ID and description
    And voices should include both male and female options

  Scenario: Model performance metrics
    Given the KittenTTS model is loaded
    When I benchmark TTS performance with "Performance test text"
    Then I should get timing metrics
    And the real-time factor should be calculated
    And model information should be included

  Scenario: Resource cleanup
    Given the KittenTTS model is loaded
    When I close the TTS system
    Then all resources should be cleaned up
    And the model should be unloaded from memory