Feature: Speech Recognition
  As a user of EasyVoice CLI
  I want the system to convert my speech to text accurately
  So that the voice agent can understand what I'm saying

  Background:
    Given the Whisper STT system is initialized with test settings
    And the model loading is mocked for testing

  Scenario: Load Whisper model successfully
    When I load the Whisper model
    Then the model should be loaded successfully
    And the model status should be "loaded"
    And no errors should occur

  Scenario: Transcribe clear speech audio
    Given the Whisper model is loaded
    And I have clear speech audio data
    When I transcribe the audio data
    Then I should get accurate transcribed text
    And the transcription should not be empty
    And the operation should complete within the timeout

  Scenario: Handle empty audio gracefully
    Given the Whisper model is loaded
    When I try to transcribe empty audio data
    Then the transcription should return None
    And an appropriate warning should be logged

  Scenario: Transcription timeout handling
    Given the Whisper model is loaded
    And transcription will take longer than timeout
    When I try to transcribe audio with a short timeout
    Then the operation should timeout gracefully
    And a timeout error should be logged
    And the result should be None

  Scenario: Handle different audio formats
    Given the Whisper model is loaded
    When I transcribe audio data in different formats
      | format    | expected_result |
      | float32   | success         |
      | int16     | success         |
      | normalized| success         |
    Then each transcription should handle the format correctly

  Scenario: Language detection
    Given the Whisper model is loaded
    And I have multilingual audio samples
    When I transcribe without specifying language
    Then the model should detect the language automatically
    And return transcribed text in the detected language

  Scenario: Model information retrieval
    Given the Whisper model is loaded
    When I request model information
    Then I should get model details including:
      | field         | expected    |
      | status        | loaded      |
      | model_name    | test-model  |
      | is_multilingual| true       |

  Scenario: Resource cleanup
    Given the Whisper model is loaded
    When I close the STT system
    Then all resources should be cleaned up
    And the model should be unloaded from memory