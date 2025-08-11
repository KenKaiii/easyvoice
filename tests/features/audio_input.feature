Feature: Audio Input
  As a user of EasyVoice CLI
  I want the system to capture audio from my microphone
  So that I can speak to the voice agent

  Background:
    Given the audio system is initialized with test settings
    And the microphone is available

  Scenario: Test microphone availability
    When I test the microphone
    Then the microphone should be detected
    And no errors should occur

  Scenario: Record audio for fixed duration
    Given the audio input is ready
    When I record audio for 2 seconds
    Then audio data should be captured
    And the audio data should not be empty
    And the recording should stop automatically

  Scenario: Voice activity detection
    Given the voice activity detector is initialized
    When I process an audio chunk with speech
    Then voice activity should be detected
    When I process an audio chunk with silence
    Then no voice activity should be detected

  Scenario: Record until silence detected
    Given the audio input is ready
    And voice activity detection is enabled
    When I start recording with silence detection
    And speech is detected initially
    And silence follows for 1 second
    Then recording should stop automatically
    And captured audio should contain the speech

  Scenario: Handle microphone errors gracefully
    Given the microphone is not available
    When I try to start audio recording
    Then an appropriate error should be raised
    And the error message should be informative

  Scenario: Audio buffer management
    Given the audio input is recording
    When audio data accumulates in the buffer
    And I request the audio data
    Then the buffer should return the audio data
    And the buffer should be cleared after retrieval

  Scenario: Timeout protection
    Given the audio input is ready
    When I start recording with a 5 second timeout
    And no silence is detected
    Then recording should stop after 5 seconds
    And a timeout warning should be logged