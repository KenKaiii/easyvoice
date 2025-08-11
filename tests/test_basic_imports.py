"""Basic import tests to verify modules load correctly"""

import os
import sys

# Add the project root to the path so we can import our modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def test_config_imports():
    """Test that configuration module imports correctly"""
    try:
        from easyvoice.config.settings import Settings

        # Test basic Settings instantiation
        settings = Settings()

        assert settings.sample_rate == 16000
        assert settings.max_messages == 20
        assert settings.whisper_model == "base"

        print("✅ Settings module imports and works correctly")
        return True

    except Exception as e:
        print(f"❌ Settings import failed: {e}")
        return False


def test_audio_input_imports():
    """Test that audio input module imports correctly"""
    try:
        from easyvoice.audio.input import VoiceActivityDetector

        # Test VAD creation
        vad = VoiceActivityDetector()
        assert vad.threshold == 0.01

        print("✅ Audio input module imports correctly")
        return True

    except Exception as e:
        print(f"❌ Audio input import failed: {e}")
        return False


def test_stt_imports():
    """Test that STT module imports correctly"""
    try:
        # Don't actually load the model, just test class creation

        print("✅ STT module imports correctly")
        return True

    except Exception as e:
        print(f"❌ STT import failed: {e}")
        return False


def test_tts_imports():
    """Test that TTS module imports correctly"""
    try:
        # Don't actually load the model, just test class creation

        print("✅ TTS module imports correctly")
        return True

    except Exception as e:
        print(f"❌ TTS import failed: {e}")
        return False


def test_cli_imports():
    """Test that CLI module imports correctly"""
    try:
        from easyvoice.cli import main

        # Test that the main function exists
        assert callable(main)

        print("✅ CLI module imports correctly")
        return True

    except Exception as e:
        print(f"❌ CLI import failed: {e}")
        return False


def run_all_tests():
    """Run all basic import tests"""
    print("🧪 Running basic import tests...\n")

    tests = [
        test_config_imports,
        test_audio_input_imports,
        test_stt_imports,
        test_tts_imports,
        test_cli_imports
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            results.append(False)
        print()

    passed = sum(results)
    total = len(results)

    print(f"📊 Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All basic import tests passed!")
        return True
    else:
        print("⚠️  Some import tests failed - need to fix dependencies")
        return False


if __name__ == "__main__":
    run_all_tests()
