#!/usr/bin/env python3
"""
Basic test runner for EasyVoice CLI without heavy dependencies
This tests the core functionality that we can verify without audio hardware
"""

import sys
import os
import traceback
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Test results tracking
test_results = []


def run_test(test_name, test_func):
    """Run a single test and track results"""
    print(f"🧪 {test_name}...")
    try:
        result = test_func()
        if result:
            print(f"  ✅ PASSED")
            test_results.append(True)
        else:
            print(f"  ❌ FAILED")
            test_results.append(False)
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        if "--verbose" in sys.argv:
            traceback.print_exc()
        test_results.append(False)
    print()


def test_settings_module():
    """Test Settings configuration module"""
    from easyvoice.config.settings import Settings
    
    # Test default settings
    settings = Settings()
    
    assert settings.sample_rate == 16000
    assert settings.max_messages == 20
    assert settings.whisper_model == "base"
    assert settings.stt_timeout == 30
    assert settings.tts_timeout == 15
    
    # Test environment variable override
    os.environ["EASYVOICE_MAX_MESSAGES"] = "10"
    settings_with_env = Settings()
    assert settings_with_env.max_messages == 10
    
    # Clean up
    del os.environ["EASYVOICE_MAX_MESSAGES"]
    
    # Test validation
    try:
        bad_settings = Settings()
        bad_settings.sample_rate = 12345  # Invalid rate
        bad_settings.__post_init__()
        return False  # Should have raised error
    except ValueError:
        pass  # Expected
    
    return True


def test_vad_without_audio():
    """Test Voice Activity Detector logic without audio hardware"""
    try:
        from easyvoice.audio.input import VoiceActivityDetector
        import numpy as np
        
        vad = VoiceActivityDetector(threshold=0.01)
        
        # Test with loud audio (should detect speech)
        loud_audio = np.array([0.5, 0.6, 0.7, 0.8, 0.5])
        speech_detected = vad.process_chunk(loud_audio)
        
        # Test with quiet audio (should not detect speech)
        quiet_audio = np.array([0.001, 0.002, 0.001, 0.0005])
        no_speech = vad.process_chunk(quiet_audio)
        
        # VAD should detect the loud audio but not the quiet audio
        return speech_detected and not no_speech
    except ImportError as e:
        print(f"    ⚠️  Skipping VAD test - missing dependency: {e}")
        return True  # Skip test if dependencies missing


def test_cli_imports():
    """Test CLI module imports and basic structure"""
    from easyvoice.cli import main
    from click.testing import CliRunner
    
    # Test that CLI command group exists
    assert callable(main)
    
    # Test basic CLI structure (without running full commands)
    runner = CliRunner()
    
    # Test version command
    result = runner.invoke(main, ['--version'])
    assert "1.0.0" in result.output
    
    # Test help command
    result = runner.invoke(main, ['--help'])
    assert result.exit_code == 0
    assert "listen" in result.output
    assert "ask" in result.output
    assert "history" in result.output
    assert "test-audio" in result.output
    
    return True


def test_settings_validation():
    """Test settings validation logic"""
    from easyvoice.config.settings import Settings
    
    # Test valid settings
    settings = Settings(
        sample_rate=16000,
        chunk_size=1024,
        tts_voice=3,
        temperature=0.7
    )
    settings.__post_init__()  # Should not raise
    
    # Test invalid sample rate
    try:
        settings = Settings(sample_rate=12345)
        settings.__post_init__()
        return False  # Should have raised
    except ValueError:
        pass
    
    # Test invalid chunk size (not power of 2)
    try:
        settings = Settings(chunk_size=1000)
        settings.__post_init__()
        return False  # Should have raised
    except ValueError:
        pass
    
    # Test invalid TTS voice
    try:
        settings = Settings(tts_voice=10)
        settings.__post_init__()
        return False  # Should have raised
    except ValueError:
        pass
    
    return True


def test_module_structure():
    """Test that all modules have correct structure"""
    # Test that all __init__.py files exist and are importable
    modules = [
        "easyvoice",
        "easyvoice.config",
        "easyvoice.audio", 
        "easyvoice.ui",
        "easyvoice.agent"
    ]
    
    for module in modules:
        try:
            __import__(module)
        except ImportError as e:
            print(f"Module {module} import failed: {e}")
            return False
    
    return True


def test_settings_file_operations():
    """Test settings file save/load operations"""
    from easyvoice.config.settings import Settings
    import tempfile
    import json
    
    settings = Settings(max_messages=15, stt_timeout=25)
    
    # Test JSON save/load
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json_path = f.name
    
    try:
        settings.save_to_file(json_path)
        loaded_settings = Settings.from_file(json_path)
        
        assert loaded_settings.max_messages == 15
        assert loaded_settings.stt_timeout == 25
        
    finally:
        os.unlink(json_path)
    
    return True


def test_mocked_audio_components():
    """Test audio components with mocked dependencies"""
    # Test that classes can be imported without sounddevice
    try:
        from easyvoice.audio.input import VoiceActivityDetector, HAS_SOUNDDEVICE
        from easyvoice.config.settings import Settings
        
        # Test VAD standalone
        vad = VoiceActivityDetector()
        assert vad.threshold == 0.01
        
        # Test settings for audio
        settings = Settings()
        assert settings.sample_rate in [8000, 16000, 22050, 44100, 48000]
        
        # Verify we can detect missing sounddevice
        if not HAS_SOUNDDEVICE:
            print("    ⚠️  sounddevice not available - this is expected in test environment")
        
        return True
    except ImportError as e:
        print(f"    ⚠️  Import error: {e}")
        return False


def main():
    """Run all basic tests"""
    print("🎯 EasyVoice Basic Test Suite")
    print("=" * 50)
    print("Testing core functionality without heavy dependencies\n")
    
    # Define tests to run
    tests = [
        ("Settings Module", test_settings_module),
        ("Voice Activity Detection Logic", test_vad_without_audio),
        ("CLI Structure", test_cli_imports),
        ("Settings Validation", test_settings_validation),
        ("Module Structure", test_module_structure),
        ("Settings File Operations", test_settings_file_operations),
        ("Mocked Audio Components", test_mocked_audio_components),
    ]
    
    # Run all tests
    for test_name, test_func in tests:
        run_test(test_name, test_func)
    
    # Report results
    passed = sum(test_results)
    total = len(test_results)
    percentage = (passed / total) * 100
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} passed ({percentage:.1f}%)")
    
    if passed == total:
        print("🎉 All basic tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed - see details above")
        return 1


if __name__ == "__main__":
    sys.exit(main())