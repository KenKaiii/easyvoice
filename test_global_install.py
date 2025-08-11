#!/usr/bin/env python3
"""
Test script to verify EasyVoice global installation works correctly
Tests both entry points and cross-platform compatibility
"""

import os
import sys
import subprocess
import tempfile
import shutil
import time
from pathlib import Path


def run_command(cmd, input_data=None, timeout=30):
    """Run a command and return output"""
    try:
        result = subprocess.run(
            cmd,
            input=input_data,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=True
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except Exception as e:
        return -2, "", str(e)


def test_global_command():
    """Test that easyvoice command is available globally"""
    print("🔍 Testing global easyvoice command...")
    
    # Test command availability
    returncode, stdout, stderr = run_command("which easyvoice", timeout=5)
    
    if returncode == 0:
        print(f"✅ easyvoice found at: {stdout.strip()}")
        
        # Test command execution
        returncode, stdout, stderr = run_command("echo 'quit' | easyvoice", timeout=10)
        
        if returncode == 0:
            print("✅ easyvoice command executes successfully")
            if "EasyVoice CLI" in stdout:
                print("✅ easyvoice shows correct banner")
            else:
                print("⚠️  easyvoice output may not be complete")
            return True
        else:
            print(f"❌ easyvoice command failed: {stderr}")
            return False
    else:
        print(f"❌ easyvoice command not found: {stderr}")
        return False


def test_python_module():
    """Test that python -m easyvoice works"""
    print("\n🔍 Testing python -m easyvoice...")
    
    # Determine Python command
    python_cmd = "python3" if shutil.which("python3") else "python"
    
    returncode, stdout, stderr = run_command(f"echo 'quit' | {python_cmd} -m easyvoice", timeout=10)
    
    if returncode == 0:
        print("✅ python -m easyvoice works")
        if "EasyVoice CLI" in stdout:
            print("✅ Module execution shows correct banner")
        else:
            print("⚠️  Module execution output may not be complete")
        return True
    else:
        print(f"❌ python -m easyvoice failed: {stderr}")
        return False


def test_cli_commands():
    """Test individual CLI commands"""
    print("\n🔍 Testing CLI commands...")
    
    commands_to_test = [
        ("status", "System Status"),
        ("config", "Configuration"),
        ("help", "Available Commands")
    ]
    
    success_count = 0
    
    for command, expected_output in commands_to_test:
        print(f"  Testing '{command}' command...")
        
        returncode, stdout, stderr = run_command(f"echo '{command}\\nquit' | easyvoice", timeout=15)
        
        if returncode == 0 and expected_output in stdout:
            print(f"    ✅ {command} command works")
            success_count += 1
        else:
            print(f"    ❌ {command} command failed or missing output")
    
    print(f"📊 {success_count}/{len(commands_to_test)} commands working")
    return success_count == len(commands_to_test)


def test_installation_integrity():
    """Test that installation is complete and working"""
    print("\n🔍 Testing installation integrity...")
    
    # Test that we can import the module
    try:
        import easyvoice
        print("✅ easyvoice module can be imported")
        
        # Test version
        if hasattr(easyvoice, '__version__'):
            print(f"✅ Version: {easyvoice.__version__}")
        else:
            print("⚠️  Version not available")
            
        return True
    except ImportError as e:
        print(f"❌ Cannot import easyvoice module: {e}")
        return False


def test_dependencies():
    """Test that core dependencies are available"""
    print("\n🔍 Testing core dependencies...")
    
    required_deps = ["rich", "httpx", "psutil", "pydantic"]
    optional_deps = ["sounddevice", "soundfile", "torch", "whisper"]
    
    core_success = True
    
    print("  Core dependencies:")
    for dep in required_deps:
        try:
            __import__(dep)
            print(f"    ✅ {dep}")
        except ImportError:
            print(f"    ❌ {dep} (required)")
            core_success = False
    
    print("  Optional audio dependencies:")
    audio_available = True
    for dep in optional_deps:
        try:
            __import__(dep)
            print(f"    ✅ {dep}")
        except ImportError:
            print(f"    ⚠️  {dep} (optional)")
            audio_available = False
    
    if audio_available:
        print("  🎤 Full audio processing available")
    else:
        print("  🎤 Audio processing in mock mode")
    
    return core_success


def test_performance():
    """Test performance and responsiveness"""
    print("\n🔍 Testing performance...")
    
    start_time = time.time()
    returncode, stdout, stderr = run_command("echo 'status\\nquit' | easyvoice", timeout=20)
    elapsed = time.time() - start_time
    
    if returncode == 0:
        print(f"✅ CLI startup time: {elapsed:.2f}s")
        
        if elapsed < 5.0:
            print("✅ Good performance (< 5s)")
        elif elapsed < 10.0:
            print("⚠️  Acceptable performance (5-10s)")
        else:
            print("❌ Slow performance (> 10s)")
        
        return elapsed < 10.0
    else:
        print("❌ Performance test failed")
        return False


def test_error_handling():
    """Test error handling and graceful degradation"""
    print("\n🔍 Testing error handling...")
    
    # Test invalid command
    returncode, stdout, stderr = run_command("echo 'invalidcommand\\nquit' | easyvoice", timeout=10)
    
    if returncode == 0 and "Unknown command" in stdout:
        print("✅ Invalid command handling works")
        
        # Test graceful exit
        returncode, stdout, stderr = run_command("echo 'quit' | easyvoice", timeout=10)
        if returncode == 0:
            print("✅ Graceful exit works")
            return True
        else:
            print("❌ Exit handling failed")
            return False
    else:
        print("❌ Error handling test failed")
        return False


def main():
    """Run all tests"""
    print("🧪 EasyVoice Global Installation Test Suite")
    print("=" * 50)
    
    tests = [
        ("Global Command", test_global_command),
        ("Python Module", test_python_module),
        ("CLI Commands", test_cli_commands),
        ("Installation Integrity", test_installation_integrity),
        ("Dependencies", test_dependencies),
        ("Performance", test_performance),
        ("Error Handling", test_error_handling),
    ]
    
    results = {}
    overall_success = True
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            results[test_name] = test_func()
            if not results[test_name]:
                overall_success = False
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
            overall_success = False
    
    # Print summary
    print("\n" + "="*60)
    print("📊 TEST RESULTS SUMMARY")
    print("="*60)
    
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:<25} {status}")
    
    print("="*60)
    
    if overall_success:
        print("🎉 ALL TESTS PASSED!")
        print("🚀 EasyVoice global installation is working perfectly!")
        print("\n💡 Quick start:")
        print("   easyvoice          # Start interactive CLI")
        print("   python -m easyvoice    # Alternative execution")
        return 0
    else:
        print("⚠️  SOME TESTS FAILED")
        print("🔧 Please check the issues above")
        
        failed_tests = [name for name, success in results.items() if not success]
        print(f"\nFailed tests: {', '.join(failed_tests)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())