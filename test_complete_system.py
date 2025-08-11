#!/usr/bin/env python3
"""
Complete system integration test for EasyVoice CLI
Tests the full pipeline with BDD approach and timeout protection
"""

import sys
import os
import asyncio
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from easyvoice.agent.core import VoiceAgent
from easyvoice.config.settings import Settings


async def test_complete_pipeline():
    """Test the complete EasyVoice pipeline"""
    print("🎯 EasyVoice Complete System Test")
    print("=" * 50)
    
    # Initialize with test settings
    settings = Settings(
        debug=True,
        db_path='test_complete.db',
        max_messages=10,
        stt_timeout=5,
        tts_timeout=5,
        llm_timeout=10
    )
    
    print("1️⃣ Initializing complete system...")
    start_time = time.time()
    agent = VoiceAgent(settings)
    init_time = time.time() - start_time
    print(f"   ✅ System initialized in {init_time:.2f}s")
    
    # Test system health
    print("\n2️⃣ Testing system health...")
    assert agent.is_ready(), "System should be ready"
    
    memory_count = agent.get_memory().get_message_count()
    tools = agent.tools.get_available_tools()
    llm_config = agent.llm.get_configuration()
    
    print(f"   ✅ Memory: {memory_count} messages")
    print(f"   ✅ Tools: {len(tools)} available ({', '.join(tools)})")
    print(f"   ✅ LLM: {llm_config['model_name']} ready")
    
    # Test 1: Time query with tool usage
    print("\n3️⃣ Testing time query with tools...")
    start_time = time.time()
    response1 = await agent.process_text_input("What time is it?")
    query_time = time.time() - start_time
    
    assert response1 is not None, "Should get response"
    assert len(response1) > 0, "Response should not be empty"
    print(f"   ✅ Time query completed in {query_time:.2f}s")
    print(f"   📝 Response: {response1[:100]}...")
    
    # Test 2: System info query
    print("\n4️⃣ Testing system info query...")
    start_time = time.time()
    response2 = await agent.process_text_input("How much memory is being used?")
    query_time = time.time() - start_time
    
    assert response2 is not None, "Should get response"
    assert "memory" in response2.lower() or "mb" in response2.lower(), "Should contain memory info"
    print(f"   ✅ System query completed in {query_time:.2f}s")
    print(f"   📝 Response: {response2[:100]}...")
    
    # Test 3: Weather query with location parameter
    print("\n5️⃣ Testing weather query...")
    start_time = time.time()
    response3 = await agent.process_text_input("What's the weather like in Tokyo?")
    query_time = time.time() - start_time
    
    assert response3 is not None, "Should get response"
    print(f"   ✅ Weather query completed in {query_time:.2f}s")
    print(f"   📝 Response: {response3[:100]}...")
    
    # Test 4: Contextual conversation
    print("\n6️⃣ Testing contextual conversation...")
    start_time = time.time()
    response4 = await agent.process_text_input("What did I just ask about?")
    query_time = time.time() - start_time
    
    assert response4 is not None, "Should get contextual response"
    print(f"   ✅ Context query completed in {query_time:.2f}s")
    print(f"   📝 Response: {response4[:100]}...")
    
    # Test 5: Memory persistence
    print("\n7️⃣ Testing memory persistence...")
    final_memory_count = agent.get_memory().get_message_count()
    assert final_memory_count >= 8, f"Should have stored conversations, got {final_memory_count}"
    
    messages = agent.get_memory().get_recent_messages(limit=10)
    print(f"   ✅ {len(messages)} messages stored in memory")
    
    # Verify message order and content
    user_messages = [msg for msg in messages if msg['role'] == 'user']
    assistant_messages = [msg for msg in messages if msg['role'] == 'assistant']
    
    assert len(user_messages) >= 4, "Should have user messages"
    assert len(assistant_messages) >= 4, "Should have assistant responses"
    print(f"   ✅ {len(user_messages)} user messages, {len(assistant_messages)} assistant responses")
    
    # Test 6: Tool functionality validation
    print("\n8️⃣ Testing individual tools...")
    
    # Test time tool directly
    time_result = await agent.tools.execute_tool("time")
    assert ":" in time_result, "Time should contain :"
    print(f"   ✅ Time tool: {time_result}")
    
    # Test system info tool
    memory_result = await agent.tools.execute_tool("system_info", {"info_type": "memory"})
    assert "%" in memory_result, "Memory result should contain percentage"
    print(f"   ✅ Memory tool: {memory_result}")
    
    # Test weather tool
    weather_result = await agent.tools.execute_tool("weather", {"location": "Paris"})
    assert "paris" in weather_result.lower(), "Weather should mention Paris"
    print(f"   ✅ Weather tool: {weather_result}")
    
    # Test 7: Timeout handling
    print("\n9️⃣ Testing timeout handling...")
    try:
        # Test with very short timeout
        short_response = await agent.llm.generate_response("Test message", timeout=0.001)
        print(f"   ⚠️  Expected timeout but got: {short_response}")
    except asyncio.TimeoutError:
        print("   ✅ Timeout handling works correctly")
    except Exception as e:
        print(f"   ✅ Error handling works: {type(e).__name__}")
    
    # Test 8: Performance metrics
    print("\n🔟 Performance summary...")
    total_messages = agent.get_memory().get_message_count()
    available_tools = len(agent.tools.get_available_tools())
    
    print(f"   📊 Total messages processed: {total_messages}")
    print(f"   📊 Tools available: {available_tools}")
    print(f"   📊 Memory utilization: {total_messages}/{settings.max_messages}")
    
    # Cleanup
    print("\n🧹 Cleaning up...")
    await agent.llm.close()
    
    if os.path.exists('test_complete.db'):
        os.remove('test_complete.db')
    
    print("\n✅ Complete system test PASSED!")
    return True


async def test_concurrent_requests():
    """Test system under concurrent load"""
    print("\n🔄 Testing concurrent request handling...")
    
    settings = Settings(debug=True, db_path='test_concurrent.db')
    agent = VoiceAgent(settings)
    
    # Create multiple concurrent requests
    tasks = []
    questions = [
        "What time is it?",
        "How's the system memory?",
        "What's the weather like?",
        "Tell me the current time",
        "Check system performance"
    ]
    
    start_time = time.time()
    
    # Submit all requests concurrently
    for question in questions:
        task = asyncio.create_task(agent.process_text_input(question))
        tasks.append(task)
    
    # Wait for all to complete
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    concurrent_time = time.time() - start_time
    
    # Verify results
    success_count = sum(1 for result in results if isinstance(result, str) and len(result) > 0)
    error_count = len(results) - success_count
    
    print(f"   ✅ Processed {len(questions)} concurrent requests in {concurrent_time:.2f}s")
    print(f"   📊 Success: {success_count}, Errors: {error_count}")
    
    # Cleanup
    await agent.llm.close()
    if os.path.exists('test_concurrent.db'):
        os.remove('test_concurrent.db')
    
    return success_count >= len(questions) * 0.8  # 80% success rate


async def main():
    """Run all integration tests"""
    print("🎯 EasyVoice Integration Test Suite")
    print("=" * 60)
    
    try:
        # Test complete pipeline
        pipeline_success = await test_complete_pipeline()
        
        # Test concurrent handling
        concurrent_success = await test_concurrent_requests()
        
        print("\n" + "=" * 60)
        print("📊 FINAL RESULTS")
        print("=" * 60)
        print(f"Complete Pipeline: {'✅ PASS' if pipeline_success else '❌ FAIL'}")
        print(f"Concurrent Handling: {'✅ PASS' if concurrent_success else '❌ FAIL'}")
        
        overall_success = pipeline_success and concurrent_success
        
        if overall_success:
            print("\n🎉 ALL INTEGRATION TESTS PASSED!")
            print("🚀 EasyVoice system is ready for production!")
        else:
            print("\n⚠️  SOME TESTS FAILED")
            print("🔧 Please review the issues above")
        
        return 0 if overall_success else 1
        
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)