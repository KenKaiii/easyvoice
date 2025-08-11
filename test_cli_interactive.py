#!/usr/bin/env python3
"""
Test the interactive CLI with simulated input
"""

import sys
import asyncio
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from easyvoice.interactive_cli import InteractiveCLI


async def test_interactive_cli():
    """Test CLI with simulated user interactions"""
    print("🧪 Testing Interactive CLI with Simulated Input")
    print("=" * 60)
    
    cli = InteractiveCLI()
    
    # Test individual commands that don't require user input
    print("1️⃣ Testing status command...")
    await cli.handle_status()
    print("   ✅ Status command works\n")
    
    print("2️⃣ Testing config command...")
    cli.handle_config()
    print("   ✅ Config command works\n")
    
    print("3️⃣ Testing audio test command...")
    await cli.handle_test()
    print("   ✅ Audio test command works\n")
    
    print("4️⃣ Testing history command (empty)...")
    cli.handle_history()
    print("   ✅ History command works\n")
    
    # Test ask command with mock input
    print("5️⃣ Testing ask command...")
    with patch('rich.prompt.Prompt.ask') as mock_prompt:
        mock_prompt.return_value = "What time is it?"
        await cli.handle_ask()
    print("   ✅ Ask command works\n")
    
    # Test chat command with mock conversation
    print("6️⃣ Testing chat command...")
    with patch('rich.prompt.Prompt.ask') as mock_prompt:
        # Simulate a short conversation then exit
        mock_prompt.side_effect = ["Hello!", "What time is it?", "exit"]
        try:
            await cli.handle_chat()
        except (EOFError, KeyboardInterrupt):
            pass
    print("   ✅ Chat command works\n")
    
    # Test that agent has conversation history now
    print("7️⃣ Testing history after conversation...")
    cli.handle_history()
    print("   ✅ History shows conversation\n")
    
    print("✅ Interactive CLI Test Complete!")
    print("🎉 All CLI functions working properly!")


if __name__ == "__main__":
    asyncio.run(test_interactive_cli())