#!/usr/bin/env python3
"""
Main entry point for EasyVoice CLI
Allows running with: python -m easyvoice
"""

import sys
import asyncio
from easyvoice.interactive_cli import main


def cli_main():
    """Synchronous entry point for the CLI"""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    cli_main()