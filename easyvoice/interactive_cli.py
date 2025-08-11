"""Interactive CLI for EasyVoice without flags"""

import asyncio
import signal
import sys
from typing import Any, Optional

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from easyvoice import __version__
from easyvoice.agent.core import VoiceAgent
from easyvoice.config.settings import Settings

console = Console()


class InteractiveCLI:
    """Interactive CLI for EasyVoice"""

    def __init__(self) -> None:
        """Initialize interactive CLI"""
        self.settings = Settings(debug=True)  # Force debug mode for CLI
        self.agent: Optional[VoiceAgent] = None
        self.running = True

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle interrupt signals"""
        self.running = False
        console.print("\n👋 Goodbye!", style="bold yellow")
        sys.exit(0)

    def print_banner(self) -> None:
        """Print welcome banner"""
        try:
            import pyfiglet

            banner = pyfiglet.figlet_format("EasyVoice", font="slant")
            banner_text = Text(banner, style="bold cyan")
        except ImportError:
            banner_text = Text("🎤 EasyVoice CLI", style="bold cyan")

        subtitle = Text(f"Lightweight Voice Agent CLI v{__version__}", style="dim")

        panel = Panel.fit(
            f"{banner_text}\n{subtitle}", border_style="cyan", padding=(0, 2)
        )
        console.print(panel)

    def show_menu(self) -> None:
        """Show main menu"""
        table = Table(
            title="Available Commands", show_header=False, border_style="blue"
        )
        table.add_column("Command", style="bold green", width=20)
        table.add_column("Description", style="white")

        table.add_row("chat", "Start text conversation")
        table.add_row("voice", "Start voice conversation (requires audio)")
        table.add_row("ask", "Ask a single question")
        table.add_row("history", "View conversation history")
        table.add_row("status", "Show system status")
        table.add_row("test", "Test audio system")
        table.add_row("config", "Show configuration")
        table.add_row("help", "Show this menu")
        table.add_row("quit", "Exit EasyVoice")

        console.print(table)
        console.print()

    async def initialize_agent(self) -> None:
        """Initialize the voice agent"""
        if self.agent is None:
            with console.status(
                "[bold green]Initializing voice agent...", spinner="dots"
            ):
                self.agent = VoiceAgent(self.settings)

                # Test basic functionality
                if self.agent.is_ready():
                    console.print("✅ Voice agent ready!", style="bold green")
                else:
                    console.print(
                        "⚠️ Voice agent partially ready",
                        style="bold yellow",
                    )

    async def handle_chat(self) -> None:
        """Handle text chat mode"""
        await self.initialize_agent()

        console.print("💬 Starting text chat mode", style="bold blue")
        console.print("Type 'exit' to return to main menu\n", style="dim")

        while self.running:
            try:
                user_input = Prompt.ask("[bold green]You")

                if user_input.lower() in ["exit", "quit", "back"]:
                    break

                if not user_input.strip():
                    continue

                # Show thinking indicator
                with console.status("[bold yellow]🤔 Thinking...", spinner="dots"):
                    if self.agent is None:
                        console.print("[red]Agent not initialized[/red]")
                        break
                    response = await self.agent.process_text_input(user_input)

                console.print(f"[bold cyan]🤖 Agent:[/bold cyan] {response}")
                console.print()

            except KeyboardInterrupt:
                break
            except (EOFError, BrokenPipeError):
                # Handle stdin closing gracefully
                console.print("\n[dim]Input stream closed, exiting chat mode...[/dim]")
                break
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                break

    async def handle_ask(self) -> None:
        """Handle single question"""
        await self.initialize_agent()

        try:
            question = Prompt.ask("[bold blue]What would you like to ask?")

            if question.strip():
                with console.status("[bold yellow]🤔 Processing...", spinner="dots"):
                    if self.agent is None:
                        console.print("[red]Agent not initialized[/red]")
                        return
                    response = await self.agent.process_text_input(question)

                console.print(f"\n[bold cyan]🤖 Response:[/bold cyan] {response}")
                console.print()
        except (EOFError, BrokenPipeError):
            console.print("\n[dim]Input cancelled[/dim]")
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")

    def handle_history(self) -> None:
        """Show conversation history"""
        if not self.agent:
            console.print("⚠️  Agent not initialized yet", style="yellow")
            return

        memory = self.agent.get_memory()
        messages = memory.get_recent_messages(limit=20)

        if not messages:
            console.print("📭 No conversation history", style="dim")
            return

        table = Table(title="Conversation History", border_style="blue")
        table.add_column("Time", style="dim", width=10)
        table.add_column("Role", style="bold", width=10)
        table.add_column("Message", style="white")

        for msg in messages[-10:]:  # Show last 10
            timestamp = msg.get("timestamp", "")[:10] if msg.get("timestamp") else ""
            role = msg["role"].title()
            content = msg["content"]

            # Truncate long messages
            if len(content) > 80:
                content = content[:77] + "..."

            table.add_row(timestamp, role, content)

        console.print(table)
        console.print()

    async def handle_status(self) -> None:
        """Show system status"""
        await self.initialize_agent()

        # Create status table
        table = Table(title="System Status", border_style="green")
        table.add_column("Component", style="bold")
        table.add_column("Status", style="white")
        table.add_column("Details", style="dim")

        # Check agent status
        if self.agent:
            table.add_row(
                "Voice Agent", "✅ Ready" if self.agent.is_ready() else "⚠️ Partial", ""
            )

            # Memory status
            memory = self.agent.get_memory()
            msg_count = memory.get_message_count()
            table.add_row(
                "Memory",
                "✅ Active",
                f"{msg_count}/{self.settings.max_messages} messages",
            )

            # Tools status
            tools = self.agent.tools.get_available_tools()
            table.add_row("Tools", "✅ Available", f"{len(tools)} tools loaded")

            # LLM status
            llm_config = self.agent.llm.get_configuration()
            llm_status = "✅ Ready" if llm_config.get("has_httpx") else "⚠️ Mock Mode"
            table.add_row("LLM", llm_status, f"Model: {llm_config.get('model_name')}")
        else:
            table.add_row("Voice Agent", "❌ Not initialized", "")

        console.print(table)
        console.print()

    async def handle_test(self) -> None:
        """Test audio system"""
        console.print("🔧 Testing audio system...", style="bold blue")

        # Test microphone (mock for now)
        with console.status("Testing microphone...", spinner="dots"):
            await asyncio.sleep(1)
        console.print("🎤 Microphone: ⚠️ Mock mode (install sounddevice for real audio)")

        # Test STT (mock)
        with console.status("Testing speech recognition...", spinner="dots"):
            await asyncio.sleep(1)
        console.print("🗣️ Speech Recognition: ⚠️ Mock mode (install openai-whisper)")

        # Test TTS (mock)
        with console.status("Testing text-to-speech...", spinner="dots"):
            await asyncio.sleep(1)
        console.print("🔊 Text-to-Speech: ⚠️ Mock mode (install KittenTTS)")

        console.print("\n✅ Audio system test completed", style="bold green")
        console.print(
            "Install full dependencies for real audio processing", style="dim"
        )
        console.print()

    def handle_config(self) -> None:
        """Show configuration"""
        table = Table(title="Configuration", border_style="cyan")
        table.add_column("Setting", style="bold")
        table.add_column("Value", style="white")

        table.add_row("Sample Rate", f"{self.settings.sample_rate} Hz")
        table.add_row("Max Messages", str(self.settings.max_messages))
        table.add_row("LLM Model", self.settings.model_name)
        table.add_row("LLM Host", self.settings.ollama_host)
        table.add_row("STT Timeout", f"{self.settings.stt_timeout}s")
        table.add_row("TTS Timeout", f"{self.settings.tts_timeout}s")
        table.add_row("Debug Mode", str(self.settings.debug))

        console.print(table)
        console.print()

    async def run(self) -> None:
        """Run the interactive CLI"""
        self.print_banner()
        console.print(
            "Welcome to EasyVoice! Type 'help' to see available commands.\n",
            style="bold green",
        )

        while self.running:
            try:
                command = (
                    Prompt.ask("[bold blue]easyvoice", default="help").strip().lower()
                )
                console.print()

                if command in ["quit", "exit", "q"]:
                    break
                elif command == "chat":
                    await self.handle_chat()
                elif command == "voice":
                    console.print(
                        "🎤 Voice mode not yet implemented - use 'chat' for now",
                        style="yellow",
                    )
                elif command == "ask":
                    await self.handle_ask()
                elif command == "history":
                    self.handle_history()
                elif command == "status":
                    await self.handle_status()
                elif command == "test":
                    await self.handle_test()
                elif command == "config":
                    self.handle_config()
                elif command in ["help", "h", "?"]:
                    self.show_menu()
                else:
                    console.print(
                        f"Unknown command: '{command}'. "
                        "Type 'help' for available commands.",
                        style="red",
                    )

                console.print()

            except KeyboardInterrupt:
                break
            except (EOFError, BrokenPipeError):
                # Handle stdin closing gracefully
                console.print("\n[dim]Input stream closed, goodbye![/dim]")
                break
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                console.print()

        console.print("👋 Thanks for using EasyVoice!", style="bold green")


async def main() -> None:
    """Main entry point"""
    cli = InteractiveCLI()
    await cli.run()


if __name__ == "__main__":
    asyncio.run(main())
