"""EasyVoice CLI entry point and command definitions"""

import asyncio
import sys
from pathlib import Path
from typing import Optional

import click

try:
    import pyfiglet

    HAS_PYFIGLET = True
except ImportError:
    pyfiglet = None  # type: ignore
    HAS_PYFIGLET = False

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from easyvoice.config.settings import Settings

console = Console()


def print_banner() -> None:
    """Print EasyVoice ASCII banner"""
    if HAS_PYFIGLET:
        banner = pyfiglet.figlet_format("EasyVoice", font="slant")
        banner_text = Text(banner, style="bold cyan")
    else:
        banner_text = Text("🎤 EasyVoice CLI", style="bold cyan")

    subtitle = Text("Lightweight Voice Agent CLI", style="dim")
    author = Text("Created by Ken Kai - AI Developer", style="dim cyan")

    panel = Panel.fit(f"{banner_text}\n{subtitle}\n{author}", border_style="cyan", padding=(0, 2))
    console.print(panel)


@click.group(invoke_without_command=True)
@click.option("--version", is_flag=True, help="Show version information")
@click.option("--config", type=click.Path(), help="Path to configuration file")
@click.pass_context
def main(ctx: click.Context, version: bool, config: Optional[str]) -> None:
    """EasyVoice - Lightweight Voice Agent CLI

    A simple yet powerful voice agent that can listen, understand, and respond
    to your voice commands with memory and tool calling capabilities.
    """
    # Ensure context object exists
    ctx.ensure_object(dict)

    if version:
        from easyvoice import __version__

        console.print(f"EasyVoice CLI version {__version__}", style="bold green")
        return

    # Load configuration
    settings = Settings()
    if config:
        # TODO: Load from config file
        console.print(f"Loading config from: {config}", style="dim")

    ctx.obj["settings"] = settings

    # If no subcommand provided, show help
    if ctx.invoked_subcommand is None:
        print_banner()
        console.print("\nAvailable commands:", style="bold")
        console.print("  listen     Start interactive voice conversation")
        console.print("  ask        Ask a single question")
        console.print("  history    Show conversation history")
        console.print("  test-audio Test audio pipeline")
        console.print("\nUse 'easyvoice --help' for more information.")


@main.command()
@click.option("--timeout", "-t", type=int, help="Session timeout in seconds")
@click.option("--push-to-talk", "-p", is_flag=True, help="Use push-to-talk mode")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.pass_context
def listen(
    ctx: click.Context, timeout: Optional[int], push_to_talk: bool, verbose: bool
) -> None:
    """Start interactive voice conversation mode

    This will start the voice agent in listening mode where you can have
    ongoing conversations. The agent will remember context and can use tools.
    """
    settings: Settings = ctx.obj["settings"]

    if timeout:
        settings.session_timeout = timeout

    print_banner()
    console.print("🎤 Starting voice conversation mode...", style="bold green")

    if push_to_talk:
        console.print("📢 Push-to-talk mode enabled (Hold TAB to talk)", style="dim")
    else:
        console.print("🔊 Voice activity detection enabled", style="dim")

    try:
        # Import here to avoid circular imports
        from easyvoice.agent.core import VoiceAgent

        agent = VoiceAgent(settings)
        asyncio.run(
            agent.start_conversation(push_to_talk=push_to_talk, verbose=verbose)
        )

    except KeyboardInterrupt:
        console.print("\n👋 Voice conversation ended", style="bold yellow")
    except Exception as e:
        console.print(f"❌ Error: {e}", style="bold red")
        if verbose:
            import traceback

            console.print(traceback.format_exc(), style="dim red")
        sys.exit(1)


@main.command()
@click.argument("question", required=True)
@click.option("--voice", "-v", is_flag=True, help="Speak the response aloud")
@click.option("--save", "-s", is_flag=True, help="Save to conversation history")
@click.pass_context
def ask(ctx: click.Context, question: str, voice: bool, save: bool) -> None:
    """Ask a single question to the voice agent

    This is useful for quick queries without starting a full conversation session.

    Examples:
        easyvoice ask "What time is it?"
        easyvoice ask "What's the weather like?" --voice
    """
    settings: Settings = ctx.obj["settings"]

    console.print(f"🤔 Processing: {question}", style="bold blue")

    try:
        from easyvoice.agent.core import VoiceAgent

        async def run_agent() -> str:
            agent = VoiceAgent(settings)
            return await agent.process_question(
                question=question, speak_response=voice, save_to_history=save
            )

        response = asyncio.run(run_agent())

        console.print(f"🤖 Response: {response}", style="bold green")

    except Exception as e:
        console.print(f"❌ Error: {e}", style="bold red")
        sys.exit(1)


@main.command()
@click.option("--limit", "-l", type=int, default=10, help="Number of messages to show")
@click.option(
    "--format",
    "-f",
    type=click.Choice(["table", "json", "plain"]),
    default="table",
    help="Output format",
)
@click.pass_context
def history(ctx: click.Context, limit: int, format: str) -> None:
    """Show conversation history

    Display recent conversations stored in the agent's memory.
    """
    settings: Settings = ctx.obj["settings"]

    try:
        from easyvoice.agent.memory import ConversationMemory

        memory = ConversationMemory(settings)
        messages = memory.get_recent_messages(limit=limit)

        if not messages:
            console.print("📭 No conversation history found", style="dim")
            return

        if format == "json":
            import json

            console.print(json.dumps(messages, indent=2))
        elif format == "plain":
            for msg in messages:
                console.print(f"{msg['role']}: {msg['content']}")
        else:  # table format
            from rich.table import Table

            table = Table(title=f"Last {len(messages)} Messages")
            table.add_column("Time", style="dim")
            table.add_column("Role", style="bold")
            table.add_column("Content")

            for msg in messages:
                table.add_row(
                    msg.get("timestamp", ""),
                    msg["role"].title(),
                    (
                        msg["content"][:100] + "..."
                        if len(msg["content"]) > 100
                        else msg["content"]
                    ),
                )

            console.print(table)

    except Exception as e:
        console.print(f"❌ Error retrieving history: {e}", style="bold red")
        sys.exit(1)


@main.command("test-audio")
@click.option("--duration", "-d", type=int, default=3, help="Test duration in seconds")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed output")
@click.pass_context
def test_audio(ctx: click.Context, duration: int, verbose: bool) -> None:
    """Test the audio pipeline

    This command tests the microphone input, speech recognition,
    text-to-speech, and audio output to ensure everything is working.
    """
    settings: Settings = ctx.obj["settings"]

    console.print("🔧 Testing audio pipeline...", style="bold blue")

    try:
        from easyvoice.audio.input import test_microphone
        from easyvoice.audio.stt import test_speech_recognition
        from easyvoice.audio.tts import test_text_to_speech

        # Test 1: Microphone input
        console.print("1️⃣ Testing microphone input...")
        if test_microphone(duration=2, verbose=verbose):
            console.print("   ✅ Microphone working", style="green")
        else:
            console.print("   ❌ Microphone failed", style="red")
            return

        # Test 2: Speech recognition
        console.print("2️⃣ Testing speech recognition...")
        console.print(f"   🎤 Say something for {duration} seconds...")

        result = asyncio.run(
            test_speech_recognition(
                duration=duration, settings=settings, verbose=verbose
            )
        )

        if result:
            console.print(f"   ✅ Recognized: '{result}'", style="green")
        else:
            console.print("   ❌ Speech recognition failed", style="red")
            return

        # Test 3: Text-to-speech
        console.print("3️⃣ Testing text-to-speech...")
        test_text = "Audio pipeline test successful!"

        if asyncio.run(
            test_text_to_speech(text=test_text, settings=settings, verbose=verbose)
        ):
            console.print("   ✅ Text-to-speech working", style="green")
        else:
            console.print("   ❌ Text-to-speech failed", style="red")
            return

        console.print("🎉 All audio tests passed!", style="bold green")

    except Exception as e:
        console.print(f"❌ Audio test failed: {e}", style="bold red")
        if verbose:
            import traceback

            console.print(traceback.format_exc(), style="dim red")
        sys.exit(1)


# Hidden development commands
@main.group(hidden=True)
def dev() -> None:
    """Development commands (hidden)"""
    pass


@dev.command()
@click.pass_context
def reset_memory(ctx: click.Context) -> None:
    """Reset conversation memory database"""
    settings: Settings = ctx.obj["settings"]

    db_path = Path(settings.db_path)
    if db_path.exists():
        db_path.unlink()
        console.print("🗑️ Memory database reset", style="bold yellow")
    else:
        console.print("📭 No memory database found", style="dim")


@dev.command()
@click.pass_context
def show_config(ctx: click.Context) -> None:
    """Show current configuration"""
    settings: Settings = ctx.obj["settings"]

    from rich.pretty import pprint

    console.print("⚙️ Current Configuration:", style="bold")
    pprint(settings.__dict__)


if __name__ == "__main__":
    main()
