"""Interactive CLI for EasyVoice without flags"""

import asyncio
import signal
import sys
from typing import Any, Optional

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from easyvoice import __version__
from easyvoice.agent.core import VoiceAgent
from easyvoice.config.settings import Settings

console = Console()

# Style constants
STYLE_BOLD_GREEN = "bold green"
STYLE_BOLD_CYAN = "bold cyan"
STYLE_BOLD_BLUE = "bold blue"
AUDIO_LEVEL_TEXT = "🎤 [bold blue]Audio Level"
PROGRESS_PERCENTAGE = "[progress.percentage]{task.percentage:>3.0f}%"
DB_DISPLAY = "({task.completed:.1f} dB)"
ERROR_AGENT_NOT_INITIALIZED = "[red]Agent not initialized[/red]"


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
            banner_text = Text(banner, style=STYLE_BOLD_CYAN)
        except ImportError:
            banner_text = Text("🎤 EasyVoice CLI", style=STYLE_BOLD_CYAN)

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
        table.add_column("Command", style=STYLE_BOLD_GREEN, width=20)
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
                    console.print("✅ Voice agent ready!", style=STYLE_BOLD_GREEN)
                else:
                    console.print(
                        "⚠️ Voice agent partially ready",
                        style="bold yellow",
                    )

    async def handle_chat(self) -> None:
        """Handle text chat mode"""
        await self.initialize_agent()

        console.print("💬 Starting text chat mode", style=STYLE_BOLD_BLUE)
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
                        console.print(ERROR_AGENT_NOT_INITIALIZED)
                        break
                    response = await self.agent.process_text_input(user_input)

                console.print(
                    f"[{STYLE_BOLD_CYAN}]🤖 Agent:[/{STYLE_BOLD_CYAN}] {response}"
                )
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
            question = Prompt.ask(f"[{STYLE_BOLD_BLUE}]What would you like to ask?")

            if question.strip():
                with console.status("[bold yellow]🤔 Processing...", spinner="dots"):
                    if self.agent is None:
                        console.print(ERROR_AGENT_NOT_INITIALIZED)
                        return
                    response = await self.agent.process_text_input(question)

                console.print(
                    f"\n[{STYLE_BOLD_CYAN}]🤖 Response:[/{STYLE_BOLD_CYAN}] {response}"
                )
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
            llm_status = (
                "✅ Ready"
                if (llm_config.get("has_httpx") or llm_config.get("has_openai"))
                else "⚠️ Mock Mode"
            )
            table.add_row("LLM", llm_status, f"Model: {llm_config.get('model_name')}")
        else:
            table.add_row("Voice Agent", "❌ Not initialized", "")

        console.print(table)
        console.print()

    async def handle_test(self) -> None:
        """Test audio system"""
        console.print("🔧 Testing audio system...", style=STYLE_BOLD_BLUE)

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

        console.print("\n✅ Audio system test completed", style=STYLE_BOLD_GREEN)
        console.print(
            "Install full dependencies for real audio processing", style="dim"
        )
        console.print()

    def _calculate_audio_level(self, audio_chunk: np.ndarray) -> float:
        """Calculate decibel level from audio chunk"""
        rms = np.sqrt(np.mean(audio_chunk**2))
        if rms > 0:
            db_level: float = 20 * np.log10(rms)
        else:
            db_level = -60.0  # Silence floor

        # Normalize -60dB to 0dB -> 0-100%
        return max(0.0, min(100.0, (db_level + 60.0) * 100.0 / 60.0))

    def _should_stop_recording(
        self,
        current_time: float,
        start_time: float,
        last_speech_time: float,
        speech_detected: bool,
        max_duration: float,
        silence_duration: float,
    ) -> bool:
        """Check if recording should stop"""
        # Check for timeout
        if current_time - start_time > max_duration:
            return True

        # If no speech detected and we've waited enough, stop
        if not speech_detected and current_time - start_time > 3.0:
            return True

        # Check for silence duration after speech
        if speech_detected and current_time - last_speech_time > silence_duration:
            return True

        return False

    def _process_audio_chunk(
        self, audio_input: Any, progress: Any, db_task: Any
    ) -> bool:
        """Process audio chunk and update visualization"""
        speech_detected = False

        with audio_input.buffer_lock:
            if audio_input.audio_buffer:
                recent_chunk = audio_input.audio_buffer[-1]

                # Update progress bar with audio level
                db_normalized = self._calculate_audio_level(recent_chunk)
                progress.update(db_task, completed=db_normalized)

                # Check for voice activity
                if audio_input.vad.process_chunk(recent_chunk):
                    speech_detected = True

        return speech_detected

    async def _record_with_visualization(self, audio_input: Any) -> np.ndarray:
        """Record audio with real-time decibel meter and waveform visualization"""
        import time

        from rich.live import Live

        if not self.settings.show_waveform:
            result: np.ndarray = await audio_input.record_until_silence(
                max_duration=30.0, silence_duration=2.0
            )
            return result

        # Start recording
        await audio_input.start_recording()

        try:
            start_time = time.time()
            last_speech_time = start_time
            speech_detected = False
            max_duration = 30.0
            silence_duration = 2.0

            # Create progress bar for decibel meter
            progress = Progress(
                TextColumn(AUDIO_LEVEL_TEXT),
                BarColumn(bar_width=40),
                TextColumn(PROGRESS_PERCENTAGE),
                TextColumn(DB_DISPLAY),
            )

            with Live(progress, refresh_per_second=10):
                db_task = progress.add_task("Recording", total=100)

                while True:
                    current_time = time.time()
                    chunk_speech = self._process_audio_chunk(
                        audio_input, progress, db_task
                    )

                    if chunk_speech:
                        last_speech_time = current_time
                        speech_detected = True

                    # Check if we should stop recording
                    if self._should_stop_recording(
                        current_time,
                        start_time,
                        last_speech_time,
                        speech_detected,
                        max_duration,
                        silence_duration,
                    ):
                        if not speech_detected and current_time - start_time > 3.0:
                            return np.array([])
                        break

                    await asyncio.sleep(0.1)

            # Get all recorded data
            audio_data: np.ndarray = audio_input.get_audio_data()
            return audio_data

        finally:
            await audio_input.stop_recording()

    async def _record_with_persistent_meter(
        self, audio_input: Any, progress: Any, db_task: Any
    ) -> np.ndarray:
        """Record audio with persistent meter updates"""
        import time

        await audio_input.start_recording()

        try:
            start_time = time.time()
            last_speech_time = start_time
            speech_detected = False
            max_duration = 30.0
            silence_duration = 2.0

            while True:
                current_time = time.time()
                chunk_speech = self._process_audio_chunk(audio_input, progress, db_task)

                if chunk_speech:
                    last_speech_time = current_time
                    speech_detected = True

                # Check if we should stop recording
                if self._should_stop_recording(
                    current_time,
                    start_time,
                    last_speech_time,
                    speech_detected,
                    max_duration,
                    silence_duration,
                ):
                    if not speech_detected and current_time - start_time > 3.0:
                        return np.array([])
                    break

                await asyncio.sleep(0.1)

            # Get all recorded data
            audio_data: np.ndarray = audio_input.get_audio_data()
            return audio_data

        finally:
            await audio_input.stop_recording()

    async def _transcribe_audio(self, audio_data: np.ndarray) -> Optional[str]:
        """Transcribe audio data to text"""
        try:
            from easyvoice.audio.stt_openai import OpenAIWhisperSTT

            stt = OpenAIWhisperSTT(self.settings)
            user_text = await stt.transcribe_audio(audio_data)
            if not user_text or not user_text.strip():
                return None
            return user_text
        except ImportError:
            console.print(
                "OpenAI Whisper not available - using placeholder", style="yellow"
            )
            return "Hello, can you hear me?"
        except Exception as e:
            console.print(f"STT error: {e}", style="red")
            return None

    async def _process_voice_input(self, audio_input: Any) -> Optional[str]:
        """Process a single voice input cycle"""

        # Record until silence with audio visualization (includes listening message)
        audio_data = await self._record_with_visualization(audio_input)

        if len(audio_data) == 0:
            return None

        console.print("🤔 Processing...", style="yellow")

        # Process audio through STT
        try:
            from easyvoice.audio.stt_openai import OpenAIWhisperSTT

            stt = OpenAIWhisperSTT(self.settings)
            user_text = await stt.transcribe_audio(audio_data)
            if not user_text or not user_text.strip():
                console.print("No speech detected, continuing...", style="dim")
                return None
            console.print(f"You said: '{user_text}'", style="dim")
            return user_text
        except ImportError:
            console.print(
                "OpenAI Whisper not available - using placeholder", style="yellow"
            )
            return "Hello, can you hear me?"
        except Exception as e:
            console.print(f"STT error: {e}", style="red")
            return None

    async def _handle_voice_response(self, user_text: str, tts: Any) -> None:
        """Generate and speak response to user input"""
        if self.agent is None:
            console.print(ERROR_AGENT_NOT_INITIALIZED)
            return

        # Generate response
        response = await self.agent.process_text_input(user_text)

        console.print(f"🤖 Agent: {response}", style=STYLE_BOLD_CYAN)

        # Speak response
        await tts.synthesize_and_play(response, wait=True)

    async def _simple_voice_loop(self, audio_input: Any, tts: Any) -> None:
        """Simple voice loop without visualization"""
        while self.running:
            try:
                user_text = await self._process_voice_input(audio_input)
                if not user_text:
                    continue
                await self._handle_voice_response(user_text, tts)
            except KeyboardInterrupt:
                break
            except Exception as e:
                console.print(f"[red]Voice error: {e}[/red]")
                break

    async def _visual_voice_loop(self, audio_input: Any, tts: Any) -> None:
        """Voice loop with persistent visual meter"""
        from rich.live import Live

        # Create persistent progress bar for decibel meter
        progress = Progress(
            TextColumn(AUDIO_LEVEL_TEXT),
            BarColumn(bar_width=40),
            TextColumn(PROGRESS_PERCENTAGE),
            TextColumn(DB_DISPLAY),
        )

        with Live(progress, refresh_per_second=10):
            db_task = progress.add_task("Listening", total=100)

            while self.running:
                try:
                    # Record audio with live meter updates
                    audio_data = await self._record_with_persistent_meter(
                        audio_input, progress, db_task
                    )

                    if len(audio_data) == 0:
                        continue

                    # Process the recorded audio
                    user_text = await self._transcribe_audio(audio_data)
                    if not user_text:
                        continue

                    # Temporarily pause meter updates for response
                    progress.update(db_task, description="🤖 [bold cyan]Responding...")
                    await self._handle_voice_response(user_text, tts)
                    progress.update(db_task, description=AUDIO_LEVEL_TEXT)

                except KeyboardInterrupt:
                    break
                except Exception as e:
                    console.print(f"[red]Voice error: {e}[/red]")
                    break

    async def _voice_conversation_loop(self, audio_input: Any, tts: Any) -> None:
        """Run voice conversation with persistent audio meter"""
        if not self.settings.show_waveform:
            await self._simple_voice_loop(audio_input, tts)
        else:
            await self._visual_voice_loop(audio_input, tts)

    async def handle_voice(self) -> None:
        """Handle voice conversation mode"""
        await self.initialize_agent()

        console.print("🎤 Starting voice conversation mode", style=STYLE_BOLD_BLUE)
        console.print("Say something to start... (Ctrl+C to exit)\n", style="dim")

        try:
            from easyvoice.audio.input import AudioInput
            from easyvoice.audio.tts import KittenTTS

            # Initialize audio components
            audio_input = AudioInput(self.settings)
            tts = KittenTTS(self.settings)

            # Run voice conversation with persistent meter
            await self._voice_conversation_loop(audio_input, tts)

        except ImportError as e:
            console.print(f"[red]Audio dependencies missing: {e}[/red]")
        except Exception as e:
            console.print(f"[red]Voice mode error: {e}[/red]")

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

    async def _process_command(self, command: str) -> bool:
        """Process a single command. Returns True to continue, False to exit."""
        if command in ["quit", "exit", "q"]:
            return False
        elif command == "chat":
            await self.handle_chat()
        elif command == "voice":
            await self.handle_voice()
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
                f"Unknown command: '{command}'. Type 'help' for available commands.",
                style="red",
            )
        return True

    async def run(self) -> None:
        """Run the interactive CLI"""
        self.print_banner()
        console.print(
            "Welcome to EasyVoice! Type 'help' to see available commands.\n",
            style=STYLE_BOLD_GREEN,
        )

        while self.running:
            try:
                command = (
                    Prompt.ask(f"[{STYLE_BOLD_BLUE}]easyvoice", default="help")
                    .strip()
                    .lower()
                )
                console.print()

                if not await self._process_command(command):
                    break

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

        console.print("👋 Thanks for using EasyVoice!", style=STYLE_BOLD_GREEN)


async def main() -> None:
    """Main entry point"""
    cli = InteractiveCLI()
    await cli.run()


if __name__ == "__main__":
    asyncio.run(main())
