"""Conversation memory with 20-message sliding window"""

import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from easyvoice.config.settings import Settings

logger = logging.getLogger(__name__)


class ConversationMemory:
    """Manages conversation history with sliding window"""

    def __init__(self, settings: Settings):
        """Initialize conversation memory

        Args:
            settings: EasyVoice configuration settings
        """
        self.settings = settings
        self.db_path = Path(settings.db_path)
        self.max_messages = settings.max_messages

        # Ensure database directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_database()

    def _init_database(self) -> None:
        """Initialize SQLite database schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """
            )
            conn.commit()

    def add_message(
        self, role: str, content: str, timestamp: Optional[str] = None
    ) -> None:
        """Add a message to conversation memory

        Args:
            role: Message role ('user' or 'assistant')
            content: Message content
            timestamp: Optional timestamp (ISO format), defaults to now
        """
        if not timestamp:
            timestamp = datetime.now().isoformat()

        with sqlite3.connect(self.db_path) as conn:
            # Add the new message
            conn.execute(
                "INSERT INTO conversations (role, content, timestamp) VALUES (?, ?, ?)",
                (role, content, timestamp),
            )

            # Enforce sliding window - remove oldest messages if over limit
            count_result = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()
            current_count = count_result[0]

            if current_count > self.max_messages:
                excess = current_count - self.max_messages
                conn.execute(
                    f"""
                    DELETE FROM conversations
                    WHERE id IN (
                        SELECT id FROM conversations
                        ORDER BY created_at ASC
                        LIMIT {excess}
                    )
                """
                )

            conn.commit()

        logger.debug(f"Added {role} message to memory: {content[:50]}...")

    def get_recent_messages(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get recent messages from memory

        Args:
            limit: Maximum number of messages to return

        Returns:
            List of message dictionaries with role, content, timestamp
        """
        if limit is None:
            limit = self.max_messages

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            cursor = conn.execute(
                """
                SELECT role, content, timestamp
                FROM conversations
                ORDER BY created_at ASC
                LIMIT ?
            """,
                (limit,),
            )

            return [dict(row) for row in cursor.fetchall()]

    def get_all_messages(self) -> List[Dict[str, Any]]:
        """Get all messages in chronological order

        Returns:
            List of all message dictionaries
        """
        return self.get_recent_messages(limit=None)

    def get_message_count(self) -> int:
        """Get total number of messages in memory

        Returns:
            Number of messages currently stored
        """
        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()
            if result is None:
                return 0
            return int(result[0])

    def clear_all(self) -> None:
        """Clear all messages from memory"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM conversations")
            conn.commit()

        logger.info("Conversation memory cleared")

    def get_context_for_llm(
        self, include_system_prompt: bool = True
    ) -> List[Dict[str, str]]:
        """Get conversation context formatted for LLM

        Args:
            include_system_prompt: Whether to include system prompt

        Returns:
            List of messages formatted for LLM consumption
        """
        messages = []

        if include_system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "You are a helpful voice assistant. "
                        "Respond conversationally and concisely."
                    ),
                }
            )

        # Add conversation history
        recent_messages = self.get_recent_messages()
        messages.extend(
            [
                {"role": msg["role"], "content": msg["content"]}
                for msg in recent_messages
            ]
        )

        return messages

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics

        Returns:
            Dictionary with memory statistics
        """
        message_count = self.get_message_count()

        with sqlite3.connect(self.db_path) as conn:
            # Get oldest and newest message timestamps
            oldest = conn.execute(
                "SELECT timestamp FROM conversations ORDER BY created_at ASC LIMIT 1"
            ).fetchone()

            newest = conn.execute(
                "SELECT timestamp FROM conversations ORDER BY created_at DESC LIMIT 1"
            ).fetchone()

            # Get role distribution
            role_stats = conn.execute(
                """
                SELECT role, COUNT(*) as count
                FROM conversations
                GROUP BY role
            """
            ).fetchall()

        return {
            "total_messages": message_count,
            "max_capacity": self.max_messages,
            "utilization_percent": (message_count / self.max_messages) * 100,
            "oldest_message": oldest[0] if oldest else None,
            "newest_message": newest[0] if newest else None,
            "role_distribution": dict(role_stats),
        }
