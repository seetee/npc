"""In-memory conversation history, trimmed to a message limit."""

from __future__ import annotations


class ConversationHistory:
    def __init__(self, limit: int = 30):
        self.limit = limit
        self._messages: list[dict[str, str]] = []

    def add_player(self, text: str) -> None:
        self._append("user", f'PLAYER (spoken): "{text}"')

    def add_ooc(self, text: str) -> None:
        self._append("user", f"GM NOTE (out-of-character): {text}")

    def add_npc(self, text: str) -> None:
        self._append("assistant", text)

    def _append(self, role: str, content: str) -> None:
        self._messages.append({"role": role, "content": content})
        if len(self._messages) > self.limit:
            del self._messages[: len(self._messages) - self.limit]

    def as_messages(self) -> list[dict[str, str]]:
        return list(self._messages)

    def __len__(self) -> int:
        return len(self._messages)
