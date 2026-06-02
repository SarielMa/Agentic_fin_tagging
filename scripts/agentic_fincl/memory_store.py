from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MemoryWrite:
    namespace: str
    payload: dict[str, Any]


class LTMStore:
    """Append-only long-term memory used by the agentic loop."""

    VALID_NAMESPACES = {"selector_memory", "error_book", "table_context_patterns"}

    def __init__(self, root_dir: Path, write_enabled: bool = True) -> None:
        self.root_dir = root_dir
        self.write_enabled = write_enabled
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.selector_memory = self._load_namespace("selector_memory")
        self.error_book = self._load_namespace("error_book")
        self.table_context_patterns = self._load_namespace("table_context_patterns")

    def set_write_enabled(self, enabled: bool) -> None:
        self.write_enabled = enabled

    def snapshot(self) -> dict[str, int]:
        return {
            "selector_memory": len(self.selector_memory),
            "error_book": len(self.error_book),
            "table_context_patterns": len(self.table_context_patterns),
        }

    def records(self, namespace: str) -> list[dict[str, Any]]:
        if namespace not in self.VALID_NAMESPACES:
            raise ValueError(f"Unknown LTM namespace: {namespace}")
        return getattr(self, namespace)

    def append(self, write: MemoryWrite) -> None:
        if write.namespace not in self.VALID_NAMESPACES:
            raise ValueError(f"Unknown LTM namespace: {write.namespace}")
        if not self.write_enabled:
            return

        payload = dict(write.payload)
        payload.setdefault("memory_id", f"{write.namespace}-{len(self.records(write.namespace)) + 1}")
        self.records(write.namespace).append(payload)
        with self._path(write.namespace).open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def append_many(self, writes: list[MemoryWrite]) -> None:
        for write in writes:
            self.append(write)

    def _load_namespace(self, namespace: str) -> list[dict[str, Any]]:
        path = self._path(namespace)
        if not path.exists():
            return []
        with path.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def _path(self, namespace: str) -> Path:
        return self.root_dir / f"{namespace}.jsonl"


def write_from_dataclass(namespace: str, item: Any) -> MemoryWrite:
    return MemoryWrite(namespace=namespace, payload=asdict(item))
