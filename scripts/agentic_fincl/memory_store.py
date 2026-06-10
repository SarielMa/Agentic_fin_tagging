# Stores the two validator-written LTM books as append-only JSONL files.
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class LTMStore:
    BOOKS = ("correct_book", "error_book")

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.correct_book = self._load("correct_book")
        self.error_book = self._load("error_book")

    def records(self, book: str) -> list[dict[str, Any]]:
        if book not in self.BOOKS:
            raise ValueError(f"Unknown memory book: {book}")
        return getattr(self, book)

    def write_correct(self, entity_type: str, value: str, context: str, tag: str, comment: str) -> None:
        self._append(
            "correct_book",
            {
                "entity_type": entity_type,
                "value": value,
                "context": context,
                "tag": tag,
                "comment": comment,
            },
        )

    def write_error(
        self,
        entity_type: str,
        value: str,
        context: str,
        predicted_tag: str,
        correct_tag: str,
        comment: str,
    ) -> None:
        self._append(
            "error_book",
            {
                "entity_type": entity_type,
                "value": value,
                "context": context,
                "predicted_tag": predicted_tag,
                "correct_tag": correct_tag,
                "comment": comment,
            },
        )

    def snapshot(self) -> dict[str, int]:
        return {book: len(self.records(book)) for book in self.BOOKS}

    def has_records(self) -> bool:
        return any(self.records(book) for book in self.BOOKS)

    def _append(self, book: str, payload: dict[str, Any]) -> None:
        payload = dict(payload)
        payload["memory_id"] = f"{book}-{len(self.records(book)) + 1}"
        self.records(book).append(payload)
        with self._path(book).open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _load(self, book: str) -> list[dict[str, Any]]:
        path = self._path(book)
        if not path.exists():
            return []
        with path.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def _path(self, book: str) -> Path:
        return self.root_dir / f"{book}.jsonl"
