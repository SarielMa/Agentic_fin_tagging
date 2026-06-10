from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import Any


class TableRowTextParser(HTMLParser):
    """Extract row-level text from simple serialized HTML tables."""

    def __init__(self) -> None:
        super().__init__()
        self.in_row = False
        self.in_cell = False
        self.current_cell: list[str] = []
        self.current_row: list[str] = []
        self.rows: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self.in_row = True
            self.current_row = []
        elif tag in {"td", "th"} and self.in_row:
            self.in_cell = True
            self.current_cell = []

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self.in_cell:
            cell_text = normalize_space("".join(self.current_cell))
            if cell_text:
                self.current_row.append(cell_text)
            self.current_cell = []
            self.in_cell = False
        elif tag == "tr" and self.in_row:
            row_text = " | ".join(self.current_row)
            if row_text:
                self.rows.append(row_text)
            self.current_row = []
            self.in_row = False

    def handle_data(self, data: str) -> None:
        if self.in_cell:
            self.current_cell.append(data)


def normalize_space(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text).replace("\xa0", " ")).strip()


def normalize_value(value: Any) -> str:
    text = normalize_space(value)
    text = text.replace("$", "").replace(",", "").replace("%", "")
    text = text.replace("\u2014", "—")
    text = re.sub(r"\s+", "", text)
    if re.fullmatch(r"\(.+\)", text):
        text = text[1:-1]
    return text.lower()


def strip_html(text: str) -> str:
    text = re.sub(r"</t[dh]>", " | ", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    return normalize_space(text)


def table_rows(context: str) -> list[str]:
    parser = TableRowTextParser()
    parser.feed(context)
    return parser.rows


def row_contains_entity(row: str, entity: Any) -> bool:
    entity_norm = normalize_value(entity)
    cells = re.split(r"\|", row)
    if any(normalize_value(cell) == entity_norm for cell in cells):
        return True
    return entity_norm in normalize_value(row)


def localize_context(context: str, category: str, entity: Any, max_chars: int = 1800) -> str:
    """Create STM evidence from the original full context.

    The FinCL CSV's serialized ``query`` field is intentionally not used. For
    tables, rows containing the target value provide compact structure-aware
    evidence. If the value cannot be localized exactly, the function falls back
    to a truncated plain-text version of the full context.
    """
    if category == "table":
        matched_rows = [row for row in table_rows(context) if row_contains_entity(row, entity)]
        if matched_rows:
            return normalize_space(" [ROW] ".join(matched_rows))[:max_chars]
        return strip_html(context)[:max_chars]

    plain = strip_html(context)
    idx = plain.lower().find(str(entity).lower())
    if idx < 0:
        return plain[:max_chars]
    half = max_chars // 2
    start = max(0, idx - half)
    end = min(len(plain), idx + half)
    return plain[start:end]


def build_query_text(entity: Any, entity_type: str, evidence: str) -> str:
    return normalize_space(f"entity {entity} type {entity_type} context {evidence}")


def rewrite_evidence_for_retrieval(evidence: str, category: str, entity: Any, entity_type: str) -> str:
    """Normalize evidence into one retrieval-query style across categories."""
    evidence_text = normalize_space(str(evidence).replace("[ROW]", " row: ").replace("|", ";"))
    fields = [
        ("format", "table_as_text" if category == "table" else "narrative_text"),
        ("entity", str(entity)),
        ("entity_type", entity_type),
        ("target_value", str(entity)),
        ("text_evidence", evidence_text),
    ]
    return normalize_space(" ".join(f"{key}: {value}" for key, value in fields if value))
