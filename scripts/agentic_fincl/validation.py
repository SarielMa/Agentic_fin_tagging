from __future__ import annotations

from typing import Any


def validate_prediction(entity_type: str, candidates: list[dict[str, Any]], selected_tag: str) -> dict[str, Any]:
    """Step 4 validator for the initial pipeline."""
    top_score = candidates[0]["score"] if candidates else 0.0
    second_score = candidates[1]["score"] if len(candidates) > 1 else 0.0
    selected = next((candidate for candidate in candidates if candidate["tag"] == selected_tag), None)

    flags = []
    if selected is None:
        flags.append("selected_tag_not_in_candidates")
    elif selected["entity_type"] != entity_type:
        flags.append("selected_type_mismatch")
    if top_score - second_score < 0.01:
        flags.append("low_top2_gap")

    return {
        "status": "flagged" if flags else "ok",
        "flags": flags,
        "top2_gap": float(top_score - second_score),
    }
