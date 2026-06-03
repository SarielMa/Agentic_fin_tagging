from __future__ import annotations

from typing import Any, Mapping


def model_input_device(model: Any) -> Any:
    """Return the device that token ids should use for this causal LM."""
    get_embeddings = getattr(model, "get_input_embeddings", None)
    if get_embeddings is not None:
        embeddings = get_embeddings()
        weight = getattr(embeddings, "weight", None)
        device = getattr(weight, "device", None)
        if device is not None and getattr(device, "type", None) != "meta":
            return device

    try:
        return next(model.parameters()).device
    except StopIteration:
        return None


def move_inputs_to_device(inputs: Mapping[str, Any], device: Any) -> dict[str, Any]:
    if device is None:
        return dict(inputs)
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
