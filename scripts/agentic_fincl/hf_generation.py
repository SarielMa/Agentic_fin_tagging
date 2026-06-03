from __future__ import annotations

import os
from typing import Any, Mapping


_LOCAL_CAUSAL_LM_CACHE: dict[tuple[str, str, str, str], tuple[Any, Any, Any]] = {}


def local_files_only() -> bool:
    return os.environ.get("FINAI_LOCAL_FILES_ONLY", "0").lower() in {"1", "true", "yes", "on"}


def load_local_causal_lm(
    model_name: str,
    torch: Any,
    auto_model_for_causal_lm: Any,
    auto_tokenizer: Any,
    device: str | None = None,
) -> tuple[Any, Any, Any]:
    """Load or reuse a local causal LM and tokenizer for this process."""
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device_map = "auto" if torch.cuda.is_available() and device is None else None
    local_only = local_files_only()
    cache_device = "auto" if device_map is not None else str(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cache_key = (model_name, str(dtype), cache_device, str(local_only))

    if cache_key not in _LOCAL_CAUSAL_LM_CACHE:
        tokenizer = auto_tokenizer.from_pretrained(model_name, local_files_only=local_only)
        model = auto_model_for_causal_lm.from_pretrained(
            model_name,
            local_files_only=local_only,
            torch_dtype=dtype,
            device_map=device_map,
        )
        explicit_device = None
        if device_map is None:
            explicit_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
            model.to(explicit_device)
        model.eval()
        _LOCAL_CAUSAL_LM_CACHE[cache_key] = (tokenizer, model, explicit_device)

    return _LOCAL_CAUSAL_LM_CACHE[cache_key]


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
