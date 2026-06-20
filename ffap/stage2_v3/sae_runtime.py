from __future__ import annotations

from typing import Any


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if callable(value):
        name = getattr(value, "__qualname__", getattr(value, "__name__", type(value).__name__))
        return f"<callable:{name}>"
    return str(value)


def cfg_value(sae: Any, name: str, default: Any = None) -> Any:
    cfg = getattr(sae, "cfg", None)
    return _json_safe(getattr(cfg, name, default))


def _call_norm_in(sae: Any, value: Any) -> tuple[Any, tuple[Any, ...]]:
    fn = getattr(sae, "run_time_activation_norm_fn_in", None)
    if not callable(fn):
        return value, ()
    result = fn(value)
    if isinstance(result, tuple):
        return result[0], tuple(result[1:])
    return result, ()


def _call_norm_out(sae: Any, value: Any, raw_input: Any, normed_input: Any, extra: tuple[Any, ...]) -> Any:
    fn = getattr(sae, "run_time_activation_norm_fn_out", None)
    if not callable(fn):
        return value
    candidate_args = (
        (value, raw_input, *extra),
        (value, normed_input, *extra),
        (value, raw_input),
        (value, normed_input),
        (value, *extra),
        (value,),
    )
    last_error = None
    for args in candidate_args:
        try:
            return fn(*args)
        except TypeError as error:
            last_error = error
    raise RuntimeError(f"Could not call SAE runtime output normalization: {last_error}")


def sae_runtime_summary(sae: Any) -> dict[str, Any]:
    return {
        "normalize_activations": cfg_value(sae, "normalize_activations"),
        "apply_b_dec_to_input": cfg_value(sae, "apply_b_dec_to_input"),
        "hook_name": cfg_value(sae, "hook_name"),
        "hook_layer": cfg_value(sae, "hook_layer"),
        "architecture": cfg_value(sae, "architecture"),
        "d_in": cfg_value(sae, "d_in"),
        "d_sae": cfg_value(sae, "d_sae"),
        "has_runtime_norm_in": callable(getattr(sae, "run_time_activation_norm_fn_in", None)),
        "has_runtime_norm_out": callable(getattr(sae, "run_time_activation_norm_fn_out", None)),
    }


def ensure_sae_runtime_normalization(sae: Any) -> dict[str, Any]:
    """Wrap SAE encode/decode so direct calls use SAE Lens runtime normalization.

    Stage 2 v3 reuses v2 helpers that call ``sae.encode`` and ``sae.decode``
    directly. Gemma Scope SAEs may keep runtime activation scaling outside those
    direct methods, so the wrapper stores the most recent normalized input
    context from ``encode`` and applies the matching output de-normalization in
    ``decode``. The wrapper is intentionally v3-local and idempotent.
    """
    summary = sae_runtime_summary(sae)
    if getattr(sae, "_ffap_runtime_norm_wrapped", False):
        return {**summary, "wrapped": True, "already_wrapped": True}
    if not summary["has_runtime_norm_in"] and not summary["has_runtime_norm_out"]:
        return {**summary, "wrapped": False, "reason": "no_runtime_norm_methods"}

    raw_encode = sae.encode
    raw_decode = sae.decode

    def encode_with_runtime_norm(value: Any) -> Any:
        normed, extra = _call_norm_in(sae, value)
        sae._ffap_runtime_norm_context = (value, normed, extra)
        return raw_encode(normed)

    def decode_with_runtime_norm(features: Any) -> Any:
        decoded = raw_decode(features)
        context = getattr(sae, "_ffap_runtime_norm_context", None)
        if context is None:
            return decoded
        raw_input, normed_input, extra = context
        return _call_norm_out(sae, decoded, raw_input, normed_input, extra)

    sae._ffap_raw_encode = raw_encode
    sae._ffap_raw_decode = raw_decode
    sae.encode = encode_with_runtime_norm
    sae.decode = decode_with_runtime_norm
    sae._ffap_runtime_norm_wrapped = True
    return {**summary, "wrapped": True, "already_wrapped": False}
