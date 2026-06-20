from __future__ import annotations

import hashlib
import math
from contextlib import contextmanager
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from .config import CANONICAL_L0, Stage2V3Config
from .data import PromptExample
from .judge import keyword_refusal
from .legacy import task_gate, v2


def layer_from_sae_id(sae_id: str) -> int:
    return int(sae_id.split("/", 1)[0].removeprefix("layer_"))


def _prompt_text(tokenizer: Any, prompt: str, use_chat_template: bool) -> str:
    return v2.maybe_chat_prompt(tokenizer, prompt, use_chat_template)


def _last_nonpadding(attention_mask: torch.Tensor) -> torch.Tensor:
    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device)
    return (attention_mask.long() * positions.unsqueeze(0)).max(dim=1).values


@torch.no_grad()
def collect_prompt_hidden(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    layer: int,
    max_length: int,
    batch_size: int,
    device: str,
    use_chat_template: bool,
) -> torch.Tensor:
    captured: list[torch.Tensor] = []

    def hook(_module: Any, _inputs: Any, output: Any) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        captured.append(hidden.detach())

    handle = model.model.layers[layer].register_forward_hook(hook)
    output_rows = []
    original_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    try:
        for start in range(0, len(prompts), batch_size):
            texts = [
                _prompt_text(tokenizer, prompt, use_chat_template)
                for prompt in prompts[start : start + batch_size]
            ]
            encoded = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            captured.clear()
            model(**encoded, use_cache=False)
            hidden = captured[-1]
            last = _last_nonpadding(encoded.attention_mask)
            output_rows.append(
                hidden[torch.arange(hidden.shape[0], device=device), last].float().cpu()
            )
    finally:
        tokenizer.padding_side = original_side
        handle.remove()
    if not output_rows:
        raise RuntimeError("No prompt-final activations were captured.")
    return torch.cat(output_rows, dim=0)


@torch.no_grad()
def prompt_feature_metrics(sae: Any, hidden: torch.Tensor, device: str) -> dict[str, Any]:
    hidden_device = hidden.to(device)
    features = sae.encode(hidden_device).float()
    reconstruction = sae.decode(features).float()
    error = hidden_device.float() - reconstruction
    centered = hidden_device.float() - hidden_device.float().mean(dim=0, keepdim=True)
    explained_variance = 1.0 - float(error.square().sum() / centered.square().sum().clamp_min(1e-12))
    cosine = F.cosine_similarity(hidden_device.float(), reconstruction, dim=-1)
    firing = features > 0
    return {
        "mean": features.mean(dim=0).cpu(),
        "firing_rate": firing.float().mean(dim=0).cpu(),
        "activity_mass": (features.mean(dim=0) * firing.float().mean(dim=0)).cpu(),
        "pool": features.detach().cpu().to(torch.float16),
        "explained_variance": explained_variance,
        "decoded_cosine": float(cosine.mean()),
        "reconstruction_mse": float(error.square().mean()),
        "l0": float(firing.sum(dim=-1).float().mean()),
        "dead_feature_rate": float((firing.sum(dim=0) == 0).float().mean()),
        "tokens": int(features.shape[0]),
    }


def extract_refusal_direction(harmful: torch.Tensor, benign: torch.Tensor) -> torch.Tensor:
    return F.normalize(harmful.float().mean(dim=0) - benign.float().mean(dim=0), dim=0)


def _prefix_requests(
    tokenizer: Any,
    prompts: list[PromptExample],
    prefixes: tuple[str, ...],
    use_chat_template: bool,
) -> list[Any]:
    requests = []
    for example in prompts:
        prompt = _prompt_text(tokenizer, example.prompt, use_chat_template)
        for index, prefix in enumerate(prefixes):
            requests.append(
                task_gate.ScoreRequest(
                    example_key=example.example_id,
                    choice_index=index,
                    gold=0,
                    prompt=prompt,
                    continuation=task_gate.normalize_continuation(prefix),
                )
            )
    return requests


def _prefix_margin_from_scores(
    scores: torch.Tensor, prompt_count: int, refusal_count: int, compliance_count: int
) -> torch.Tensor:
    grouped = scores.reshape(prompt_count, refusal_count + compliance_count)
    refusal = torch.logsumexp(grouped[:, :refusal_count], dim=1) - math.log(refusal_count)
    compliance = torch.logsumexp(grouped[:, refusal_count:], dim=1) - math.log(compliance_count)
    return refusal - compliance


@contextmanager
def _direction_hook(
    model: Any,
    layer: int,
    direction: torch.Tensor,
    mode: str | None,
    scale: float,
    prompt_end_holder: dict[str, torch.Tensor],
):
    if mode is None:
        yield
        return
    direction = direction.to(next(model.parameters()).device)

    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        ends = prompt_end_holder["ends"]
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        selected = hidden[rows, ends].float()
        if mode == "subtract":
            patched_selected = selected - (selected @ direction).unsqueeze(-1) * direction
        elif mode == "add":
            patched_selected = selected + scale * direction
        else:
            raise ValueError(f"Unknown direction intervention: {mode}")
        patched = hidden.clone()
        patched[rows, ends] = patched_selected.to(hidden.dtype)
        return (patched,) + output[1:] if isinstance(output, tuple) else patched

    handle = model.model.layers[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@torch.no_grad()
def refusal_decision_margins(
    model: Any,
    tokenizer: Any,
    examples: list[PromptExample],
    config: Stage2V3Config,
    device: str,
    layer: int | None = None,
    direction: torch.Tensor | None = None,
    intervention: str | None = None,
    direction_scale: float = 0.0,
) -> list[dict[str, Any]]:
    rows = []
    prefixes = config.refusal_prefixes + config.compliance_prefixes
    holder: dict[str, torch.Tensor] = {}
    context = (
        _direction_hook(model, layer, direction, intervention, direction_scale, holder)
        if layer is not None and direction is not None
        else _direction_hook(model, 0, torch.empty(0), None, 0.0, holder)
    )
    with context:
        for start in range(0, len(examples), config.batch_examples):
            chunk = examples[start : start + config.batch_examples]
            requests = _prefix_requests(tokenizer, chunk, prefixes, config.use_chat_template)
            input_ids, attention_mask, continuation_mask = v2.collate_requests(
                tokenizer, requests, config.max_length, device
            )
            first_continuation = continuation_mask.float().argmax(dim=1)
            holder["ends"] = (first_continuation - 1).clamp_min(0)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            scores = v2.continuation_scores(
                outputs.logits, input_ids, attention_mask, continuation_mask
            )
            margins = _prefix_margin_from_scores(
                scores, len(chunk), len(config.refusal_prefixes), len(config.compliance_prefixes)
            )
            rows.extend(
                {"unit_id": item.example_id, "margin": float(margin.cpu())}
                for item, margin in zip(chunk, margins)
            )
    return rows


def _paired_bootstrap_ci(values: np.ndarray, samples: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    boot = np.empty(samples, dtype=float)
    for index in range(samples):
        boot[index] = float(rng.choice(values, len(values), replace=True).mean())
    low, high = np.quantile(boot, [0.025, 0.975])
    return [float(low), float(high)]


def direction_effect(
    baseline: list[dict[str, Any]], intervention: list[dict[str, Any]], seed: int
) -> dict[str, Any]:
    base = {row["unit_id"]: row["margin"] for row in baseline}
    changed = {row["unit_id"]: row["margin"] for row in intervention}
    keys = sorted(set(base) & set(changed))
    differences = np.asarray([changed[key] - base[key] for key in keys], dtype=float)
    std = float(differences.std(ddof=1)) if len(differences) > 1 else 0.0
    return {
        "n": len(keys),
        "mean_difference": float(differences.mean()),
        "ci95": _paired_bootstrap_ci(differences, 5000, seed),
        "standardized_effect": float(differences.mean() / std) if std > 0 else 0.0,
    }


def run_layer_scan(
    config: Stage2V3Config,
    manifest: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    tokenizer = v2.AutoTokenizer.from_pretrained(config.model_id)
    v2.set_pad_token(tokenizer)
    model = v2.load_model(config.model_id, device)
    harmful_calibration = [PromptExample(**item) for item in manifest["harmful"]["calibration"]]
    benign_calibration = [PromptExample(**item) for item in manifest["benign"]["calibration"]]
    harmful_dev = [PromptExample(**item) for item in manifest["harmful"]["dev"]]
    benign_dev = [PromptExample(**item) for item in manifest["benign"]["dev"]]
    candidates = []
    directions: dict[int, torch.Tensor] = {}
    for sae_id in config.sae_ids:
        layer = layer_from_sae_id(sae_id)
        sae, _metadata = v2.load_sae_compat(config.sae_release, sae_id, device)
        v2.freeze_sae(sae)
        harmful_hidden = collect_prompt_hidden(
            model, tokenizer, [item.prompt for item in harmful_calibration], layer,
            config.max_length, config.batch_examples, device, config.use_chat_template,
        )
        benign_hidden = collect_prompt_hidden(
            model, tokenizer, [item.prompt for item in benign_calibration], layer,
            config.max_length, config.batch_examples, device, config.use_chat_template,
        )
        metrics = prompt_feature_metrics(sae, torch.cat((harmful_hidden, benign_hidden)), device)
        direction = extract_refusal_direction(harmful_hidden, benign_hidden)
        directions[layer] = direction
        harmful_projection = harmful_hidden @ direction
        benign_projection = benign_hidden @ direction
        direction_scale = float(harmful_projection.mean() - benign_projection.mean())
        harmful_base = refusal_decision_margins(model, tokenizer, harmful_dev, config, device)
        harmful_subtract = refusal_decision_margins(
            model, tokenizer, harmful_dev, config, device, layer, direction, "subtract", direction_scale
        )
        benign_base = refusal_decision_margins(model, tokenizer, benign_dev, config, device)
        benign_add = refusal_decision_margins(
            model, tokenizer, benign_dev, config, device, layer, direction, "add", direction_scale
        )
        harmful_effect = direction_effect(harmful_base, harmful_subtract, config.split_seed + layer)
        benign_effect = direction_effect(benign_base, benign_add, config.split_seed + 100 + layer)
        canonical_l0 = CANONICAL_L0[layer]
        l0_ratio = metrics["l0"] / canonical_l0
        compatibility_pass = (
            metrics["decoded_cosine"] >= config.sae_cosine_min
            and metrics["explained_variance"] >= config.sae_explained_variance_min
        )
        mediation_pass = harmful_effect["ci95"][1] < 0 and benign_effect["ci95"][0] > 0
        warnings = []
        if not config.l0_ratio_min <= l0_ratio <= config.l0_ratio_max:
            warnings.append("Observed prompt-final L0 differs from canonical L0 by more than 2x.")
        selection_score = -harmful_effect["standardized_effect"] + benign_effect["standardized_effect"]
        candidates.append(
            {
                "layer": layer,
                "sae_id": sae_id,
                "sae_metadata": {"release": config.sae_release, "sae_id": sae_id},
                "metrics": {key: value for key, value in metrics.items() if not isinstance(value, torch.Tensor)},
                "canonical_l0": canonical_l0,
                "l0_ratio": l0_ratio,
                "direction_scale": direction_scale,
                "harmful_subtraction": harmful_effect,
                "benign_addition": benign_effect,
                "compatibility_pass": compatibility_pass,
                "mediation_pass": mediation_pass,
                "eligible": compatibility_pass and mediation_pass,
                "selection_score": selection_score,
                "warnings": warnings,
            }
        )
        del sae
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    eligible = [item for item in candidates if item["eligible"]]
    selected = max(eligible, key=lambda item: item["selection_score"]) if eligible else None
    smoke_override = False
    if selected is None and config.extra.get("smoke") and candidates:
        selected = max(candidates, key=lambda item: item["selection_score"])
        selected = {**selected, "smoke_override": True, "eligible": False}
        smoke_override = True
    artifact_dir = config.output_root / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    for layer, direction in directions.items():
        torch.save(direction, artifact_dir / f"refusal_direction_layer{layer}.pt")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "status": "PASS" if selected else "INCONCLUSIVE_R0",
        "smoke_override": smoke_override,
        "selected": selected,
        "candidates": candidates,
        "conclusion": (
            f"Layer {selected['layer']} was selected by smoke override for plumbing only; this is not R0 evidence."
            if smoke_override
            else f"Layer {selected['layer']} passed matched-SAE transfer and directional mediation gates."
            if selected
            else "No candidate layer passed both matched-SAE transfer and directional mediation gates."
        ),
    }


def prompt_feature_reference(
    model: Any,
    tokenizer: Any,
    sae: Any,
    examples: list[PromptExample],
    layer: int,
    config: Stage2V3Config,
    device: str,
) -> dict[str, Any]:
    hidden = collect_prompt_hidden(
        model, tokenizer, [item.prompt for item in examples], layer,
        config.max_length, config.batch_examples, device, config.use_chat_template,
    )
    return prompt_feature_metrics(sae, hidden, device)


def refusal_attribution_scores(
    model: Any,
    tokenizer: Any,
    sae: Any,
    examples: list[PromptExample],
    reference: dict[str, Any],
    layer: int,
    config: Stage2V3Config,
    device: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    prefixes = config.refusal_prefixes + config.compliance_prefixes
    total = None
    units = 0
    holder: dict[str, Any] = {}

    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        ends = holder["ends"]
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        selected = hidden[rows, ends].detach().requires_grad_(True)
        features = sae.encode(selected)
        features.retain_grad()
        reconstruction = sae.decode(features)
        error_node = (selected - reconstruction).detach()
        patched = hidden.clone()
        patched[rows, ends] = (reconstruction + error_node).to(hidden.dtype)
        holder["features"] = features
        return (patched,) + output[1:] if isinstance(output, tuple) else patched

    handle = model.model.layers[layer].register_forward_hook(hook)
    try:
        for start in range(0, len(examples), config.batch_examples):
            chunk = examples[start : start + config.batch_examples]
            requests = _prefix_requests(tokenizer, chunk, prefixes, config.use_chat_template)
            input_ids, attention_mask, continuation_mask = v2.collate_requests(
                tokenizer, requests, config.max_length, device
            )
            holder["ends"] = (continuation_mask.float().argmax(dim=1) - 1).clamp_min(0)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            scores = v2.continuation_scores(
                outputs.logits, input_ids, attention_mask, continuation_mask
            )
            margins = _prefix_margin_from_scores(
                scores, len(chunk), len(config.refusal_prefixes), len(config.compliance_prefixes)
            )
            loss = F.softplus(-margins).mean()
            loss.backward()
            features = holder["features"]
            baseline = reference["mean"].to(device, features.dtype).unsqueeze(0)
            effect = (features.grad.detach() * (baseline - features.detach())).sum(dim=0)
            effect = effect * len(chunk)
            total = effect if total is None else total + effect
            units += len(chunk)
            model.zero_grad(set_to_none=True)
    finally:
        handle.remove()
    if total is None:
        raise RuntimeError("No refusal attribution effects were produced.")
    return (total / max(1, units)).float().cpu(), {
        "examples": units,
        "objective": "prompt-final refusal-prefix margin",
        "continuation_features_patched": False,
        "ablation": "mean",
    }


def _refusal_objective_loss(
    model: Any,
    tokenizer: Any,
    sae: Any,
    examples: list[PromptExample],
    reference: dict[str, Any],
    layer: int,
    config: Stage2V3Config,
    device: str,
    feature_id: int | None = None,
    mode: str = "mean",
    seed: int = 0,
) -> float:
    prefixes = config.refusal_prefixes + config.compliance_prefixes
    holder: dict[str, Any] = {}
    generator = torch.Generator(device=device).manual_seed(seed)

    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        if feature_id is None:
            return output
        ends = holder["ends"]
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        selected = hidden[rows, ends]
        features = sae.encode(selected)
        reconstruction = sae.decode(features)
        error_node = (selected - reconstruction).detach()
        patched_features = features.clone()
        if mode == "mean":
            patched_features[:, feature_id] = reference["mean"][feature_id].to(
                device, features.dtype
            )
        elif mode == "zero":
            patched_features[:, feature_id] = 0
        elif mode == "resample":
            pool = reference["pool"][:, feature_id].to(device, features.dtype)
            indices = torch.randint(
                pool.shape[0], (features.shape[0],), generator=generator, device=device
            )
            patched_features[:, feature_id] = pool[indices]
        else:
            raise ValueError(f"Unknown refusal ablation mode: {mode}")
        patched = hidden.clone()
        patched[rows, ends] = (sae.decode(patched_features) + error_node).to(hidden.dtype)
        return (patched,) + output[1:] if isinstance(output, tuple) else patched

    handle = model.model.layers[layer].register_forward_hook(hook) if feature_id is not None else None
    total = 0.0
    units = 0
    try:
        with torch.no_grad():
            for start in range(0, len(examples), config.batch_examples):
                chunk = examples[start : start + config.batch_examples]
                requests = _prefix_requests(tokenizer, chunk, prefixes, config.use_chat_template)
                input_ids, attention_mask, continuation_mask = v2.collate_requests(
                    tokenizer, requests, config.max_length, device
                )
                holder["ends"] = (continuation_mask.float().argmax(dim=1) - 1).clamp_min(0)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                scores = v2.continuation_scores(
                    outputs.logits, input_ids, attention_mask, continuation_mask
                )
                margins = _prefix_margin_from_scores(
                    scores, len(chunk), len(config.refusal_prefixes), len(config.compliance_prefixes)
                )
                loss = F.softplus(-margins).mean()
                total += float(loss.cpu()) * len(chunk)
                units += len(chunk)
    finally:
        if handle is not None:
            handle.remove()
    return total / max(1, units)


def validate_refusal_attribution(
    model: Any,
    tokenizer: Any,
    sae: Any,
    examples: list[PromptExample],
    causal_scores: torch.Tensor,
    reference: dict[str, Any],
    layer: int,
    config: Stage2V3Config,
    device: str,
    seed: int,
) -> dict[str, Any]:
    ranked = torch.argsort(causal_scores)
    count = config.validation_features_per_tail
    selected = torch.unique(torch.cat((ranked[:count], ranked[-count:]))).tolist()
    clean = _refusal_objective_loss(
        model, tokenizer, sae, examples, reference, layer, config, device
    )
    rows = []
    for mode in ("mean", "resample", "zero"):
        for feature_id in selected:
            ablated = _refusal_objective_loss(
                model,
                tokenizer,
                sae,
                examples,
                reference,
                layer,
                config,
                device,
                int(feature_id),
                mode,
                seed + int(feature_id),
            )
            rows.append(
                {
                    "feature_id": int(feature_id),
                    "mode": mode,
                    "predicted_delta": float(causal_scores[feature_id]),
                    "measured_delta": ablated - clean,
                }
            )
    by_mode = {}
    for mode in ("mean", "resample", "zero"):
        subset = [row for row in rows if row["mode"] == mode]
        predicted = np.asarray([row["predicted_delta"] for row in subset])
        measured = np.asarray([row["measured_delta"] for row in subset])
        correlation = spearmanr(predicted, measured)
        by_mode[mode] = {
            "n": len(subset),
            "rho": float(correlation.statistic),
            "p_value": float(correlation.pvalue),
            "sign_agreement": float(np.mean(np.sign(predicted) == np.sign(measured))),
        }
    return {"clean_loss": clean, "rows": rows, "by_mode": by_mode}


@torch.no_grad()
def generate_prompt_rows(
    model: Any,
    tokenizer: Any,
    examples: list[PromptExample],
    config: Stage2V3Config,
    device: str,
) -> list[dict[str, Any]]:
    output = []
    pad_id = v2.set_pad_token(tokenizer)
    original_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        for start in range(0, len(examples), config.generation_batch_size):
            chunk = examples[start : start + config.generation_batch_size]
            prompts = [
                _prompt_text(tokenizer, item.prompt, config.use_chat_template) for item in chunk
            ]
            encoded = tokenizer(
                prompts, padding=True, truncation=True, max_length=config.max_length,
                return_tensors="pt",
            ).to(device)
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=config.max_new_tokens,
                pad_token_id=pad_id,
                use_cache=True,
            )
            prompt_width = encoded.input_ids.shape[1]
            texts = tokenizer.batch_decode(generated[:, prompt_width:], skip_special_tokens=True)
            for example, text in zip(chunk, texts):
                output.append(
                    {
                        "unit_id": example.example_id,
                        "target": example.label,
                        "prompt": example.prompt,
                        "response": text,
                        "response_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                        "keyword_refusal": keyword_refusal(text),
                        "generated_tokens": len(
                            tokenizer(text, add_special_tokens=False).input_ids
                        ),
                    }
                )
    finally:
        tokenizer.padding_side = original_side
    return output
