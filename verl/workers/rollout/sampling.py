"""Sampling parameters shared by the agent-loop execution backends."""

from __future__ import annotations

from typing import Any


def build_agent_loop_sampling_params(config, *, validate: bool) -> dict[str, Any]:
    """Build request parameters without discarding rollout-level settings."""
    params = dict(
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        repetition_penalty=config.repetition_penalty,
        logprobs=config.calculate_log_probs,
    )
    if validate:
        params["top_p"] = config.val_kwargs.top_p
        params["top_k"] = config.val_kwargs.top_k
        params["temperature"] = config.val_kwargs.temperature
    return params


__all__ = ["build_agent_loop_sampling_params"]
