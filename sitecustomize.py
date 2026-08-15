"""Opt-in interpreter-start compatibility hooks for spawned rollout workers.

vLLM V1 starts its engine with Python's ``spawn`` method.  A patch applied only in the
Ray/server parent is therefore not inherited by the engine process.  ``sitecustomize``
is imported by Python in every spawned interpreter; keep all hooks explicitly gated so
ordinary verl and non-vLLM runs pay no import or behavioural cost.
"""

import os


if os.getenv("VERL_PATCH_VLLM_GPTJ_MROPE", "0") == "1":
    from verl.utils.vllm.gptj_mrope_patch import apply_gptj_mrope_compat_patch

    apply_gptj_mrope_compat_patch()
