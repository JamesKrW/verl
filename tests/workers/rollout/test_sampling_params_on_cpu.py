from types import SimpleNamespace

from verl.workers.rollout.sampling import build_agent_loop_sampling_params


def test_validation_sampling_preserves_rollout_repetition_penalty():
    config = SimpleNamespace(
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        repetition_penalty=1.1,
        calculate_log_probs=True,
        val_kwargs=SimpleNamespace(temperature=0.8, top_p=0.6, top_k=2),
    )

    assert build_agent_loop_sampling_params(config, validate=True) == {
        "temperature": 0.8,
        "top_p": 0.6,
        "top_k": 2,
        "repetition_penalty": 1.1,
        "logprobs": True,
    }
