# test_grpo_advantage.py
import numpy as np
import torch
import pytest

from verl.trainer.ppo.core_algos import compute_step_level_grpo_advantage  # adjust import if needed


# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------
def mask(bs: int, valid_len: int | list[int], seq_len: int) -> torch.Tensor:
    """
    Build a binary response mask of shape (bs, seq_len).

    - If `valid_len` is an int → every sample has the same valid length.
    - If `valid_len` is a list/tuple → per-sample valid lengths.
    """
    if isinstance(valid_len, int):
        valid_len = [valid_len] * bs

    m = torch.zeros(bs, seq_len, dtype=torch.long)
    for i, v in enumerate(valid_len):
        m[i, :v] = 1
    return m


def reference_expected(token_rewards, resp_mask, index, eps=1e-6):
    """
    Straightforward (slow but clear) re-implementation used
    *inside the tests only* to compute the gold output.
    """
    bs = len(token_rewards)

    # --- collect group statistics ------------------------------------------
    groups = {}
    for i in range(bs):
        grp = index[i]
        groups.setdefault(grp, []).extend([step["reward"]
                                           for step in token_rewards[i]])

    grp_mean = {}
    grp_std = {}
    for g, lst in groups.items():
        lst_t = torch.tensor(lst, dtype=torch.float64)
        if len(lst) == 1:
            grp_mean[g] = torch.tensor(0.0)
            grp_std[g] = torch.tensor(1.0)
        else:
            grp_mean[g] = lst_t.mean()
            grp_std[g]  = lst_t.std()

    # --- build expected tensor ---------------------------------------------
    expected = resp_mask.clone().float()          # 1s in valid region, 0s elsewhere
    L = resp_mask.size(1)

    for i in range(bs):
        mean = grp_mean[index[i]]
        std  = grp_std[index[i]] + eps
        pre  = 0
        for step in token_rewards[i]:
            end = L if step["position"] == -1 else step["position"]
            z = (step["reward"] - mean) / std
            expected[i, pre:end] *= z
            pre = end

    return expected


# -------------------------------------------------------------------------
# 1–4  : your original sanity cases  (unchanged)
# 5–7  : new, more complicated scenarios
# -------------------------------------------------------------------------

# 1. single sample → mean=0, std=1
def test_single_batch_no_normalisation():
    tok_rewards = [[{"reward": 2.0, "position": -1}]]
    idx         = np.array([0])
    resp_mask   = mask(1, 5, 10)

    adv, ret = compute_step_level_grpo_advantage(tok_rewards, resp_mask, idx)
    expected = reference_expected(tok_rewards, resp_mask, idx)

    assert torch.allclose(adv, expected)
    assert torch.equal(adv, ret)


# 2. two responses in the same group → z-score normalisation
def test_group_zscore_normalisation():
    tok_rewards = [
        [{"reward": 2.0, "position": -1}],
        [{"reward": 4.0, "position": -1}],
    ]
    idx       = np.array([0, 0])
    resp_mask = mask(2, 4, 6)

    adv, _ = compute_step_level_grpo_advantage(tok_rewards, resp_mask, idx)
    expected = reference_expected(tok_rewards, resp_mask, idx)

    assert torch.allclose(adv, expected, atol=1e-6)


# 3. multi-step rewards with positions
def test_stepwise_application():
    tok_rewards = [[
        {"reward": 1.0, "position": 3},   # tokens 0-2
        {"reward": 2.0, "position": -1},  # tokens 3-5
    ]]
    idx       = np.array([0])
    resp_mask = mask(1, 6, 7)

    adv, _ = compute_step_level_grpo_advantage(tok_rewards, resp_mask, idx)
    expected = reference_expected(tok_rewards, resp_mask, idx)

    assert torch.allclose(adv, expected, atol=1e-6)


# 4. std == 0  → epsilon guard
def test_zero_std_epsilon_guard():
    tok_rewards = [[{"reward": 5.0, "position": -1}],
                   [{"reward": 5.0, "position": -1}]]
    idx       = np.array([0, 0])
    resp_mask = mask(2, 3, 5)

    adv, _ = compute_step_level_grpo_advantage(tok_rewards, resp_mask, idx)
    expected = reference_expected(tok_rewards, resp_mask, idx)

    assert torch.allclose(adv, expected, atol=1e-6)


# -------------------------------------------------------------------------
# 5. → NEW  multi-group, multi-step, variable lengths
# -------------------------------------------------------------------------
def test_multi_group_multi_step_variable_lengths():
    """
    Three samples, two groups, mixed number of steps and different
    valid lengths; ensures normalisation happens per-group and that the
    positional slicing is correct for each sample.
    """
    tok_rewards = [
        [  # sample-0  (group 0)
            {"reward": 1.0, "position": 3},
            {"reward": -1.0, "position": -1},   # len=6
        ],
        [  # sample-1  (group 1)
            {"reward": 3.0, "position": 2},
            {"reward": 5.0, "position": 4},
            {"reward": 7.0, "position": -1},    # len=7
        ],
        [  # sample-2  (group 0)
            {"reward": 2.0, "position": -1},    # len=5
        ],
    ]
    idx          = np.array([0, 1, 0])           # two groups: 0 and 1
    valid_lengths = [6, 6, 5]
    seq_len      = 8                             # pad to 8 for convenience
    resp_mask    = mask(3, valid_lengths, seq_len)

    adv, _ = compute_step_level_grpo_advantage(tok_rewards, resp_mask, idx)
    expected = reference_expected(tok_rewards, resp_mask, idx)

    assert torch.allclose(adv, expected, atol=1e-6)


# -------------------------------------------------------------------------
# 6. mask padding respected (positions extend past valid tokens)
# -------------------------------------------------------------------------
def test_mask_padding_respected():
    """
    The step's `position` can exceed valid length.  The extra tokens should
    stay zero because resp_mask has zeros there.
    """
    tok_rewards = [[{"reward": 10.0, "position": 6}]]   # position > valid_len
    idx         = np.array([0])
    resp_mask   = mask(1, 4, 8)                         # valid tokens = 0..3

    adv, _ = compute_step_level_grpo_advantage(tok_rewards, resp_mask, idx)
    expected = reference_expected(tok_rewards, resp_mask, idx)

    # All padded tokens (4..7) must remain zero
    assert torch.equal(adv[:, 4:], torch.zeros_like(adv[:, 4:]))
    assert torch.allclose(adv, expected, atol=1e-6)


# -------------------------------------------------------------------------
# 7. large batch, randomised rewards  (stress / shape check)
# -------------------------------------------------------------------------
def test_large_random_batch_stress():
    """
    Stress-test with 32 samples, up to 5 steps each, two groups,
    sequence length 32.  Random rewards but reproducible seed.
    """
    torch.manual_seed(0)
    rng = torch.Generator().manual_seed(42)

    bs        = 32
    seq_len   = 32
    max_steps = 5

    tok_rewards = []
    idx = []
    valid_len = []

    for i in range(bs):
        group = 0 if i < bs // 2 else 1
        idx.append(group)

        steps = torch.randint(1, max_steps + 1, (1,), generator=rng).item()
        positions = sorted(torch.randint(1, seq_len - 1, (steps - 1,),
                                         generator=rng).tolist()) + [-1]
        rewards   = torch.randn(steps, generator=rng).tolist()

        tok_rewards.append(
            [{"reward": r, "position": p} for r, p in zip(rewards, positions)]
        )
        valid_len.append(torch.randint(5, seq_len + 1, (1,),
                                       generator=rng).item())

    idx        = np.array(idx)
    resp_mask  = mask(bs, valid_len, seq_len)

    adv, _ = compute_step_level_grpo_advantage(tok_rewards, resp_mask, idx)
    expected = reference_expected(tok_rewards, resp_mask, idx)

    assert adv.shape == (bs, seq_len)
    assert torch.allclose(adv, expected, atol=1e-6)


# -------------------------------------------------------------------------
if __name__ == "__main__":
    pytest.main([__file__])
