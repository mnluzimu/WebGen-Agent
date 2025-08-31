# test_grpo_advantage.py
import numpy as np
import torch
import pytest

from verl.trainer.ppo.core_algos import compute_step_level_grpo_advantage  # adjust import

# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
def mask(bs: int, valid_length: int, length: int) -> torch.Tensor:
    """All-ones mask of shape (bs, length)."""
    result = torch.zeros(bs, length, dtype=torch.long)
    result[:, :valid_length] = 1
    return result

# ---------------------------------------------------------------------
# 1. single sample → mean=0, std=1 (fallback branch)
# ---------------------------------------------------------------------
def test_single_batch_no_normalisation():
    tok_rewards = [[{"reward": 2.0, "position": -1}]]   # one scalar for all tokens
    idx         = np.array([0])                         # only one group
    resp_mask   = mask(1, 5, 10)

    advantages, returns = compute_step_level_grpo_advantage(tok_rewards,
                                                            resp_mask,
                                                            idx)

    expected = mask(1, 5, 10).float()
    expected[0, 0:5] = 2.0
    assert torch.allclose(advantages, expected)
    assert torch.equal(advantages, returns)


# ---------------------------------------------------------------------
# 2. two responses in same group → z-score normalisation
# ---------------------------------------------------------------------
def test_group_zscore_normalisation():
    # response 0: reward 2, response 1: reward 4   → mean=3, std=1
    tok_rewards = [
        [{"reward": 2.0, "position": -1}],
        [{"reward": 4.0, "position": -1}],
    ]
    idx       = np.array([0, 0])
    resp_mask = mask(2, 4, 5)

    adv, _ = compute_step_level_grpo_advantage(tok_rewards, resp_mask, idx)

    expected = resp_mask.float()
    expected[0, 0:4] = (2.0 - torch.mean(torch.tensor([2.0, 4.0]))) / torch.std(torch.tensor([2.0, 4.0]))  # z-score for response 0
    expected[1, 0:4] = (4.0 - torch.mean(torch.tensor([2.0, 4.0]))) / torch.std(torch.tensor([2.0, 4.0]))  # z-score for response 1
    assert torch.allclose(adv, expected, atol=1e-6)


# ---------------------------------------------------------------------
# 3. multi-step rewards with positions
# ---------------------------------------------------------------------
def test_stepwise_application():
    """
    response length = 6
    step 0 applies reward 1 to tokens 0..2
    step 1 applies reward 2 to tokens 3..5
    """
    tok_rewards = [[
        {"reward": 1.0, "position": 3},
        {"reward": 2.0, "position": -1},
    ]]
    idx       = np.array([0])
    resp_mask = mask(1, 6, 7)

    adv, _ = compute_step_level_grpo_advantage(tok_rewards, resp_mask, idx)

    expected = resp_mask.float()
    expected[0, 0:3] = (1.0 - torch.mean(torch.tensor([1.0, 2.0]))) / torch.std(torch.tensor([1.0, 2.0]))  # z-score for step 0
    expected[0, 3:6] = (2.0 - torch.mean(torch.tensor([1.0, 2.0]))) / torch.std(torch.tensor([1.0, 2.0]))   # z-score for step 1
    assert torch.allclose(adv, expected)


# ---------------------------------------------------------------------
# 4. check epsilon path (std≈0 → should not blow up)
# ---------------------------------------------------------------------
def test_zero_std_epsilon_guard():
    tok_rewards = [[{"reward": 5.0, "position": -1}],
                   [{"reward": 5.0, "position": -1}]]   # identical → std=0
    idx       = np.array([0, 0])
    resp_mask = mask(2, 3, 5)

    adv, _ = compute_step_level_grpo_advantage(tok_rewards, resp_mask, idx)

    expected = resp_mask.float()
    expected[0, 0:3] = 0.0  # mean=5, std=0 → z-score is 0
    expected[1, 0:3] = 0.0  # same for second response
    assert torch.allclose(adv, expected, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__])
    # test_single_batch_no_normalisation()
    # test_group_zscore_normalisation()
    # test_stepwise_application()
    # test_zero_std_epsilon_guard()