import os
import socket
import contextlib
import torch
from torch.utils.data import SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
import sglang as sgl

from verl.workers.agentic.llm_sql_agent.sqlact import SQLActAgentGroup
from verl.workers.agentic.llm_sql_agent.generation import GenerationConfig
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from verl.utils import hf_tokenizer, hf_processor
from verl import DataProto

from verl.workers.reward_manager import SQLRewardManager

import hydra
from verl.trainer.ppo import core_algos
import uuid
import numpy as np

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def get_free_port() -> int:
    """Finds a free TCP port on localhost."""
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def init_dist_single():
    """Initialise a *single‑process* distributed group exactly once.

    Child processes created via torch.multiprocessing **import** this module
    but do **not** execute the `main()` function (guarded by
    `if __name__ == "__main__"`). Therefore they *inherit* the env‑vars and
    connect to the store started by rank‑0; they must **not** create a new
    TCPStore.
    """
    if torch.distributed.is_initialized():
        return  # already done in this process

    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")

    # Reserve a port only in the parent process (rank‑0)
    os.environ.setdefault("MASTER_PORT", str(get_free_port()))

    torch.distributed.init_process_group(backend="nccl", init_method="env://")


# -----------------------------------------------------------------------------
# Main training loop encapsulated to avoid re‑execution in spawn children
# -----------------------------------------------------------------------------

@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    init_dist_single()
    world_size = torch.distributed.get_world_size()

    # ---------------------------------------------------------------------
    # Paths & constants
    # ---------------------------------------------------------------------
    MODEL_PATH = "/mnt/cache/sharemath/models/Qwen/Qwen2.5-Coder-7B-Instruct"
    PARQUET_PATH = "/mnt/cache/luzimu/code_agent/data/SkyRL-SQL-653-data/train.parquet"
    DB_PATH = "/mnt/cache/luzimu/code_agent/data/OmniSQL-datasets/data"

    MAX_PROMPT_LEN = 27767
    MAX_RESPONSE_LEN = 5000
    BATCH_SIZE = 4
    NUM_TRAJ = 5

    # Tokeniser / processor ------------------------------------------------
    tokenizer = hf_tokenizer(MODEL_PATH, trust_remote_code=True)
    processor = hf_processor(MODEL_PATH, use_fast=True)

    # Dataset --------------------------------------------------------------
    train_dataset = RLHFDataset(
        parquet_files=PARQUET_PATH,
        tokenizer=tokenizer,
        processor=processor,
        prompt_key="prompt",
        image_key="images",
        max_prompt_length=MAX_PROMPT_LEN,
        filter_prompts=True,
        return_raw_chat=False,
        truncation="error",
        filter_overlong_prompts=False,
        num_workers=1,
    )

    train_loader = StatefulDataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        sampler=SequentialSampler(train_dataset),
        collate_fn=collate_fn,
        num_workers=0,
        drop_last=True,
    )

    # Single SGL engine that all iterations share -------------------------
    engine = sgl.Engine(
        model_path=MODEL_PATH,
        port=get_free_port(),
        dtype="bfloat16",
        max_total_tokens=60 * MAX_PROMPT_LEN,
        max_prefill_tokens=2 * MAX_PROMPT_LEN,
        enable_memory_saver=True,
        mem_fraction_static=0.9,
        tp_size=max(1, world_size // 2),
        log_level="INFO",
    )

    sampling_params = {
        "temperature": 0.6,
        "top_p": 0.95,
        "max_new_tokens": MAX_RESPONSE_LEN,
        "top_k": -1,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "skip_special_tokens": False,
    }

    gen_config = GenerationConfig(
        max_turns=5,
        max_start_length=6144,
        max_prompt_length=MAX_PROMPT_LEN,
        max_response_length=MAX_RESPONSE_LEN,
        max_obs_length=MAX_RESPONSE_LEN,
        num_gpus=max(1, world_size // 2),
        db_path=DB_PATH,
        no_think_rl=False,
    )

    reward_fn = SQLRewardManager(tokenizer=tokenizer,
                                       num_examine=0,
                                       config=config,
                                       compute_score=None)

    i = 0
    log_dir = "/mnt/cache/luzimu/code_agent/WebGen-RL/log/debug_sql"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    for batch_dict in train_loader:
        batch = DataProto.from_single_dict(batch_dict)
        input_keys = ["input_ids", "attention_mask", "position_ids"]
        gen_batch = batch.pop(batch_keys=input_keys, non_tensor_batch_keys=["db_id", "data_source"])
        prompts = gen_batch

        # with open(os.path.join(log_dir, f"{i}.log"), "w", encoding="utf-8") as f:
        #     f.write(f'input_ids:\n{gen_batch.batch["input_ids"]}\n\nattention_mask:\n{gen_batch.batch["attention_mask"]}\n\nposition_ids:\n{gen_batch.batch["position_ids"]}\n\ndata_source:\n{gen_batch.non_tensor_batch["data_source"]}\n\n')
        #     f.write(f'input_ids.size():\n{gen_batch.batch["input_ids"].size()}\n\nattention_mask.size():\n{gen_batch.batch["attention_mask"].size()}\n\nposition_ids.size():\n{gen_batch.batch["position_ids"].size()}\n\n')

        # generate
        print(f'input_ids:\n{gen_batch.batch["input_ids"]}\n\nattention_mask:\n{gen_batch.batch["attention_mask"]}\n\nposition_ids:\n{gen_batch.batch["position_ids"]}\n\ndata_source:\n{gen_batch.non_tensor_batch["data_source"]}\n\n')
        print(f'input_ids.size():\n{gen_batch.batch["input_ids"].size()}\n\nattention_mask.size():\n{gen_batch.batch["attention_mask"].size()}\n\nposition_ids.size():\n{gen_batch.batch["position_ids"].size()}\n\n')

        agent_group = SQLActAgentGroup(
            batch=prompts,
            infer_engine=engine,
            num_trajectories=NUM_TRAJ,
            gen_config=gen_config,
            tokenizer=engine.tokenizer_manager.tokenizer,
            sampling_params=sampling_params,
        )

        def print_batch(batch_batch):
            for k, v in batch_batch.items():
                print(f"{k}.size(): {v.size()}")

        gen_batch_output = agent_group.run()
        print("======== gen_batch_output.batch: ", gen_batch_output.batch)
        print_batch(gen_batch_output.batch)
        print("======== gen_batch_output.non_tensor_batch: ", gen_batch_output.non_tensor_batch)

        batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                             dtype=object)
        batch = batch.repeat(repeat_times=NUM_TRAJ, interleave=True)
        batch = batch.union(gen_batch_output)
        print("======== batch.batch: ", batch.batch)
        print_batch(batch.batch)
        print("======== batch.non_tensor_batch: ", batch.non_tensor_batch)

        # compute reward
        reward_tensor_dict, reward_metrics = reward_fn(batch)
        print("======== reward_tensor_dict: ", reward_tensor_dict)
        print("======== reward_metrics: ", reward_metrics)

        # process
        batch.batch['token_level_scores'] = reward_tensor_dict['all']
        for k, v in reward_tensor_dict.items():
            batch.batch[k] = v
        batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

        def compute_response_mask(data: DataProto):
            responses = data.batch['responses']
            response_length = responses.size(1)
            attention_mask = data.batch['attention_mask']
            return attention_mask[:, -response_length:]

        if "response_mask" not in batch.batch.keys():
            batch.batch['response_mask'] = compute_response_mask(batch)

        print("======== batch.batch['token_level_rewards']: ", batch.batch['token_level_rewards'].tolist())
        print("======== batch.batch['response_mask']: ", batch.batch['response_mask'].tolist())
        print("======== batch.non_tensor_batch['uid']: ", batch.non_tensor_batch['uid'])

        print("======== batch.batch['token_level_rewards'].size(): ", batch.batch['token_level_rewards'].size())
        print("======== batch.batch['response_mask'].size(): ", batch.batch['response_mask'].size())

        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=batch.batch['token_level_rewards'],
            response_mask=batch.batch['response_mask'],
            index=batch.non_tensor_batch['uid'])
        batch.batch['advantages'] = advantages
        batch.batch['returns'] = returns

        print("======== batch.batch['advantages']: ", batch.batch['advantages'])
        print("======== batch.batch['returns']: ", batch.batch['returns'])

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Ensure every child process is spawned (not fork) to avoid CUDA context copy
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
