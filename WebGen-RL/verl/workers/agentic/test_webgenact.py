import os
import socket
import contextlib
import torch
from torch.utils.data import SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
import sglang as sgl

from verl.workers.agentic.webgen_agent.webgenact import WebGenActAgentGroup
from verl.workers.agentic.webgen_agent.generation import GenerationConfig
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from verl.utils import hf_tokenizer, hf_processor
from verl import DataProto

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

def main() -> None:
    init_dist_single()
    world_size = torch.distributed.get_world_size()

    # ---------------------------------------------------------------------
    # Paths & constants
    # ---------------------------------------------------------------------
    # MODEL_PATH = "/mnt/cache/sharemath/models/Qwen/Qwen2.5-Coder-7B-Instruct"
    MODEL_PATH = "/mnt/cache/luzimu/code_agent/outs/Qwen2_5-Coder-7B-Instruct_webgen-agent_train_721-06020859/Qwen2_5-Coder-7B-Instruct_webgen-agent_train_721-06020859_checkpoint-18"
    # PARQUET_PATH = "/mnt/cache/luzimu/code_agent/data/SkyRL-SQL-653-data/train.parquet"
    PARQUET_PATH = "/mnt/cache/luzimu/code_agent/data/WebGen-Instruct_1_408/train.parquet"
    DB_PATH = "/mnt/cache/luzimu/code_agent/data/OmniSQL-datasets/data"

    MAX_PROMPT_LEN = 26767
    MAX_RESPONSE_LEN = 6000
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
        vlm_model="/mnt/cache/sharemath/models/Qwen/Qwen2.5-VL-32B-Instruct",
    )

    i = 0
    log_dir = "/mnt/cache/luzimu/code_agent/WebGen-RL/log/debug0"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    for batch_dict in train_loader:
        batch = DataProto.from_single_dict(batch_dict)
        input_keys = ["input_ids", "attention_mask", "position_ids"]
        gen_batch = batch.pop(batch_keys=input_keys, non_tensor_batch_keys=["data_source"])
        prompts = gen_batch

        with open(os.path.join(log_dir, f"{i}.log"), "w", encoding="utf-8") as f:
            f.write(f'input_ids:\n{gen_batch.batch["input_ids"]}\n\nattention_mask:\n{gen_batch.batch["attention_mask"]}\n\nposition_ids:\n{gen_batch.batch["position_ids"]}\n\ndata_source:\n{gen_batch.non_tensor_batch["data_source"]}\n\n')

        print(f'input_ids:\n{gen_batch.batch["input_ids"]}\n\nattention_mask:\n{gen_batch.batch["attention_mask"]}\n\nposition_ids:\n{gen_batch.batch["position_ids"]}\n\ndata_source:\n{gen_batch.non_tensor_batch["data_source"]}\n\n')
        agent_group = WebGenActAgentGroup(
            batch=prompts,
            infer_engine=engine,
            num_trajectories=NUM_TRAJ,
            gen_config=gen_config,
            tokenizer=engine.tokenizer_manager.tokenizer,
            sampling_params=sampling_params,
        )

        results = agent_group.run()
        print(results)

    

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Ensure every child process is spawned (not fork) to avoid CUDA context copy
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
