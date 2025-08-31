from verl.workers.agentic.webgen_agent.webgenact import WebGenActAgentGroup
from verl.workers.agentic.webgen_agent.generation import GenerationConfig
from torchdata.stateful_dataloader import StatefulDataLoader
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
import torch
from verl.utils import hf_tokenizer, hf_processor
from torch.utils.data import SequentialSampler
import sglang as sgl
from verl import DataProto
import os

os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "30003"
if ""
torch.distributed.init_process_group()
total_world_size = torch.distributed.get_world_size()

local_path = "/mnt/cache/sharemath/models/Qwen/Qwen2.5-Coder-7B-Instruct"
tokenizer = hf_tokenizer(local_path, trust_remote_code=True)
processor = hf_processor(local_path, use_fast=True)

train_dataset = RLHFDataset(
    parquet_files="/mnt/cache/luzimu/code_agent/data/SkyRL-SQL-653-data/train.parquet",
    tokenizer=tokenizer,
    processor=processor,
    prompt_key="prompt",
    image_key='images',
    max_prompt_length=29048,
    filter_prompts=True,
    return_raw_chat=False,
    truncation='error',
    filter_overlong_prompts=False,
    num_workers=1,
)

sampler = SequentialSampler(data_source=train_dataset)
train_dataloader = StatefulDataLoader(
    dataset=train_dataset,
    batch_size=256,
    num_workers=0,
    drop_last=True,
    collate_fn=collate_fn,
    sampler=sampler
)

for batch_dict in train_dataloader:
    batch: DataProto = DataProto.from_single_dict(batch_dict)
    batch_keys = ['input_ids', 'attention_mask', 'position_ids']
    gen_batch = batch.pop(batch_keys=batch_keys, non_tensor_batch_keys=["db_id", "data_source"])

    engine = sgl.Engine(
        model_path="/mnt/cache/sharemath/models/Qwen/Qwen2.5-Coder-7B-Instruct",
        port=40000,
        dtype="bfloat16",
        max_total_tokens=60*32048,
        max_prefill_tokens=2*32048,
        enable_memory_saver=True,
        mem_fraction_static=0.9,
        tp_size=4,
        log_level="INFO",
    )

    sampling_params = {
        "temperature": 0.6,
        "top_p": 0.95,
        "max_new_tokens": 3000,
        "top_k": -1,
        "min_p": 0,
        "repetition_penalty": 1,
        "skip_special_tokens": False,
    }

    gen_config = GenerationConfig(
        max_turns=5,
        max_start_length=6144,
        max_prompt_length=29048,
        max_response_length=3000,
        max_obs_length=3000,
        num_gpus= total_world_size // 2,
        db_path="/mnt/cache/luzimu/code_agent/data/OmniSQL-datasets/data",
        no_think_rl=False,
    )
    agent_group = WebGenActAgentGroup(
        batch=prompts,
        infer_engine=engine,
        num_trajectories=5,
        gen_config=gen_config,
        tokenizer=engine.tokenizer_manager.tokenizer, 
        sampling_params=sampling_params,
    )
    results = agent_group.run()

    print(results)