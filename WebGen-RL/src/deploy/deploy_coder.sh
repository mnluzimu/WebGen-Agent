__conda_setup="$('/usr/local/lib/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/usr/local/lib/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/usr/local/lib/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/usr/local/lib/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup

conda activate /mnt/cache/luzimu/DeepSeek-R1/.env/vllm

# vllm serve /mnt/cache/luzimu/code_agent/outs/Qwen2_5-Coder-32B-Instruct_app-bench_train_debug-04211545 \
# vllm serve /mnt/cache/luzimu/code_agent/outs/Qwen2_5-Coder-32B-Instruct_app-bench_train_debug-04211545/Qwen2_5-Coder-32B-Instruct_app-bench_train_debug-04211545_checkpoint-42 \
# vllm serve /mnt/cache/luzimu/code_agent/outs/Qwen2_5-Coder-32B-Instruct_app-bench_train_batch13_filtered-04241109/Qwen2_5-Coder-32B-Instruct_app-bench_train_batch13_filtered-04241109_checkpoint-90 \
# vllm serve /mnt/cache/luzimu/code_agent/outs/Qwen2_5-Coder-32B-Instruct_app-bench_train_batch13_filtered_decontaminated-04241810/Qwen2_5-Coder-32B-Instruct_app-bench_train_batch13_filtered_decontaminated-04241810_checkpoint-60 \
# vllm serve /mnt/cache/luzimu/code_agent/outs/Qwen2_5-Coder-32B-Instruct_app-bench_train_batch13_filtered_decontaminated_new-04251527/Qwen2_5-Coder-32B-Instruct_app-bench_train_batch13_filtered_decontaminated_new-04251527_checkpoint-54 \
# vllm serve /mnt/cache/luzimu/code_agent/outs/Qwen2_5-Coder-32B-Instruct_app-bench_train_batch13_filtered_decontaminated_new-04251527/Qwen2_5-Coder-32B-Instruct_app-bench_train_batch13_filtered_decontaminated_new-04251527_checkpoint-36 \
# vllm serve /mnt/cache/luzimu/code_agent/outs/Qwen2_5-Coder-32B-Instruct_app-bench_train_batch1234_filtered_decontaminated_new-04271632/Qwen2_5-Coder-32B-Instruct_app-bench_train_batch1234_filtered_decontaminated_new-04271632_checkpoint-50 \
# vllm serve /mnt/cache/luzimu/code_agent/outs/Qwen2_5-Coder-32B-Instruct_app-bench_train_batch1_filtered_decontaminated_new_half-04281202/Qwen2_5-Coder-32B-Instruct_app-bench_train_batch1_filtered_decontaminated_new_half-04281202_checkpoint-10 \
# vllm serve /mnt/cache/luzimu/code_agent/outs/Qwen2_5-Coder-14B-Instruct_app-bench_train_batch13_filtered_decontaminated_new-04281235/Qwen2_5-Coder-14B-Instruct_app-bench_train_batch13_filtered_decontaminated_new-04281235_checkpoint-35 \
# vllm serve /mnt/cache/luzimu/code_agent/outs/Qwen2_5-Coder-7B-Instruct_app-bench_train_batch13_filtered_decontaminated_new-04281036/Qwen2_5-Coder-7B-Instruct_app-bench_train_batch13_filtered_decontaminated_new-04281036_checkpoint-35 \
# vllm serve /mnt/cache/luzimu/code_agent/outs/Qwen2_5-Coder-32B-Instruct_app-bench_train_batch1_filtered_decontaminated_new_half-04281202/Qwen2_5-Coder-32B-Instruct_app-bench_train_batch1_filtered_decontaminated_new_half-04281202_checkpoint-15 \
# vllm serve /mnt/cache/luzimu/code_agent/outs/Qwen2_5-Coder-32B-Instruct_app-bench_train_batch1_filtered_decontaminated_new-04251449/Qwen2_5-Coder-32B-Instruct_app-bench_train_batch1_filtered_decontaminated_new-04251449_checkpoint-18 \
# vllm serve /mnt/cache/sharemath/models/qwen/Qwen2.5-Coder-32B-Instruct \
# vllm serve /mnt/cache/luzimu/code_agent/outs/Qwen2_5-Coder-7B-Instruct_webgen-agent_train_721-06020859/Qwen2_5-Coder-7B-Instruct_webgen-agent_train_721-06020859_checkpoint-18 \
# vllm serve /mnt/cache/luzimu/code_agent/outs/Qwen2_5-Coder-7B-Instruct_webgen-agent_train_new_711-06071005/Qwen2_5-Coder-7B-Instruct_webgen-agent_train_new_711-06071005_checkpoint-16 \
# vllm serve /mnt/cache/luzimu/code_agent/outs/Qwen2_5-Coder-32B-Instruct_webgen-agent_train_new_711_28672-06091416/Qwen2_5-Coder-32B-Instruct_webgen-agent_train_new_711_28672-06091416_checkpoint-9 \
vllm serve /mnt/cache/sharemath/models/Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --dtype auto \
    --host 0.0.0.0 \
    --port 28457 \
    --pipeline-parallel-size 1 \
    --tensor-parallel-size 4 \
    --cpu-offload-gb 0 \
    # --hf-overrides '{"rope_scaling": {"factor": 4.0, "original_max_position_embeddings": 32768, "rope_type": "yarn"}}'