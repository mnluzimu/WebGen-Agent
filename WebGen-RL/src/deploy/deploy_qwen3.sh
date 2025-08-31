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

conda activate /mnt/cache/luzimu/code_agent/.env/webgen_vllm

vllm serve /mnt/cache/luzimu/code_agent/outs/Qwen3-8B-Instruct_webgen-agent_train_721-06021141/Qwen3-8B-Instruct_webgen-agent_train_721-06021141_checkpoint-19 \
    --dtype auto \
    --host 0.0.0.0 \
    --port 8000 \
    --pipeline-parallel-size 1 \
    --tensor-parallel-size 4 \
    --cpu-offload-gb 0 \
    # --hf-overrides '{"rope_scaling": {"factor": 4.0, "original_max_position_embeddings": 32768, "rope_type": "yarn"}}'