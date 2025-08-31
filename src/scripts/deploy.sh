vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --dtype auto \
    --host 0.0.0.0 \
    --port 8000 \
    --pipeline-parallel-size 1 \
    --tensor-parallel-size 4 \
    --cpu-offload-gb 0