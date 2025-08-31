################################
# new env installation
################################
conda create -p env/webgen-rl python=3.10
conda activate env/webgen-rl

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

git clone https://github.com/NVIDIA/apex.git
cd /mnt/cache/luzimu/apex
MAX_JOB=32 pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./

bash /mnt/cache/luzimu/clash/start_clash.sh
cd /mnt/cache/luzimu/code_agent/WebGen-RL
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh

# cd /mnt/cache/luzimu/verl
pip install --no-deps -e .
# pip install -r /mnt/cache/luzimu/code_agent/WebGen-RL/requirements_webgen-rl.txt

bash scripts/install_vllm_sglang_mcore.sh

pip install func_timeout

pip install tensorboard

pip install selenium