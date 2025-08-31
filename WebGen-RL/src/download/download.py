import os
import traceback

from tqdm import tqdm
from huggingface_hub import snapshot_download, login

CACHE_DIR='.cache'

login(token="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")

def download_model(huggingface_path, local_path):
    while True:
        try:
            snapshot_download(
                huggingface_path, 
                local_dir=local_path, 
                cache_dir=CACHE_DIR, 
                local_dir_use_symlinks=True
            )
            break
        except KeyboardInterrupt:
            break
        except:
            traceback.print_exc()
            continue


def download_data(huggingface_path, local_path, allow_patterns=None):
    while True:
        try:
            snapshot_download(
                huggingface_path, 
                local_dir=local_path, 
                cache_dir=CACHE_DIR, 
                repo_type='dataset',
                local_dir_use_symlinks=True,
                allow_patterns=allow_patterns
            )
            break
        except KeyboardInterrupt:
            break
        except:
            traceback.print_exc()
            continue
        

def download_metrics(huggingface_path, local_path):
    while True:
        try:
            snapshot_download(
                huggingface_path, 
                local_dir=local_path, 
                cache_dir=CACHE_DIR, 
                local_dir_use_symlinks=True
            )
            break
        except KeyboardInterrupt:
            break
        except:
            traceback.print_exc()
            continue


def transfer_soft_link(src, dst):
    os.makedirs(dst, exist_ok=True)

    files = os.listdir(src)

    for file in tqdm(files):
        if os.path.islink(os.path.join(src, file)):
            blob = os.path.join(src, os.readlink(os.path.join(src, file)))
        else:
            blob = os.path.join(src, file)

        print(blob, os.path.join(dst, file))
        os.system('MOVE "%s" "%s"' % (blob, os.path.join(dst, file)))


def main():
    download_model("Qwen/Qwen3-8B", "models/Qwen3-8B")
    download_model("Qwen/Qwen2.5-Coder-7B-Instruct", "models/Qwen2.5-Coder-7B-Instruct")

if __name__ == '__main__':
    main()