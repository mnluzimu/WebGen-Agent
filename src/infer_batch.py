#!/usr/bin/env python
# coding: utf-8
"""
Parallel WebGenAgent runner — **unordered / incremental save, picklable-safe**.

Key features
------------
1. **Restart by id** – already finished samples (present in output.jsonl)
   are skipped automatically.
2. **Unordered writes** – results are appended as soon as each worker finishes.
3. **Pickling-safe workers** – every worker returns a *plain dict*;
   uncaught exceptions are converted to text so the parent process never
   has to un-pickle third-party exception objects (e.g. `openai.error.APIStatusError`).
4. **Progress bar** – reflects real-time completion.

Example
-------
python infer_batch_v3.py \
       --model Qwen2.5-VL-7B \
       --vlm_model Qwen2.5-VL-7B \
       --data-path data/test.jsonl \
       --workspace-root workspaces \
       --log-root logs \
       --max-iter 20 \
       --num-workers 8
"""
import os
import sys
import json
import argparse
import traceback
from pathlib import Path
from functools import partial
from typing import List, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

# -----------------------------------------------------------------------------#
#  project import path
# -----------------------------------------------------------------------------#
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)
from agent import WebGenAgent  # noqa: E402  (import after sys.path.insert)

# -----------------------------------------------------------------------------#
#  helpers
# -----------------------------------------------------------------------------#
def load_jsonl(in_file: str) -> List[Dict]:
    """Safely read a JSONL file into a list (empty list if file missing)."""
    if not os.path.isfile(in_file):
        return []
    data = []
    with open(in_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_jsonl(records: List[Dict], out_file: str, mode: str = "a") -> None:
    """Append / write records (list of dicts) to JSONL."""
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, mode, encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

# -----------------------------------------------------------------------------#
#  argument parser
# -----------------------------------------------------------------------------#
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch a WebGenAgent experiment (unordered save, "
                    "pickling-safe workers)."
    )
    parser.add_argument("--model", required=True, help="Model name or path.")
    parser.add_argument("--vlm_model", required=True, help="VLM model name or path.")
    parser.add_argument("--fb_model", required=True, help="Feedback model name or path.")
    parser.add_argument("--data-path", required=True,
                        help="Path to the JSONL file containing the data.")
    parser.add_argument("--workspace-root", required=True, type=Path,
                        help="Directory where the agent creates / modifies files.")
    parser.add_argument("--eval-tag", default="", type=str,
                        help="Optional tag to append to the run name.")
    parser.add_argument("--log-root", required=True, type=Path,
                        help="Directory where run logs will be written.")
    parser.add_argument("--max-iter", type=int, default=20, metavar="N",
                        help="Maximum reasoning / action iterations (default: 20).")
    parser.add_argument("--num-workers", type=int, default=os.cpu_count(),
                        help="Parallel worker processes "
                             "(default: number of CPU cores).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing workspace/log directory "
                             "if it already exists.")
    parser.add_argument("--error-limit", type=int, default=5, metavar="N",
                        help="Max number of continuous errors before backtracking.")
    parser.add_argument("--max-tokens", type=int, default=-1, metavar="N")
    parser.add_argument("--max-completion-tokens", type=int, default=-1, metavar="N")
    parser.add_argument("--temperature", type=float, default=0.5, metavar="N")
    return parser

# -----------------------------------------------------------------------------#
#  worker job – always return a picklable payload
# -----------------------------------------------------------------------------#
def process(args_namespace: argparse.Namespace, sample: dict) -> dict:
    """
    Run WebGenAgent on a single sample.
    *Never* propagates a non-picklable exception: any error is converted
    to a plain dict (`ok: False`, `error`, `trace`).
    """
    args = vars(args_namespace)          # Namespace → plain dict
    payload = {"id": sample["id"]}       # Always include the sample id

    try:
        # ---------- per-sample directories ----------
        workspace_dir = os.path.join(args["workspace_root"], sample["id"])
        log_dir = os.path.join(args["log_root"], sample["id"])
        os.makedirs(workspace_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        # ---------- run the agent ----------
        agent = WebGenAgent(
            model=args["model"],
            vlm_model=args["vlm_model"],
            fb_model=args["fb_model"],
            workspace_dir=workspace_dir,
            log_dir=log_dir,
            instruction=sample["instruction"],
            max_iter=args["max_iter"],
            overwrite=args["overwrite"],
            error_limit=args["error_limit"],
            max_tokens=args["max_tokens"],
            max_completion_tokens=args["max_completion_tokens"],
            temperature=args["temperature"],
        )
        result = agent.run()             # whatever the agent normally returns

        payload["ok"] = True
        payload["result"] = result

    except Exception as exc:
        # Convert ANY exception to a simple, picklable payload
        payload.update({
            "ok": False,
            "error": str(exc),
            "trace": traceback.format_exc()
        })

    return payload

# -----------------------------------------------------------------------------#
#  main
# -----------------------------------------------------------------------------#
def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Expand workspace / log roots to include run-specific suffix
    run_tag = "WebGenAgent_{}_{}_iter{}".format(
        Path(args.data_path).stem,
        Path(args.model).name.replace(":", "_"),
        args.max_iter,
    )
    if args.eval_tag:
        run_tag += f"_{args.eval_tag}"

    args.workspace_root = os.path.abspath(os.path.join(args.workspace_root, run_tag))
    args.log_root       = os.path.abspath(os.path.join(args.log_root, run_tag))
    output_file = os.path.join(args.log_root, "output.jsonl")

    # ------------------------------------------------------------------#
    #  restart logic — skip ids already processed
    # ------------------------------------------------------------------#
    completed_records = load_jsonl(output_file)
    completed_ids = {rec.get("id") for rec in completed_records}

    all_samples = load_jsonl(args.data_path)
    remaining_samples = [s for s in all_samples if s.get("id") not in completed_ids]

    if not remaining_samples:
        print("All samples already processed. Nothing to do.")
        return

    print(f"Loaded {len(all_samples)} total samples; "
          f"{len(completed_ids)} already done; "
          f"{len(remaining_samples)} remaining.")

    # Partial so that executor only receives (sample) and not the whole args repeatedly
    worker = partial(process, args)

    # ------------------------------------------------------------------#
    #  parallel execution with unordered, incremental save
    # ------------------------------------------------------------------#
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        future_to_sample = {
            executor.submit(worker, sample): sample for sample in remaining_samples
        }

        with tqdm(total=len(remaining_samples), desc="Process Samples") as pbar:
            for future in as_completed(future_to_sample):
                payload = future.result()     # always picklable

                if payload.get("ok"):
                    # Success – persist only the agent’s result
                    record = payload["result"]
                    record["id"] = payload["id"]
                    save_jsonl([record], output_file, mode="a")
                else:
                    # Failure – print warning; optionally persist the error payload
                    sys.stderr.write(
                        f"[ERROR] Sample id={payload['id']} failed: "
                        f"{payload['error']}\n"
                        f"{payload['trace']}\n"
                    )

                    raise Exception(f"[ERROR] Sample id={payload['id']} failed: {payload['error']}")

                pbar.update(1)


if __name__ == "__main__":
    main()
