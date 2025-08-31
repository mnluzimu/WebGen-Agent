import json
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Set
import os
import time


_EXCLUDED_DIRS: Set[str] = {"node_modules", "dist", ".next"}
_EXCLUDED_FILES: Set[str] = {"package-lock.json"}


def remove_dir(directory):
    for _ in range(5):
        try:
            shutil.rmtree(directory)
            return True
        except:
            time.sleep(5)
    return False


def _is_excluded(path: Path, root: Path) -> bool:
    """
    Return True if *path* should be skipped according to the exclusion rules.

    • Any file *named* in _EXCLUDED_FILES is skipped.
    • Any file or directory that lies under a directory named in _EXCLUDED_DIRS
      is skipped (e.g. `project/node_modules/**`, `project/dist/**`).
    """
    # Skip explicit filenames
    if path.name in _EXCLUDED_FILES:
        return True

    # Skip anything that resides inside an excluded directory
    for part in path.relative_to(root).parts:
        if part in _EXCLUDED_DIRS:
            return True
    return False


def directory_to_dict(root_dir: str) -> Dict[str, str]:
    """
    Traverse *root_dir* recursively, **excluding**:
        – package-lock.json (anywhere in the tree)
        – everything inside any `node_modules` or `dist` directory

    Returns a mapping {relative_posix_path: file_content}.
    """
    root = Path(root_dir).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"{root} is not a directory")

    file_map: Dict[str, str] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if _is_excluded(path, root):
            continue

        rel_path = path.relative_to(root).as_posix()
        file_map[rel_path] = path.read_text(encoding="utf-8", errors="replace")
    return file_map


def dict_to_directory(
    file_map: Dict[str, str],
    target_dir: str,
    *,
    overwrite: bool = True
) -> None:
    """
    Recreate a directory tree at *target_dir* from *file_map*.
    If *target_dir* exists and *overwrite* is True, it is deleted first.

    Note: the exclusion rules are irrelevant here because `file_map`
    is expected to have been created by `directory_to_dict`, so it already
    lacks the excluded paths.
    """
    target = Path(target_dir).expanduser().resolve()

    if target.exists():
        if overwrite:
            remove_dir(target)
        else:
            raise FileExistsError(
                f"{target} already exists. "
                "Set overwrite=True to replace it."
            )
    target.mkdir(parents=True, exist_ok=True)

    for rel_path, content in file_map.items():
        file_path = target / rel_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
#  3. Restore workspace from the latest stepN.json log
# ---------------------------------------------------------------------------
_STEP_RE = re.compile(r"step(\d+)\.json$", re.IGNORECASE)

def _extract_step_index(self, filename: str) -> int:
        m = re.search(r"step(\d+)\.json$", filename)
        return int(m.group(1)) if m else -1

def restore_from_last_step(
    log_dir: str, workspace_dir: str, max_teps: int = 20
) -> Tuple[List[dict], str]:
    """
    Locate the highest‑numbered *stepN.json* in *log_dir*, rebuild *workspace_dir*
    using its "files" snapshot, and return its "messages" list and
    "gui_instruction" string.

    Returns
    -------
    messages : list[dict]
    gui_instruction : str
    """
    log_path = Path(log_dir).expanduser().resolve()
    if not log_path.is_dir():
        return None, None, None, None, None, None

    # Gather and sort the step files by numeric suffix
    step_files = sorted(
        (
            p
            for p in log_path.iterdir()
            if p.is_file() and _STEP_RE.match(p.name) and int(_STEP_RE.match(p.name).group(1)) < max_teps
        ),
        key=lambda p: int(_STEP_RE.match(p.name).group(1)),  # type: ignore[arg-type]
    )

    if not step_files:
        return None, None, None, None, None, None

    nodes = {}
    for file in step_files:
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
            nodes[os.path.basename(file)] = {
                "screenshot_grade": data["screenshot_grade"], 
                "webvoyager_grade": data["webvoyager_grade"],
                "pre": data["pre"],
                "has_error": data.get("has_error", False)
            }
        except:
            step_files.remove(file)

    latest = step_files[-1]
    print(f"[{os.path.basename(log_dir)}] Found latest step log: {latest}")
    step_idx = int(_STEP_RE.match(latest.name).group(1))
    with latest.open(encoding="utf-8") as f:
        data = json.load(f)

    # Recreate workspace
    if os.path.exists(workspace_dir):
        remove_dir(workspace_dir)
    os.makedirs(workspace_dir, exist_ok=True)
    # Rebuild the workspace directory from the files snapshot
    dict_to_directory(data["files"], workspace_dir, overwrite=True)

    # Return metadata
    return data.get("messages", []), data.get("gui_instruction", ""), step_idx, data.get("screenshot_grade", 0), data.get("webvoyager_grade", 0), nodes


if __name__ == "__main__":
    # root_dir = "/mnt/cache/agent/Zimu/WebGen-Agent/workspaces_root/WebGen-Bench_deepseek-v3-250324_iter20_screenshot-suggestions/000057"
    # files_json = directory_to_dict(root_dir)
    # print(files_json)
    # import json
    # with open("/mnt/cache/agent/Zimu/WebGen-Agent/service_logs/debug/files.json", "w", encoding="utf-8") as f:
    #     json.dump(files_json, f, ensure_ascii=False)

    # import json
    # with open("/mnt/cache/agent/Zimu/WebGen-Agent/service_logs/debug/files.json", "r", encoding="utf-8") as f:
    #     files_json = json.load(f)

    # dict_to_directory(files_json, "/mnt/cache/agent/Zimu/WebGen-Agent/service_logs/debug/000057_reconstruct")

    log_dir = "/mnt/cache/agent/Zimu/WebGen-Agent/service_logs/WebGenAgentV1_WebGen-Bench_deepseek-v3-250324_iter20_select_best/000010"
    workspace_dir = "/mnt/cache/agent/Zimu/WebGen-Agent/workspaces_root/debug_agentv1_2/000010"
    messages, gui_instruction, step_idx, screenshot_grade, webvoyager_grade, nodes = restore_from_last_step(log_dir, workspace_dir)
    print(messages)
    print(gui_instruction)
    print(step_idx)
    print(screenshot_grade)
    print(webvoyager_grade)
    print(nodes)