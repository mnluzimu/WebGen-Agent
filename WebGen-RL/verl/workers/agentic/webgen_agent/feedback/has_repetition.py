import re
from collections import Counter


# --------------------------------------------------------------------------- #
# 1.  Canonicaliser – removes “local” differences so similar snippets match   #
# --------------------------------------------------------------------------- #
_COLOR_HEX  = re.compile(r'#(?:[0-9a-fA-F]{3}){1,2}\b')
_COLOR_RGB  = re.compile(r'rgba?\([^)]*\)')
_NUMBER     = re.compile(r'[-+]?\d*\.?\d+(?:e[-+]?\d+)?')

def _canonicalise(text: str) -> str:
    """Replace numbers / colour codes with <VAR>, collapse whitespace, lower-case."""
    text = _COLOR_HEX.sub('<VAR>', text)
    text = _COLOR_RGB.sub('<VAR>', text)
    text = _NUMBER.sub('<VAR>', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text


def has_repetition(text: str, *, chunk_len: int = 50, dup_ratio: float = 0.40) -> bool:
    """
    Return True if `text` looks repetitive.

    Parameters
    ----------
    text : str
        The string to check.
    chunk_len : int, default 80
        Length (in characters) of the sliding window / chunk.
    dup_ratio : float, default 0.30
        Minimum fraction of chunks that must be duplicates to call it repetitive.

    Notes
    -----
    * Works well for “copy-pasted blocks” (CSS, JSON, log lines, etc.).
    * Insensitive to whitespace differences.
    """
    text = _canonicalise(text)
    text = text.replace("<|endoftext|>", "")
    # text = _canonicalise(text)
    if len(text) < chunk_len * 2:        # too short to repeat meaningfully
        return False

    # --- 1. basic normalisation -------------------------------------------
    # remove leading/trailing spaces per line and collapse multiple blanks
    norm = re.sub(r'\s+', ' ', '\n'.join(line.strip() for line in text.splitlines()))
    if len(norm) < chunk_len * 2:
        return False

    # --- 2. slice into sliding chunks -------------------------------------
    chunks = [norm[i:i + chunk_len] for i in range(0, len(norm) - chunk_len + 1, 1)]
    counts = Counter(chunks)
    dup_chunks = sum(c for c in counts.values() if c >= 3)
    print(f"Total chunks: {len(chunks)}, Duplicate chunks: {dup_chunks}")
    total_chunks = len(chunks)

    return dup_chunks / total_chunks >= dup_ratio


# --------------------------------------------------------------------------
if __name__ == "__main__":
    import json
    in_file = "/mnt/cache/luzimu/code_agent/WebGen-RL/log/debug_rollout_2025-06-24T02-53-58-122/execute_pred_0_2025-06-24T03-19-26-043.json"
    with open(in_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    responses_str = data["responses_str"]
    repetition = []
    no_repetition = []
    for i, response_str in enumerate(responses_str):
        response_str = response_str.replace("<|endoftext|>", "")
        if has_repetition(response_str):
            repetition.append(response_str)
        else:
            no_repetition.append(response_str)
    print(f"Total responses: {len(responses_str)}")
    with open("/mnt/cache/luzimu/code_agent/WebGen-RL/verl/workers/agentic/webgen_agent/feedback/repetition_result.json", "w", encoding="utf-8") as f:
        json.dump({
            "repetition": repetition,
            "no_repetition": no_repetition
        }, f, ensure_ascii=False, indent=4)

