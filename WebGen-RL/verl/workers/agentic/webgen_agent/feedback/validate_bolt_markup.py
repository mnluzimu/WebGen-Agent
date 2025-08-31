import re
from typing import Tuple, Optional

_TAG_RE = re.compile(r"<\s*(/?)\s*(boltArtifact|boltAction)\b[^>]*?>")

def validate_bolt_markup(text: str) -> Tuple[bool, Optional[str]]:
    """
    Return (True, None) if `text` contains well-nested <boltArtifact> / <boltAction>
    blocks.  Otherwise return (False, explanation).

    Only the tag boundaries are checked; attributes and inner content are left
    untouched, so ordinary HTML/JSX inside a boltAction will not confuse the parser.
    """
    stack: list[str] = []

    for match in _TAG_RE.finditer(text):
        closing, tag = match.groups()
        pos = match.start()

        if closing:                              # e.g. </boltAction>
            if not stack:
                return False, f"Unmatched closing </{tag}> at byte {pos}"
            if stack[-1] != tag:
                return False, (
                    f"Mismatched closing </{tag}> at byte {pos}; "
                    f"expected </{stack[-1]}>"
                )
            stack.pop()
        else:                                    # e.g. <boltAction ...>
            stack.append(tag)

    if stack:                                    # something was never closed
        return False, f"Missing closing tag for <{stack[-1]}>"

    return True, None


if __name__ == "__main__":
    # Example usage
    # import json
    # in_file = "/mnt/cache/luzimu/code_agent/WebGen-RL/log/debug_rollout_2025-06-24T02-53-58-122/execute_pred_0_2025-06-24T03-19-26-043.json"
    # with open(in_file, "r", encoding="utf-8") as f:
    #     data = json.load(f)
    # responses_str = data["responses_str"]
    # results = []
    # for i, response_str in enumerate(responses_str):
    #     response_str = response_str.replace("<|endoftext|>", "")
    #     is_valid, explanation = validate_bolt_markup(response_str)
    #     results.append({
    #         "index": i,
    #         "response": response_str,
    #         "is_valid": is_valid,
    #         "explanation": explanation
    #     })
    # print(f"Total responses: {len(responses_str)}")
    # with open("/mnt/cache/luzimu/code_agent/WebGen-RL/verl/workers/agentic/webgen_agent/feedback/validate_result.json", "w", encoding="utf-8") as f:
    #     json.dump({
    #         "results": results,
    #     }, f, ensure_ascii=False, indent=4)

    s = "<boltAction type=\"screenshot_validated\"/>"
    is_valid, explanation = validate_bolt_markup(s)
    print(is_valid, explanation)