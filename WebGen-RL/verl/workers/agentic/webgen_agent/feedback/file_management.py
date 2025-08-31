import os
import re
import html
from pathlib import Path
import re
from typing import List, Dict
from .has_repetition import has_repetition
from .validate_bolt_markup import validate_bolt_markup
from .has_syntax_error import has_syntax_error


def prompt_to_messages(prompt: str) -> List[Dict[str, str]]:
    """
    Convert the *string* produced by tokenizer.apply_chat_template(..., tokenize=False)
    back to a list of {"role": ..., "content": ...} messages.

    Works out-of-the-box for:
      • <|im_start|> / <|im_end|>   (OpenAI / Mistral / Zephyr / Gemma / Qwen-chat-ML)
      • [INST] ... [/INST]          (Llama-2 / Code-Llama / Alpaca / Vicuna)

    Extend the `TEMPLATES` list if your model uses another wrapper.
    """
    TEMPLATES = [
        {
            "name": "openai",
            "pattern": re.compile(
                r"<\|im_start\|>(\w+)\n(.*?)<\|im_end\|>",
                flags=re.DOTALL,
            ),
        },
    ]

    for tpl in TEMPLATES:
        matches = tpl["pattern"].findall(prompt)
        if matches:
            messages = []
            for role, content in matches:
                role = role or "user"         # Llama-2 often omits role inside [INST]
                messages.append({"role": role.strip(), "content": content.strip()})
            return messages

    raise ValueError("Unrecognised chat-template markup – add a parser rule.")


vite_file_content = """import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 0, // Will use random available port
    strictPort: false,
    watch: {
      usePolling: true,
      interval: 1000
    },
    hmr: {
      port: 0 // Random port for HMR
    }
  },
  preview: {
    port: 0 // Random port for preview
  }
})"""


def extract_and_write_files(response: str, workspace_dir: str):
    """
    Parses a string with <boltAction type="file" filePath="...">...</boltAction> blocks
    and writes the files to the corresponding paths under workspace_dir.
    """
    os.makedirs(workspace_dir, exist_ok=True)
    
    # Find all boltAction blocks
    pattern = r'<boltAction type="file" filePath="(.*?)">(.*?)</boltAction>'
    matches = re.findall(pattern, response, flags=re.DOTALL)
    
    is_syntax_error = False
    for file_path, file_content in matches:
        # Decode HTML entities (e.g., &lt; becomes <)
        decoded_content = html.unescape(file_content)
        
        # Create full file path
        full_path = os.path.join(workspace_dir, file_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        # Write the file content
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(decoded_content)

        has_error, explanation = has_syntax_error(full_path, decoded_content)
        # if has_error:
        #     print(f"Syntax error in file {full_path}: {explanation}")
        #     is_syntax_error = True
        print(f"Created: {full_path}")

    # Create full file path
    full_path = os.path.join(workspace_dir, "vite.config.js")
    if not os.path.isfile(full_path):
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        # Write the file content
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(vite_file_content)
        print(f"Created: {full_path}")
    
    return is_syntax_error


def create_workspace(response: str, workspace_dir: str):
    messages = prompt_to_messages(response)
    for message in messages:
        if message["role"] == "assistant":
            is_syntax_error = extract_and_write_files(message["content"], workspace_dir)

    is_webvoyager, is_finished, is_good_format, is_repetition = False, False, True, False
    if is_syntax_error:
        is_good_format = False
        print(f"Syntax error detected in workspace: {workspace_dir}")

    if messages[-1]["role"] == "assistant" and ('boltAction type="screenshot_validated"' in messages[-1]["content"] or "boltAction type='screenshot_validated'" in messages[-1]["content"]):
        if (len(messages[-1]["content"]) > 400) or ("boltArtifact" in messages[-1]["content"]):
            is_good_format = False
        is_webvoyager = True
    elif messages[-1]["role"] == "assistant" and ('boltAction type="finish"' in messages[-1]["content"] or "boltAction type='finish'" in messages[-1]["content"]):
        if (len(messages[-1]["content"]) > 400) or ("boltArtifact" in messages[-1]["content"]):
            is_good_format = False
        is_finished = True

    if messages[-1]["role"] == "assistant" and has_repetition(messages[-1]["content"]):
        is_good_format = False
        is_repetition = True

    return is_webvoyager, is_finished, is_good_format, is_repetition


if __name__ == "__main__":
    import json
    in_file = "/mnt/cache/luzimu/code_agent/WebGen-RL/log/debug_rollout_2025-06-24T02-53-58-122/execute_pred_0_2025-06-24T03-19-26-043.json"
    with open(in_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    responses_str = data["responses_str"]
    inputs_str = data["inputs_str"]
    results = []

    for response_str, input_str in zip(responses_str, inputs_str):
        response_str = response_str.replace("<|endoftext|>", "")
        input_str = input_str.replace("<|endoftext|>", "")
        workspace_dir = f"/mnt/cache/luzimu/code_agent/WebGen-RL/workspaces_root/debug/000001"
        
        is_webvoyager, is_finished, is_good_format = create_workspace(input_str + response_str, workspace_dir)
        results.append({
            "response": response_str,
            "input": input_str,
            "is_webvoyager": is_webvoyager,
            "is_finished": is_finished,
            "is_good_format": is_good_format
        })
    print(f"Total responses: {len(responses_str)}")
    with open("/mnt/cache/luzimu/code_agent/WebGen-RL/verl/workers/agentic/webgen_agent/feedback/create_result.json", "w", encoding="utf-8") as f:
        json.dump({
            "results": results,
        }, f, ensure_ascii=False, indent=4)
