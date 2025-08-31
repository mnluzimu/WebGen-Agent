from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Iterable, Sequence

from tqdm import tqdm
import subprocess
import os
import sys
import time
import re
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import json
import tempfile
import shutil
import random

from .timestamp import current_timestamp
from .get_webvoyager_feedback import get_webvoyager_feedback
from .get_screenshot_description_new import get_screenshot_description, get_screenshot_grade
from .file_management import create_workspace, prompt_to_messages
from .generate_gui_agent_instruction import generate_gui_agent_instruction

import signal, platform, subprocess, time

import re
from numbers import Number
from typing import Union

def extract_number(value: object) -> Union[int, float]:
    """
    Return a numeric value according to the following rules:

    1. If *value* is already a number (int, float, numpy number, Decimal,…), return it unchanged.
    2. If *value* is a string, look for the first decimal or scientific-notation number
       inside the string and return it (as int if it has no decimal point, otherwise float).
    3. If no number is found, or *value* is neither a number nor a string, return 0.

    Examples
    --------
    >>> extract_number(42)
    42
    >>> extract_number("height: 1.78 m")
    1.78
    >>> extract_number("error code: -12")
    -12
    >>> extract_number("no digits here")
    0
    >>> extract_number(None)
    0
    """
    # 1) Already numeric?
    if isinstance(value, Number) and not isinstance(value, bool):  # bool is a subclass of int
        return value

    # 2) Extract from string
    if isinstance(value, str):
        # Regex matches integers, floats, and scientific notation, with optional sign
        pattern = re.compile(r'[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?')
        match = pattern.search(value)
        if match:
            num_str = match.group(0)
            # Decide int vs float
            return int(num_str) if re.fullmatch(r'[-+]?\d+', num_str) else float(num_str)

    # 3) Fallback
    return 0


def stop_process_tree(proc: subprocess.Popen, timeout: float = 10.0):
    """
    Gracefully stop the background service that was started with
    `start_new_session=True`.  Works on both POSIX and Windows.
    """
    if proc.poll() is not None:          # already dead
        return

    try:
        if platform.system() == "Windows":
            # Windows: ask for CTRL‑BREAK (graceful) then fall back to Terminate
            proc.send_signal(signal.CTRL_BREAK_EVENT)
            proc.wait(timeout)           # give it time to shut down
            if proc.poll() is None:
                proc.terminate()         # hard kill
        else:
            # POSIX: signal the whole process‑group
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout)
            if proc.poll() is None:      # still alive? use SIGKILL
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception as e:
        print(f"[WARN] could not stop process cleanly: {e}")


def load_json(in_file):
    with open(in_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def run_commands(cmds, cwd):
    """
    Runs a list of commands in the given directory.
    Captures and returns a list of (command, output) tuples.
    """
    results = []
    for cmd in cmds:
        print(f"Running: {cmd}")
        process = subprocess.run(
            cmd,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True
        )
        output = process.stdout + (process.stderr or "")
        print(output)  # Optional: print to console
        results.append((cmd, output))
        if process.returncode != 0:
            print(f"Command failed: {cmd}")
            # sys.exit(process.returncode)
    return results


def start_background_service(start_cmd, cwd, log_file="service.log"):
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "w")

    print(f"Starting service: {start_cmd}")
    process = subprocess.Popen(
        start_cmd,
        shell=True,
        cwd=cwd,
        stdout=log,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True
    )
    
    print(f"Service started with PID: {process.pid}. Logs: {log_path}")
    return process, log_path


def wait_for_url_in_log(log_path, timeout=30):
    print("Waiting for URL to appear in log...")
    url_pattern = re.compile(
        r"http://(?:localhost|(?:\d{1,3}\.){3}\d{1,3}):\d+/?"
    )
    deadline = time.time() + timeout
    url = None

    while time.time() < deadline:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            match = url_pattern.search(content)
            if match:
                url = match.group(0)
                print(f"Found service URL: {url}")
                return url
        time.sleep(1)

    raise TimeoutError("Timed out waiting for service URL in log.")


def take_screenshot(url, output_path="screenshot.png"):
    print(f"Taking screenshot of: {url}")
    options = Options()
    tmp_profile = tempfile.mkdtemp()
    options.add_argument(f"--user-data-dir={tmp_profile}")
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1024,768")

    driver = webdriver.Chrome(options=options)
    driver.get(url)
    time.sleep(3)  # Allow JS to render
    output_path = output_path.replace(".png", f"_{current_timestamp()}.png")
    driver.save_screenshot(output_path)
    driver.quit()
    print(f"Screenshot saved to: {output_path}")
    return output_path


def execute_for_feedback(project_dir, log_dir, cmds=["npm install"], start_cmd="npm run dev", step_idx=None):
    feedback = {
        "install_results": [],
        "install_error": [],
        "start_results": "",
        "start_error": False,
        "screenshot_path": "",
        "screenshot_error": ""
    }
    log_file = os.path.join(log_dir, "service.log")
    package_json = os.path.join(project_dir, "package.json")
    if os.path.isfile(package_json):
        try:
            data = load_json(package_json)
            if "scripts" in data.keys() and "dev" not in data["scripts"].keys():
                if "start" in data["scripts"].keys():
                    start_cmd = "npm run start"
                elif "serve" in data["scripts"].keys():
                    start_cmd = "npm run serve"
                    if "build" in data["scripts"].keys():
                        cmds.append("npm run build")
        except:
            start_cmd = "npm run dev"

    # install dependencies
    results = run_commands(cmds, cwd=project_dir)
    feedback["install_results"] = results
    for cmd, output in results:
        if "error" in output.lower():
            feedback["install_error"].append(cmd)

    # start service
    process, log_path = start_background_service(start_cmd, cwd=project_dir, log_file=log_file)

    # get screenshot
    try:
        url = wait_for_url_in_log(log_path)
        if step_idx is not None:
            img_path = os.path.join(log_dir, f"screenshot_step{step_idx}.png")
        else:
            img_path = os.path.join(log_dir, f"screenshot.png")
        img_path = take_screenshot(url, output_path=img_path)
        feedback["screenshot_url"] = url
        if os.path.isfile(img_path):
            feedback["screenshot_path"] = img_path
    except Exception as e:
        feedback["screenshot_error"] = f"Error: {e}"
        print(f"Error: {e}")
    finally:
        stop_process_tree(process)

    # get start output
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        start_output = f.read()
        start_output = start_output.replace("\u0000", "")
        lines = start_output.split("\n")
        suffix = ""
        if len(lines) > 500:
            lines = lines[:50] + ["......\n[Truncated]\n......"] + lines[-50:]
        new_lines = []
        port_in_use_num = 0
        i = 0
        while i < len(lines):
            if "is in use, trying another one..." in lines[i]:
                if port_in_use_num > 3:
                    i += 1
                    continue
                else:
                    port_in_use_num += 1
                    
            new_lines.append(lines[i])
            i += 1
        lines = new_lines
        start_output = "\n".join(lines)
        if len(start_output) > 10000:
            start_output = start_output[:5000] + "\n......\n[Truncated]\n......\n" + start_output[-5000:]
            suffix = "\n\n...... [Output Too Long, Truncated]"
        start_output = start_output.strip() + suffix
        feedback["start_results"] = start_output
    if "error" in feedback["start_results"].lower():
        feedback["start_error"] = True

    with open(os.path.join(log_dir, "service.pid"), "w") as f:
        f.write(str(process.pid))

    return feedback


def execute_for_webvoyager_feedback(instruction, project_dir, log_dir, vlm_model, model, cmds=["npm install"], start_cmd="npm run dev", step_idx=None):
    feedback = {
        "install_results": [],
        "install_error": [],
        "start_results": "",
        "start_error": False,
        "webvoyager_feedback": "",
        "webvoyager_text": "",
        "webvoyager_error": "",
    }
    log_file = os.path.join(log_dir, "service.log")
    package_json = os.path.join(project_dir, "package.json")
    if os.path.isfile(package_json):
        try:
            data = load_json(package_json)
            if "scripts" in data.keys() and "dev" not in data["scripts"].keys():
                if "start" in data["scripts"].keys():
                    start_cmd = "npm run start"
                elif "serve" in data["scripts"].keys():
                    start_cmd = "npm run serve"
                    if "build" in data["scripts"].keys():
                        cmds.append("npm run build")
        except:
            start_cmd = "npm run dev"

    # install dependencies
    results = run_commands(cmds, cwd=project_dir)
    feedback["install_results"] = results
    for cmd, output in results:
        if "error" in output.lower():
            feedback["install_error"].append(cmd)

    # start service
    process, log_path = start_background_service(start_cmd, cwd=project_dir, log_file=log_file)

    # get screenshot
    try:
        url = wait_for_url_in_log(log_path)
        if step_idx is not None:
            idx = f"step{step_idx}"
        else:
            idx = current_timestamp()
        feedback["webvoyager_text"], feedback["webvoyager_feedback"] = get_webvoyager_feedback(
            idx=idx, 
            output_dir=log_dir, 
            instruction=instruction, 
            url=url, 
            vlm_model=vlm_model,
            model=model
        )
    except Exception as e:
        feedback["webvoyager_error"] = f"Error: {e}"
        print(f"Error: {e}")
    finally:
        stop_process_tree(process)

    # get start output
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        start_output = f.read()
        start_output = start_output.replace("\u0000", "")
        lines = start_output.split("\n")
        suffix = ""
        if len(lines) > 500:
            lines = lines[:50] + ["......\n[Truncated]\n......"] + lines[-50:]
        new_lines = []
        port_in_use_num = 0
        i = 0
        while i < len(lines):
            if "is in use, trying another one..." in lines[i]:
                if port_in_use_num > 3:
                    i += 1
                    continue
                else:
                    port_in_use_num += 1
                    
            new_lines.append(lines[i])
            i += 1
        lines = new_lines
        start_output = "\n".join(lines)
        if len(start_output) > 10000:
            start_output = start_output[:5000] + "\n......\n[Truncated]\n......\n" + start_output[-5000:]
            suffix = "\n\n...... [Output Too Long, Truncated]"
        start_output = start_output.strip() + suffix
        feedback["start_results"] = start_output
    if "error" in feedback["start_results"].lower():
        feedback["start_error"] = True

    with open(os.path.join(log_dir, "service.pid"), "w") as f:
        f.write(str(process.pid))

    if os.path.isfile(log_path):
        os.remove(log_path)

    return feedback


reminders_prompt = """CRITICAL: Always provide the FULL, updated content of a file when editing. This means, if you need to create or modify a file:

      - Include ALL code in the file, even if parts are unchanged
      - NEVER use placeholders like "// rest of the code remains the same..." or "<- leave original code here ->"
      - ALWAYS show the complete, up-to-date file content when updating the file
      - Avoid any form of truncation or summarization"""


def remove_dir(directory):
    for _ in range(5):
        try:
            shutil.rmtree(directory)
            return True
        except:
            time.sleep(5)
    return False


def get_feedback_single(data):
    time.sleep(random.random() * 2)
    response, gui_instruction, vlm_model, instruction = data
    model = vlm_model
    workspace_dir = tempfile.mkdtemp(prefix="workspace_")
    log_dir = tempfile.mkdtemp(prefix="log_")
    is_webvoyager, is_finished, is_good_format, is_repetition = create_workspace(response, workspace_dir)

    if is_finished:
        return {
            "info": None, 
            "feedback_str": None, 
            "has_error": False, 
            "is_finished": True,
            "is_webvoyager": is_webvoyager,
            "webvoyager_grade": 0, 
            "screenshot_grade": 0,
            "is_good_format": is_good_format,
            "is_repetition": is_repetition
        }
    step_idx = 0

    webvoyager_grade, screenshot_grade = 0, 0

    cmds = ["npm install"]
    start_cmd = "npm run dev"
    if is_webvoyager:
        feedback = execute_for_webvoyager_feedback(
            gui_instruction, 
            workspace_dir, 
            log_dir, 
            vlm_model,
            model,
            cmds, 
            start_cmd,
            step_idx
        )
    else:
        feedback = execute_for_feedback(workspace_dir, log_dir, cmds, start_cmd, step_idx)
    output = []
    suffix = []
    error_stages = []

    has_error = False
    # install error
    install_error = feedback["install_error"]
    if len(install_error) > 0:
        has_error = True
        error_stages.append("dependency installation")
        output.append("**Installation of dependencies emitted errors:**\n\n" + "\n\n".join([f"> {cmd}\n{o}" for cmd, o in feedback["install_results"]]))
        if len(install_error) == 1:
            suffix.append(f"Execution of command `{install_error[0]}` has failed.")
        else:
            suffix.append("Execution of commands " + ", ".join([f"`{e}`" for e in install_error][:-1]) + f" and `{install_error[-1]}` have failed.")
    else:
        output.append("Installation of dependencies was successful.")

    # start error
    if feedback["start_error"]:
        has_error = True
        error_stages.append("service starting")
        output.append(f"**Starting the service emitted errors:**\n\n{feedback['start_results']}")
        suffix.append("The service emitted errors when it was being started.")
    else:
        output.append("Starting the service was successful.")
    
    if is_webvoyager:
        if len(feedback["webvoyager_error"]) > 0:
            webvoyager_grade = 0
            output.append(f"There was an error when navigating the website with the GUI agent:\n\n{feedback['webvoyager_error']}")
            has_error = True
            error_stages.append("GUI agent navigation")
        else:
            webvoyager_feedback = feedback['webvoyager_feedback']
            webvoyager_grade = webvoyager_feedback.get("grade", 0)
            webvoyager_grade = extract_number(webvoyager_grade)
            if "improvement_suggestions" in webvoyager_feedback.keys():
                if len(webvoyager_feedback["improvement_suggestions"]) > 0:
                    output.append(f"**The suggestions based on the GUI-agent testing result:**\n\n{webvoyager_feedback['improvement_suggestions']}")
                else:
                    output.append(f"The GUI agent testing is successful and no further improvement is necessary.")
            else:
                output.append(f"Failed to get GUI agent feedback or GUI agent error messages.")
                has_error = True
                error_stages.append("GUI agent trajectory collection")
    else:
        screenshot_description = None
        screenshot_grade_json, screenshot_grade = None, 0
        if len(feedback["screenshot_error"]) > 0:
            output.append(f"There was an error when getting the screenshot of the started website:\n\n{feedback['screenshot_error']}")
            has_error = True
            error_stages.append("screenshot collection")
        elif os.path.isfile(feedback["screenshot_path"]):
            screenshot_description = get_screenshot_description(feedback["screenshot_path"], vlm_model)
            screenshot_grade_json, screenshot_grade = get_screenshot_grade(feedback["screenshot_path"], vlm_model, instruction)
            screenshot_grade = extract_number(screenshot_grade)
            if screenshot_grade <= 1:
                has_error = True
            screenshot_description_success = False
            if "error_message" in screenshot_description.keys() and screenshot_description['error_message'] is not None and screenshot_description['error_message'] != "":
                output.append(f"**The screenshot contains errors:**\n\n{screenshot_description['error_message']}")
                has_error = True
                error_stages.append("screenshot")
                screenshot_description_success = True
            if "screenshot_description" in screenshot_description.keys() and screenshot_description['screenshot_description'] is not None and screenshot_description['screenshot_description'] != "":
                output.append(f"**The screenshot description:**\n\n{screenshot_description['screenshot_description']}")
                screenshot_description_success = True
            if "suggestions" in screenshot_description.keys() and screenshot_description['suggestions'] is not None and screenshot_description['suggestions'] != "":
                output.append(f"**Suggestions for Improvement:**\n\n{screenshot_description['suggestions']}")
                screenshot_description_success = True
            if not screenshot_description_success:
                output.append(f"Failed to get screenshot description or screenshot error messages.")
                has_error = True
                error_stages.append("screenshot description")

    if has_error:
        suffix.append("Modify the code to fix the errors in " + ", ".join(error_stages) + ".")
    else:
        if is_webvoyager:
            suffix.append("Observe the above feedback and decide whether further modifications to the code are needed based on the GUI-agent testing summary. If no further modification is necessary, output <boltAction type=\"finish\"/> to signal that the task is finished. Otherwise, continue modifying the code until the requirements are fulfilled. IMPORTANT: If you decide to make modifications, do not output the finish signal.")
        else:
            suffix.append("Observe the above feedback and decide whether further modifications to the code are needed based on the screenshot observations. If no further modification is necessary, output <boltAction type=\"screenshot_validated\"/> to signal that the screenshot is satisfactory. Otherwise, continue modifying the code until the requirements are fulfilled. IMPORTANT: If you decide to make modifications, do not output the finish signal.")

    if is_webvoyager:
        info = {"feedback": feedback, "webvoyager_grade": webvoyager_grade}
    else:
        info = {"feedback": feedback, "screenshot_description": screenshot_description, "screenshot_grade_json": screenshot_grade_json, "screenshot_grade": screenshot_grade}
    feedback_str = "\n\n".join(output) + "\n\n" + "\n".join(suffix) + "\n\n" + f"**The instruction describing the website you are currently developing:**\n\n{instruction}\n\n" + reminders_prompt
    
    try:
        remove_dir(workspace_dir)
        remove_dir(log_dir)
    except:
        print("[Warning] Failed to remove directories!")

    return {
        "info": info, 
        "feedback_str": feedback_str, 
        "has_error": has_error, 
        "is_finished": False,
        "is_webvoyager": is_webvoyager,
        "webvoyager_grade": webvoyager_grade, 
        "screenshot_grade": screenshot_grade,
        "is_good_format": is_good_format,
        "is_repetition": is_repetition
    }   


def _run_single(idx_data: tuple[int, tuple[Any, ...]]):
    """
    Parameters
    ----------
    idx_data : (int, tuple) 
        *idx*   – original position in the batch  
        *data*  – arguments for get_feedback_single (already packed)

    Returns
    -------
    (idx, result, exc)
        *result* is whatever get_feedback_single returns,
        *exc*    is an Exception instance if one was raised, else None.
    """
    idx, data = idx_data
    try:
        return idx, get_feedback_single(data), None
    except Exception as exc:                # noqa: BLE001 - let caller decide
        return idx, None, exc


# ──────────────────────────────────────────────────────────────────────────
# Parallel front-end
# ──────────────────────────────────────────────────────────────────────────
def get_feedback(
    inputs_str:        Sequence[str],
    responses_str:     Sequence[str],
    gui_instructions:  Sequence[str],
    instructions:      Sequence[str],
    vlm_model:         str,
    max_workers: int = 4,
    quiet: bool = False,
) -> list[Any]:
    """
    Parallel version of the original get_feedback.

    Heavy CPU work is farmed out to separate processes.  Progress is tracked
    with a tqdm bar; failed items are returned as None (and their exceptions
    collected in `errors`, see below).

    Returns
    -------
    list
        Results in the **same order** as the inputs.  Failed items contain None.
    """
    print(f"inputs_str_len: {len(inputs_str)}, responses_str_len: {len(responses_str)}, gui_instructions_len: {len(gui_instructions)}, instructions_len: {len(instructions)}")
    # 1. Pack task arguments in the exact order they came in
    packed: list[tuple[Any, ...]] = [
        (inp + resp, gui_inst, vlm_model, inst)
        for inp, resp, gui_inst, inst in zip(
            inputs_str, responses_str, gui_instructions, instructions, strict=True
        )
    ]

    # 2. Prepare containers for results & errors
    results: list[Any | None] = [None] * len(packed)
    errors:  dict[int, Exception] = {}

    # 3. Spin up the worker pool
    #    NOTE: spawn is default on Windows; it avoids fork-unsafe issues.
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        # Submit every job with its position so we can reorder later
        futures = {
            pool.submit(_run_single, (idx, data)): idx
            for idx, data in enumerate(packed)
        }

        # 4. Consume results as they finish, updating the progress bar
        iterable = as_completed(futures)
        if not quiet:              # toggle progress bar if desired
            iterable = tqdm(iterable, total=len(futures), desc="Executing")

        for fut in iterable:
            idx = futures[fut]     # original position
            try:
                pos, value, exc = fut.result()   # pos == idx
                if exc is None:
                    results[pos] = value
                else:
                    errors[pos] = exc
            except Exception as exc:             # submit/serialization errors
                errors[idx] = exc

    # 5. Optional: report aggregated failures
    if errors and not quiet:
        print(f"[get_feedback] {len(errors)} / {len(packed)} items failed.")
        # Uncomment below to inspect all tracebacks
        for i, e in errors.items():
            print(f"─ Item #{i} ─\n{e}\n")

    return results
    
if __name__ == "__main__":
    response = "<|im_start|>assistant\nI'll create a stock report generator website using React and Vite. The implementation will include:\n\n1. Setting up Vite with React\n2. Creating components for search, report customization, and display\n3. Implementing mock stock data (since we can't connect to real APIs in WebContainer)\n4. Styling with navy components on white background\n\n<boltArtifact id=\"stock-report-generator\" title=\"Stock Report Generator Website\">\n  <boltAction type=\"file\" filePath=\"package.json\">{\n  \"name\": \"stock-report-generator\",\n  \"private\": true,\n  \"version\": \"0.0.0\",\n  \"type\": \"module\",\n  \"scripts\": {\n    \"dev\": \"vite\",\n    \"build\": \"vite build\",\n    \"preview\": \"vite preview\"\n  },\n  \"dependencies\": {\n    \"react\": \"^18.2.0\",\n    \"react-dom\": \"^18.2.0\",\n    \"react-icons\": \"^4.10.1\"\n  },\n  \"devDependencies\": {\n    \"@types/react\": \"^18.0.28\",\n    \"@types/react-dom\": \"^18.0.11\",\n    \"@vitejs/plugin-react\": \"^3.1.0\",\n    \"vite\": \"^4.2.0\"\n  }\n}</boltAction>\n\n  <boltAction type=\"file\" filePath=\"index.html\">&lt;!DOCTYPE html&gt;\n&lt;html lang=\"en\"&gt;\n  &lt;head&gt;\n    &lt;meta charset=\"UTF-8\" /&gt;\n    &lt;meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" /&gt;\n    &lt;title&gt;Stock Report Generator&lt;/title&gt;\n  &lt;/head&gt;\n  &lt;body&gt;\n    &lt;div id=\"root\"&gt;&lt;/div&gt;\n    &lt;script type=\"module\" src=\"/src/main.jsx\"&gt;&lt;/script&gt;\n  &lt;/body&gt;\n&lt;/html&gt;</boltAction>\n\n  <boltAction type=\"file\" filePath=\"src/main.jsx\">import React from 'react'\nimport ReactDOM from 'react-dom/client'\nimport App from './App.jsx'\nimport './index.css'\n\nReactDOM.createRoot(document.getElementById('root')).render(\n  &lt;React.StrictMode&gt;\n    &lt;App /&gt;\n  &lt;/React.StrictMode&gt;\n)</boltAction>\n\n  <boltAction type=\"file\" filePath=\"src/index.css\">body {\n  margin: 0;\n  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',\n    'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue',\n    sans-serif;\n  -webkit-font-smoothing: antialiased;\n  -moz-osx-font-smoothing: grayscale;\n  background-color: white;\n}\n\ncode {\n  font-family: source-code-pro, Menlo, Monaco, Consolas, 'Courier New',\n    monospace;\n}</boltAction>\n\n  <boltAction type=\"file\" filePath=\"src/App.jsx\">import { useState } from 'react'\nimport { FiSearch, FiDownload, FiBarChart2, FiDollarSign, FiTrendingUp } from 'react-icons/fi'\nimport './App.css'\n\nconst mockStocks = {\n  'AAPL': {\n    name: 'Apple Inc.',\n    price: 189.98,\n    change: 2.45,\n    changePercent: 1.31,\n    marketCap: '2.95T',\n    peRatio: 29.34,\n    dividend: 0.92,\n    sector: 'Technology',\n    description: 'Apple Inc. designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories worldwide.'\n  },\n  'MSFT': {\n    name: 'Microsoft Corporation',\n    price: 420.72,\n    change: -1.23,\n    changePercent: -0.29,\n    marketCap: '3.13T',\n    peRatio: 36.75,\n    dividend: 0.75,\n    sector: 'Technology',\n    description: 'Microsoft Corporation develops, licenses, and supports software, services, devices, and solutions worldwide.'\n  },\n  'GOOGL': {\n    name: 'Alphabet Inc.',\n    price: 175.34,\n    change: 3.12,\n    changePercent: 1.81,\n    marketCap: '2.21T',\n    peRatio: 27.45,\n    dividend: 0.00,\n    sector: 'Communication Services',\n    description: 'Alphabet Inc. provides online advertising services in the United States, Europe, the Middle East, Africa, the Asia-Pacific, Canada, and Latin America.'\n  }\n}\n\nfunction App() {\n  const [searchTerm, setSearchTerm] = useState('')\n  const [selectedStock, setSelectedStock] = useState(null)\n  const [reportOptions, setReportOptions] = useState({\n    basicInfo: true,\n    financials: true,\n    trends: true,\n    analysis: true\n  })\n  const [report, setReport] = useState(null)\n\n  const handleSearch = (e) => {\n    e.preventDefault()\n    const stock = mockStocks[searchTerm.toUpperCase()]\n    if (stock) {\n      setSelectedStock({ symbol: searchTerm.toUpperCase(), ...stock })\n      setReport(null)\n    }\n  }\n\n  const generateReport = () => {\n    if (!selectedStock) return\n    \n    let reportContent = []\n    \n    if (reportOptions.basicInfo) {\n      reportContent.push({\n        title: 'Basic Information',\n        icon: &lt;FiDollarSign /&gt;,\n        content: [\n          `Symbol: ${selectedStock.symbol}`,\n          `Name: ${selectedStock.name}`,\n          `Price: $${selectedStock.price}`,\n          `Change: ${selectedStock.change >= 0 ? '+' : ''}${selectedStock.change} (${selectedStock.changePercent}%)`,\n          `Market Cap: $${selectedStock.marketCap}`,\n          `Sector: ${selectedStock.sector}`\n        ]\n      })\n    }\n    \n    if (reportOptions.financials) {\n      reportContent.push({\n        title: 'Financial Data',\n        icon: &lt;FiBarChart2 /&gt;,\n        content: [\n          `P/E Ratio: ${selectedStock.peRatio}`,\n          `Dividend Yield: ${selectedStock.dividend}%`,\n          `52 Week High: $${(selectedStock.price * 1.2).toFixed(2)}`,\n          `52 Week Low: $${(selectedStock.price * 0.85).toFixed(2)}`\n        ]\n      })\n    }\n    \n    if (reportOptions.trends) {\n      reportContent.push({\n        title: 'Market Trends',\n        icon: &lt;FiTrendingUp /&gt;,\n        content: [\n          `3 Month Performance: ${selectedStock.changePercent > 0 ? '+' : ''}${(selectedStock.changePercent * 2.5).toFixed(2)}%`,\n          `6 Month Performance: ${selectedStock.changePercent > 0 ? '+' : ''}${(selectedStock.changePercent * 3.8).toFixed(2)}%`,\n          `1 Year Performance: ${selectedStock.changePercent > 0 ? '+' : ''}${(selectedStock.changePercent * 5.2).toFixed(2)}%`,\n          `Analyst Rating: ${selectedStock.changePercent > 0 ? 'Buy' : 'Hold'}`\n        ]\n      })\n    }\n    \n    if (reportOptions.analysis) {\n      reportContent.push({\n        title: 'Analysis',\n        icon: &lt;FiBarChart2 /&gt;,\n        content: [\n          selectedStock.description,\n          `Key Strengths: Strong brand recognition, ${selectedStock.sector === 'Technology' ? 'innovative product pipeline' : 'diversified revenue streams'}`,\n          `Risks: ${selectedStock.sector === 'Technology' ? 'Regulatory challenges, supply chain constraints' : 'Market competition, economic cycles'}`\n        ]\n      })\n    }\n    \n    setReport({\n      stock: selectedStock,\n      generatedAt: new Date().toLocaleString(),\n      content: reportContent\n    })\n  }\n\n  return (\n    &lt;div className=\"app\"&gt;\n      &lt;header className=\"header\"&gt;\n        &lt;h1&gt;Stock Report Generator&lt;/h1&gt;\n      &lt;/header&gt;\n      \n      &lt;div className=\"search-container\"&gt;\n        &lt;form onSubmit={handleSearch}&gt;\n          &lt;div className=\"search-box\"&gt;\n            &lt;input\n              type=\"text\"\n              placeholder=\"Enter stock symbol (e.g. AAPL)\"\n              value={searchTerm}\n              onChange={(e) =&gt; setSearchTerm(e.target.value)}\n            /&gt;\n            &lt;button type=\"submit\"&gt;&lt;FiSearch /&gt; Search&lt;/button&gt;\n          &lt;/div&gt;\n        &lt;/form&gt;\n      &lt;/div&gt;\n      \n      {selectedStock && (\n        &lt;div className=\"stock-container\"&gt;\n          &lt;div className=\"stock-info\"&gt;\n            &lt;h2&gt;{selectedStock.symbol} - {selectedStock.name}&lt;/h2&gt;\n            &lt;div className=\"price\"&gt;\n              &lt;span className=\"current-price\"&gt;${selectedStock.price}&lt;/span&gt;\n              &lt;span className={`change ${selectedStock.change >= 0 ? 'positive' : 'negative'}`}&gt;\n                {selectedStock.change >= 0 ? '+' : ''}{selectedStock.change} ({selectedStock.changePercent}%)\n              &lt;/span&gt;\n            &lt;/div&gt;\n          &lt;/div&gt;\n          \n          &lt;div className=\"report-options\"&gt;\n            &lt;h3&gt;Customize Report&lt;/h3&gt;\n            &lt;div className=\"options-grid\"&gt;\n              &lt;label&gt;\n                &lt;input\n                  type=\"checkbox\"\n                  checked={reportOptions.basicInfo}\n                  onChange={() =&gt; setReportOptions({...reportOptions, basicInfo: !reportOptions.basicInfo})}\n                /&gt;\n                Basic Information\n              &lt;/label&gt;\n              &lt;label&gt;\n                &lt;input\n                  type=\"checkbox\"\n                  checked={reportOptions.financials}\n                  onChange={() =&gt; setReportOptions({...reportOptions, financials: !reportOptions.financials})}\n                /&gt;\n                Financial Data\n              &lt;/label&gt;\n              &lt;label&gt;\n                &lt;input\n                  type=\"checkbox\"\n                  checked={reportOptions.trends}\n                  onChange={() =&gt; setReportOptions({...reportOptions, trends: !reportOptions.trends})}\n                /&gt;\n                Market Trends\n              &lt;/label&gt;\n              &lt;label&gt;\n                &lt;input\n                  type=\"checkbox\"\n                  checked={reportOptions.analysis}\n                  onChange={() =&gt; setReportOptions({...reportOptions, analysis: !reportOptions.analysis})}\n                /&gt;\n                Analysis\n              &lt;/label&gt;\n            &lt;/div&gt;\n            &lt;button onClick={generateReport} className=\"generate-btn\"&gt;\n              Generate Report\n            &lt;/button&gt;\n          &lt;/div&gt;\n        &lt;/div&gt;\n      )}\n      \n      {report && (\n        &lt;div className=\"report-container\"&gt;\n          &lt;div className=\"report-header\"&gt;\n            &lt;h2&gt;{report.stock.symbol} Stock Report&lt;/h2&gt;\n            &lt;div className=\"report-meta\"&gt;\n              &lt;span&gt;Generated: {report.generatedAt}&lt;/span&gt;\n              &lt;button className=\"download-btn\"&gt;&lt;FiDownload /&gt; Download PDF&lt;/button&gt;\n            &lt;/div&gt;\n          &lt;/div&gt;\n          \n          {report.content.map((section, index) =&gt; (\n            &lt;div key={index} className=\"report-section\"&gt;\n              &lt;h3&gt;{section.icon} {section.title}&lt;/h3&gt;\n              &lt;ul&gt;\n                {section.content.map((item, i) =&gt; (\n                  &lt;li key={i}&gt;{item}&lt;/li&gt;\n                ))}\n              &lt;/ul&gt;\n            &lt;/div&gt;\n          ))}\n        &lt;/div&gt;\n      )}\n    &lt;/div&gt;\n  )\n}\n\nexport default App</boltAction>\n\n  <boltAction type=\"file\" filePath=\"src/App.css\">.app {\n  max-width: 1200px;\n  margin: 0 auto;\n  padding: 20px;\n  background-color: white;\n}\n\n.header {\n  text-align: center;\n  margin-bottom: 30px;\n  color: navy;\n}\n\n.search-container {\n  margin-bottom: 30px;\n}\n\n.search-box {\n  display: flex;\n  max-width: 500px;\n  margin: 0 auto;\n}\n\n.search-box input {\n  flex: 1;\n  padding: 10px;\n  border: 1px solid navy;\n  border-radius: 4px 0 0 4px;\n  font-size: 16px;\n}\n\n.search-box button {\n  padding: 10px 15px;\n  background-color: navy;\n  color: white;\n  border: none;\n  border-radius: 0 4px 4px 0;\n  cursor: pointer;\n  display: flex;\n  align-items: center;\n  gap: 5px;\n}\n\n.search-box button:hover {\n  background-color: #001f3f;\n}\n\n.stock-container {\n  background-color: #f8f9fa;\n  border-radius: 8px;\n  padding: 20px;\n  margin-bottom: 30px;\n  border: 1px solid #dee2e6;\n}\n\n.stock-info h2 {\n  margin-top: 0;\n  color: navy;\n}\n\n.price {\n  display: flex;\n  align-items: center;\n  gap: 15px;\n  margin: 10px 0;\n}\n\n.current-price {\n  font-size: 28px;\n  font-weight: bold;\n  color: navy;\n}\n\n.change {\n  font-size: 16px;\n  padding: 4px 8px;\n  border-radius: 4px;\n}\n\n.change.positive {\n  background-color: #d4edda;\n  color: #155724;\n}\n\n.change.negative {\n  background-color: #f8d7da;\n  color: #721c24;\n}\n\n.report-options {\n  margin-top: 20px;\n  padding-top: 20px;\n  border-top: 1px solid #dee2e6;\n}\n\n.report-options h3 {\n  margin-top: 0;\n  color: navy;\n}\n\n.options-grid {\n  display: grid;\n  grid-template-columns: repeat(2, 1fr);\n  gap: 10px;\n  margin: 15px 0;\n}\n\n.options-grid label {\n  display: flex;\n  align-items: center;\n  gap: 8px;\n  cursor: pointer;\n}\n\n.generate-btn {\n  padding: 10px 20px;\n  background-color: navy;\n  color: white;\n  border: none;\n  border-radius: 4px;\n  cursor: pointer;\n  font-size: 16px;\n}\n\n.generate-btn:hover {\n  background-color: #001f3f;\n}\n\n.report-container {\n  background-color: white;\n  border-radius: 8px;\n  padding: 20px;\n  box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);\n  border: 1px solid #dee2e6;\n}\n\n.report-header {\n  display: flex;\n  justify-content: space-between;\n  align-items: center;\n  margin-bottom: 20px;\n  padding-bottom: 15px;\n  border-bottom: 1px solid #dee2e6;\n}\n\n.report-header h2 {\n  margin: 0;\n  color: navy;\n}\n\n.report-meta {\n  display: flex;\n  align-items: center;\n  gap: 15px;\n}\n\n.download-btn {\n  padding: 8px 15px;\n  background-color: navy;\n  color: white;\n  border: none;\n  border-radius: 4px;\n  cursor: pointer;\n  display: flex;\n  align-items: center;\n  gap: 5px;\n}\n\n.download-btn:hover {\n  background-color: #001f3f;\n}\n\n.report-section {\n  margin-bottom: 25px;\n}\n\n.report-section h3 {\n  display: flex;\n  align-items: center;\n  gap: 10px;\n  color: navy;\n  margin-bottom: 10px;\n}\n\n.report-section ul {\n  margin: 0;\n  padding-left: 20px;\n}\n\n.report-section li {\n  margin-bottom: 8px;\n  line-height: 1.5;\n}\n\n@media (max-width: 768px) {\n  .options-grid {\n    grid-template-columns: 1fr;\n  }\n  \n  .report-header {\n    flex-direction: column;\n    align-items: flex-start;\n    gap: 10px;\n  }\n  \n  .report-meta {\n    width: 100%;\n    justify-content: space-between;\n  }\n}</boltAction>\n\n  <boltAction type=\"start\">npm run dev</boltAction>\n</boltArtifact><|im_end|><|im_start|>user\nInstallation of dependencies was successful.\n\nStarting the service was successful.\n\n**The screenshot description:**\n\nThe screenshot shows a simple webpage titled 'Stock Report Generator' in bold blue text at the top center. Below the title, there is a search input field with placeholder text 'Enter stock symbol (e.g. AAPL)' and a blue button labeled 'Search' with a magnifying glass icon on its left. The overall layout is minimalistic, featuring a single-column design with ample white space around the elements. The color scheme consists of a white background with blue accents for the title and search button. The font appears to be a standard sans-serif typeface, and there are no other UI components or additional content visible.\n\n**Suggestions for Improvement:**\n\n1. Consider adding some visual hierarchy or decorative elements to make the page more engaging. 2. Include a brief description or instructions near the search field to guide users on how to use the tool. 3. Add loading indicators or feedback messages to inform users after they click the 'Search' button.\n\nObserve the above feedback and decide whether further modifications to the code are needed based on the screenshot observations. If no further modification is necessary, output <boltAction type=\"screenshot_validated\"/> to signal that the screenshot is satisfactory. Otherwise, continue modifying the code until the requirements are fulfilled. IMPORTANT: If you decide to make modifications, do not output the finish signal.\n\n**The instruction describing the website you are currently developing:**\n\nPlease implement a website for generating stock reports to provide stock information and analysis. The website should have the functionality to search and summarize stock information, and generate customized stock reports based on user requirements. Users should be able to input stock codes or names, select report formats and content, and the website will automatically generate the corresponding reports. The reports should include basic stock information, market trends, financial data, and more. Set the background color to white and the component color to navy.\n\nCRITICAL: Always provide the FULL, updated content of a file when editing. This means, if you need to create or modify a file:\n\n      - Include ALL code in the file, even if parts are unchanged\n      - NEVER use placeholders like \"// rest of the code remains the same...\" or \"<- leave original code here ->\"\n      - ALWAYS show the complete, up-to-date file content when updating the file\n      - Avoid any form of truncation or summarization<|im_end|><|im_start|>assistant\n<boltAction type=\"finish\"/><|im_end|>"
    gui_instruction = "Verify white background and navy buttons. Search and summarize stock information, generate customized stock reports by inputting stock codes or names, and selecting report formats and content. Check that reports include basic stock information, market trends, and financial data.\n\nIf prompted for a username, password, or email in the process of testing, enter \"admin,\" \"admin123456\", and \"admin@example.com\", respectively.\n\nAnswer with one of the following:\n- YES: if the testing instruction was fully achieved during your interactions.\n- NO: if the testing instruction was not achieved at all.\nProvide your final answer based on your testing experience."
    vlm_model = "/mnt/cache/sharemath/models/Qwen/Qwen2.5-VL-32B-Instruct"
    instruction = "Please implement a website for generating stock reports to provide stock information and analysis. The website should have the functionality to search and summarize stock information, and generate customized stock reports based on user requirements. Users should be able to input stock codes or names, select report formats and content, and the website will automatically generate the corresponding reports. The reports should include basic stock information, market trends, financial data, and more. Set the background color to white and the component color to navy."
    data = (response, gui_instruction, vlm_model, instruction)
    result = get_feedback_single(data)
    import json
    with open("/mnt/cache/luzimu/code_agent/WebGen-RL/src/tests/feedback.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)

    # import json
    # with open("/mnt/cache/luzimu/code_agent/WebGen-RL/src/tests/inputs_responses1.json", "r", encoding="utf-8") as f:
    #     inputs_responses = json.load(f)

    # vlm_model = "/mnt/cache/sharemath/models/Qwen/Qwen2.5-VL-32B-Instruct"
    # instructions = []
    # gui_instructions = []
    # for input_str, response_str in zip(inputs_responses["inputs_str"], inputs_responses["responses_str"]):
    #     messages = prompt_to_messages(input_str + response_str)
    #     instruction = messages[1]["content"]
    #     gui_instruction = generate_gui_agent_instruction(instruction, vlm_model)
    #     instructions.append(instruction)
    #     gui_instructions.append(gui_instruction)

    # start = time.time()
    # results = get_feedback(inputs_responses["inputs_str"], inputs_responses["responses_str"], gui_instructions, instructions, vlm_model)
    # exe_time = time.time() - start

    # for instruction, gui_instruction, result in zip(instructions, gui_instructions, results):
    #     result["instruction"] = instruction
    #     result["gui_instruction"] = gui_instruction
    # print(exe_time)
    # with open("/mnt/cache/luzimu/code_agent/WebGen-RL/src/tests/feedback.jsonl", "w", encoding="utf-8") as f:
    #     for result in results:
    #         f.write(json.dumps(result, ensure_ascii=False) + "\n")