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

import sys
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

from .timestamp import current_timestamp
from .get_webvoyager_feedback import get_webvoyager_feedback

import signal, platform, subprocess, time


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
        if process:
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
        if len(start_output) > 50000:
            start_output = start_output[:5000] + "\n......\n[Truncated]\n......\n" + start_output[-5000:]
            suffix = "\n\n...... [Output Too Long, Truncated]"
        start_output = start_output.strip() + suffix
        feedback["start_results"] = start_output
    if "error" in feedback["start_results"].lower():
        feedback["start_error"] = True

    with open(os.path.join(log_dir, "service.pid"), "w") as f:
        f.write(str(process.pid))

    return feedback


def execute_for_webvoyager_feedback(instruction, project_dir, log_dir, vlm_model, model, cmds=["npm install"], start_cmd="npm run dev", step_idx=None, max_tokens=-1, max_completion_tokens=-1):
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
            model=model,
            max_tokens=max_tokens,
            max_completion_tokens=max_completion_tokens
        )
    except Exception as e:
        feedback["webvoyager_error"] = f"Error: {e}"
        print(f"Error: {e}")
    finally:
        if process:
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
        if len(start_output) > 50000:
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


if __name__ == "__main__":
    url = "http://127.0.0.1:5187/"
    take_screenshot(url, output_path="/mnt/cache/agent/Zimu/WebGen-Agent/service_logs/debug/000002/screenshot.png")