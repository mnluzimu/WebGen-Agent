import os
import json
import shutil
import time

import sys
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

from prompts import system_prompt, function_test_prompt, reminders_prompt
from utils import (
    llm_generation,
    extract_and_write_files,
    get_sorted_file_paths,
    execute_for_feedback,
    execute_for_webvoyager_feedback,
    get_screenshot_description,
    get_screenshot_grade,
    current_timestamp,
    generate_gui_agent_instruction,
    directory_to_dict,
    dict_to_directory,
    restore_from_last_step
)


import re
from typing import Dict, Tuple, Any


def remove_dir(directory):
    for _ in range(5):
        try:
            shutil.rmtree(directory)
            return True
        except:
            time.sleep(5)
    return False


class WebGenAgent:

    def __init__(
        self,
        model: str,
        vlm_model: str,
        fb_model: str,
        workspace_dir: str,
        log_dir: str,
        instruction: str,
        max_iter: int,
        overwrite: bool,
        error_limit: int,
        max_tokens: int = -1,
        max_completion_tokens: int = -1,
        temperature: float = 0.5
    ) -> None:
        self.model = model
        self.vlm_model = vlm_model
        self.fb_model = fb_model
        self.max_tokens = max_tokens
        self.max_completion_tokens = max_completion_tokens
        self.temperature = temperature
        if os.path.exists(workspace_dir):
            remove_dir(workspace_dir)
        os.makedirs(workspace_dir)
        # if exist history files, overwrite them
        if overwrite:
            if os.path.exists(log_dir):
                remove_dir(log_dir)
            os.makedirs(log_dir)
        else:
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
        
        self.id = os.path.basename(log_dir)
        self.is_finished = False
        messages, gui_instruction, step_idx, screenshot_grade, webvoyager_grade, nodes = restore_from_last_step(log_dir, workspace_dir, max_iter) 
        self.workspace_dir = workspace_dir
        self.log_dir = log_dir
        if messages is not None and len(messages) > 0:
            self.messages = messages
            self.gui_instruction = gui_instruction
            if messages[-1]["info"].get("is_finish", False):
                self.is_finished = True
            self.step_idx = step_idx
            print(f"[{self.id}] Resuming from step {step_idx},")
            self.screenshot_grade, self.webvoyager_grade = screenshot_grade, webvoyager_grade
            self.nodes = nodes
            self.pre = step_idx
            self.error_count = self.get_error_count(f"step{step_idx}.json")
        else:
            self.messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": instruction},
            ]
            self.gui_instruction = generate_gui_agent_instruction(instruction, fb_model, self.max_tokens, self.max_completion_tokens)
            self.pre = -1
            self.step_idx = -1
            self.screenshot_grade, self.webvoyager_grade = 0, 0
            self.nodes = {}
            self.error_count = 0
        self.instruction = instruction
        self.max_iter = max_iter
        self.error_limit = error_limit
        self.restart_limit = 5
    
    def get_concise_messages(self):
        concise_messages = []
        for message in self.messages:
            concise_messages.append({"role": message["role"], "content": message["content"]})
        return concise_messages

    def get_feedback(self, cmds, start_cmd, step_idx, is_webvoyager=False, gui_instruction=None):
        if is_webvoyager:
            feedback = execute_for_webvoyager_feedback(
                gui_instruction, 
                self.workspace_dir, 
                self.log_dir, 
                self.vlm_model,
                self.fb_model,
                cmds, 
                start_cmd,
                step_idx
            )
        else:
            feedback = execute_for_feedback(self.workspace_dir, self.log_dir, cmds, start_cmd, step_idx)
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
                if "improvement_suggestions" in webvoyager_feedback.keys():
                    if len(webvoyager_feedback["improvement_suggestions"]) > 0:
                        output.append(f"**The suggestions based on the GUI-agent testing result:**\n\n{webvoyager_feedback['improvement_suggestions']}")
                    else:
                        output.append(f"The GUI agent testing is successful and no further improvement is necessary.")
                else:
                    output.append(f"Failed to get GUI agent feedback or GUI agent error messages.")
                    has_error = True
                    error_stages.append("GUI agent trajectory collection")
            self.webvoyager_grade = webvoyager_grade
        else:
            screenshot_description = None
            screenshot_grade_json, screenshot_grade = None, 0
            if len(feedback["screenshot_error"]) > 0:
                output.append(f"There was an error when getting the screenshot of the started website:\n\n{feedback['screenshot_error']}")
                has_error = True
                error_stages.append("screenshot collection")
            elif os.path.isfile(feedback["screenshot_path"]):
                screenshot_description = get_screenshot_description(feedback["screenshot_path"], self.vlm_model)
                screenshot_grade_json, screenshot_grade = get_screenshot_grade(feedback["screenshot_path"], self.vlm_model, self.instruction)
                self.screenshot_grade = screenshot_grade
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
        feedback_str = "\n\n".join(output) + "\n\n" + "\n".join(suffix) + "\n\n" + f"**The instruction describing the website you are currently developing:**\n\n{self.instruction}\n\n" + reminders_prompt
        return info, feedback_str, has_error

    def get_cmds(self, output):
        cmds = ["npm install"]
        start_cmd = "npm run dev"
        return cmds, start_cmd

    def step(self, i):
        concise_messages = self.get_concise_messages()
        output = llm_generation(concise_messages, self.model, max_tokens=self.max_tokens, max_completion_tokens=self.max_completion_tokens, temperature=self.temperature)

        info = {}
        has_error = False
        if 'boltAction type="finish"' in output or "boltAction type='finish'" in output:
            info["is_finish"] = True
            self.messages.append({"role": "assistant", "content": output, "info": info})
        elif 'boltAction type="screenshot_validated"' in output or "boltAction type='screenshot_validated'" in output:
            self.messages.append({"role": "assistant", "content": output, "info": info})
            # webvoyager and get feedback
            cmds, start_cmd = self.get_cmds(output)
            info, feedback_str, has_error = self.get_feedback(cmds, start_cmd, i, is_webvoyager=True, gui_instruction=self.gui_instruction)

            self.messages.append({"role": "user", "content": feedback_str, "info": info})
        else:
            extract_and_write_files(output, self.workspace_dir)
            self.screenshot_grade, self.webvoyager_grade = 0, 0
            self.messages.append({"role": "assistant", "content": output, "info": info})
            
            # execute and get feedback
            cmds, start_cmd = self.get_cmds(output)
            info, feedback_str, has_error = self.get_feedback(cmds, start_cmd, i)

            self.messages.append({"role": "user", "content": feedback_str, "info": info})
        return info, has_error

    def save_history(self, i, pre=None, has_error=False):
        print(f"[{self.id}] Saving history for step {i},")
        output_file = os.path.join(self.log_dir, f"step{i}.json")
        if self.screenshot_grade <= 2:
            has_error = True
        
        self.nodes[f"step{i}.json"] = {
            "screenshot_grade": self.screenshot_grade, 
            "webvoyager_grade": self.webvoyager_grade,
            "pre": pre,
            "has_error": has_error
        }

        with open(output_file, "w", encoding="utf-8") as f:
            data = {
                "messages": self.messages, 
                "gui_instruction": self.gui_instruction, 
                "files": directory_to_dict(self.workspace_dir), 
                "screenshot_grade": self.screenshot_grade, 
                "webvoyager_grade": self.webvoyager_grade,
                "pre": pre,
                "nodes": self.nodes,
                "has_error": has_error
            }
            json.dump(data, f)
        
        if has_error:
            self.error_count += 1
        else:
            self.error_count = 0

    def _extract_step_index(self, filename: str) -> int:
        m = re.search(r"step(\d+)\.json$", filename)
        return int(m.group(1)) if m else -1

    def choose_best_node(self) -> Tuple[str, Dict[str, Any]]:
        """
        Ranking rules
        -------------
        1. **Screenshot filter**  
        * If any node has ``screenshot_grade >= 3``, restrict the search
            to those nodes only.  
        * Otherwise consider *all* nodes (even if their
            ``screenshot_grade`` is 2 or below).
        2. **Primary metric** – highest ``webvoyager_grade`` wins.
        3. **Secondary metric** – if tied, highest ``screenshot_grade`` wins.
        4. **Tertiary metric** – if still tied, choose the node whose filename
        has the largest step index *i* in ``"step{i}.json"``.
        """
        if not self.nodes or len(self.nodes) == 0:
            None, None, False
            # raise ValueError("No nodes supplied.")

        # Screenshot filter
        good_nodes = [
            item for item in self.nodes.items()
            if item[1].get("screenshot_grade", 0) is not None and item[1].get("screenshot_grade", 0) >= 3 and not item[1].get("has_error", False)
        ]

        # Check whether valid nodes exist
        has_valid = True
        if len(good_nodes) == 0:
            has_valid = False
            candidates = list(self.nodes.items())
        else:
            candidates = good_nodes

        # Rank using a composite key
        def rank_key(item):
            fname, record = item
            return (
                record.get("webvoyager_grade", -float("inf")),
                record.get("screenshot_grade", -float("inf")),
                self._extract_step_index(fname),
            )

        best_item = max(candidates, key=rank_key)
        return best_item[0], best_item[1], has_valid

    def get_error_count(self, file_name):
        error_count = 0
        curr = self._extract_step_index(file_name)
        while curr != -1:
            if self.nodes[f"step{curr}.json"]["has_error"]:
                error_count += 1
            else:
                break
            curr = self.nodes[f"step{curr}.json"]["pre"]
        return error_count
    
    def run(self):
        errored = False
        restart = False
        restart_num = 0
        if not self.is_finished:
            for i in range(self.step_idx + 1, self.max_iter):
                print(f"[{self.id}] Error count: {self.error_count}")
                if self.error_count >= self.error_limit or restart:
                    if restart:
                        print(f"[{self.id}] Restarting step {i} due to previous error...")
                    else:
                        print(f"[{self.id}] Exceed error limit, backtracking...")
                    file_name, record, has_valid = self.choose_best_node()
                    if has_valid:
                        with open(os.path.join(self.log_dir, file_name), "r", encoding="utf-8") as f:
                            data = json.load(f)
                        dict_to_directory(data["files"], self.workspace_dir)
                        self.messages = data["messages"]
                        self.pre = self._extract_step_index(file_name)
                        self.screenshot_grade, self.webvoyager_grade = self.nodes[f"step{self.pre}.json"]["screenshot_grade"], self.nodes[f"step{self.pre}.json"]["webvoyager_grade"]
                        self.error_count = self.get_error_count(file_name)
                    else:
                        self.messages = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": self.instruction},
                        ]
                        self.pre = -1
                        self.screenshot_grade, self.webvoyager_grade = 0, 0
                        self.error_count = 0
                        if os.path.exists(self.workspace_dir):
                            remove_dir(self.workspace_dir)
                        os.makedirs(self.workspace_dir)

                print(f"[{self.id}] ==== Step {i} ====")
                try:
                    info, has_error = self.step(i)
                except Exception as e:
                    print(f"[{self.id}] Caught error:", e)
                    has_error = True
                    info = {"error": str(e)}
                    if restart_num < self.restart_limit:
                        print(f"[{self.id}] Restarting step {i} due to error, attempt {restart_num + 1}/{self.restart_limit}...")
                        restart_num += 1
                        restart = True
                        time.sleep(5)
                        continue
                    else:
                        print(f"[{self.id}] Max restart limit reached, terminating...")
                        errored = True
                        break
                self.save_history(i, pre=self.pre, has_error=has_error)
                self.pre = i
                if info.get("is_finish", False):
                    print(f"[{self.id}] Task completed!")
                    self.is_finished = True
                    break
                
        if not self.is_finished:
            print(f"[{self.id}] Max iteration reached!")
        if errored:
            print(f"[{self.id}] Terminated due to error!")
        
        file_name, record, has_valid = self.choose_best_node()
        with open(os.path.join(self.log_dir, file_name), "r", encoding="utf-8") as f:
            data = json.load(f)
        dict_to_directory(data["files"], self.workspace_dir)
        data["node"] = file_name
        
        print(f"[{self.id}] Chosen node {file_name}, returning...")
        return data
    
    