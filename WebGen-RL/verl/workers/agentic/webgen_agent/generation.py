import torch
import re
import os
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from .tensor_helper import TensorHelper, TensorConfig
from verl import DataProto
from .utils import execute_sqls_parallel
import pandas as pd 
from time import perf_counter
import logging
import copy

from .feedback.file_management import prompt_to_messages
from .feedback.generate_gui_agent_instruction import generate_gui_agent_instruction
from .feedback import get_feedback
from .feedback.has_repetition import has_repetition

@dataclass
class GenerationConfig:
    max_turns: int
    max_start_length: int
    max_prompt_length: int 
    max_response_length: int
    max_obs_length: int
    num_gpus: int
    vlm_model: str
    no_think_rl: bool = False
    project_name: str = "webgen"
    experiment_name: str = "webgen_experiment"
    

class LLMGenerationManager:
    def __init__(
        self,
        tokenizer,
        generator,
        config: GenerationConfig,
        is_validation: bool = False,
    ):
        self.tokenizer = tokenizer
        self.generator = generator 
        self.config = config
        self.is_validation = is_validation

        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.max_obs_length,
            max_start_length=config.max_start_length
        ))

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']

    def _postprocess_responses(self, input_ids: torch.Tensor, responses: torch.Tensor, exceed_length_mask: torch.Tensor, exceed_length_tag: torch.Tensor) -> torch.Tensor:
        """Process responses to stop at sql operation or solution operation."""
        orig_device = responses.device
        responses_str = self.tokenizer.batch_decode(
            responses, 
            skip_special_tokens=False
        )
        for i in range(len(responses_str)):
            responses_str[i] = responses_str[i].replace("<|endoftext|>", "")
            if len(self.tokenizer.encode(responses_str[i])) >= self.config.max_response_length:
                if not has_repetition(responses_str[i]):
                    # If exceeds max length and not a repetition, the whole trajectory is deserted and not used for training
                    exceed_length_mask[i] = True
                # the current response exceeds max length, if it is a repetition, we still keep it to compute loss on it
                exceed_length_tag[i] = True
            if not responses_str[i].endswith("<|im_end|>"):
                responses_str[i] += "<|im_end|>"

        inputs_str = self.tokenizer.batch_decode(
            input_ids, 
            skip_special_tokens=False
        )

        responses = self._batch_tokenize(responses_str).to(orig_device)

        # import json
        # with open("/mnt/cache/luzimu/code_agent/WebGen-RL/src/tests/inputs_responses1.json", "w", encoding="utf-8") as f:
        #     json.dump({"inputs_str": inputs_str, "responses_str": responses_str}, f, ensure_ascii=False)

        return responses, responses_str, inputs_str, exceed_length_mask, exceed_length_tag

    def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
        """Process next observations from environment."""
        next_obs_ids = self.tokenizer(
            next_obs, 
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,  # Prevents adding special tokens
        )['input_ids'].type(torch.int64)

        if next_obs_ids.shape[1] > self.config.max_obs_length:
            print(f"[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, {next_obs_ids.shape[1]} & {self.config.max_obs_length}")            
            next_obs_ids = next_obs_ids[:, :self.config.max_obs_length]

        return next_obs_ids

    def _update_rolling_state(self, rollings: DataProto, cur_responses: torch.Tensor, 
                            next_obs_ids: torch.Tensor) -> Dict:
        """Update rolling state with new responses and observations."""
        # Concatenate and handle padding 
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            cur_responses,
            next_obs_ids
        ])
        
        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)

        new_rollings = DataProto.from_dict({
            'input_ids': new_input_ids[:, -max_len:],
            'position_ids': new_position_ids[:, -max_len:],
            'attention_mask': new_attention_mask[:, -max_len:]
        })
        new_rollings.meta_info.update(rollings.meta_info)
        
        return new_rollings

    def _info_masked_concatenate_with_padding(self, 
                prompt: torch.Tensor, 
                prompt_with_mask: torch.Tensor, 
                response: torch.Tensor, 
                info: torch.Tensor = None,
                pad_to_left: bool = True
            ) -> torch.Tensor:
        """Concatenate tensors and handle padding. Additionally, create a mask (info_mask) to cover the information block if it exists."""
        pad_id = self.tokenizer.pad_token_id
        tensors = [prompt, response]
        tensors_with_mask = [prompt_with_mask, response]
        if info is not None:
            tensors.append(info)
            info_mask = torch.full(info.size(), pad_id, dtype=info.dtype, device=info.device) # information mask
            tensors_with_mask.append(info_mask)
        
        concatenated = torch.cat(tensors, dim=1)
        concatenated_with_info = torch.cat(tensors_with_mask, dim=1)
        mask = concatenated != pad_id if pad_to_left else concatenated == pad_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        padded_tensor = concatenated.gather(1, sorted_indices)
        padded_tensor_with_info = concatenated_with_info.gather(1, sorted_indices)

        return padded_tensor, padded_tensor_with_info

    def _update_right_side(self, right_side: Dict, 
                          cur_responses: torch.Tensor,
                          next_obs_ids: torch.Tensor = None) -> Dict:
        """Update right side state."""
        if next_obs_ids != None:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    next_obs_ids, 
                    pad_to_left=False
                )
        else:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    pad_to_left=False
                )
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        return {'responses': responses[:, :max_len], 'responses_with_info_mask': responses_with_info_mask[:, :max_len]}

    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
            Wrapper for generation that handles multi-GPU padding requirements.
            if num_gpus <= 1, return self.generator.generate_sequences(active_batch)
            if active_batch size is not divisible by num_gpus, pad with first sequence
            then remove padding from output
        """
        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            return self.generator.generate_sequences(active_batch)
            
        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus
        
        for key in active_batch.batch.keys():
            active_batch.batch[key] = active_batch.batch[key].long()
        if remainder == 0:
            return self.generator.generate_sequences(active_batch)
        
        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}
        
        for k, v in active_batch.batch.items():
            # Use first sequence as padding template
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)

        padded_active_batch = DataProto.from_dict(padded_batch)
        for key in padded_active_batch.batch.keys():
            padded_active_batch.batch[key] = padded_active_batch.batch[key].long()

        # Generate with padded batch
        # print(f"[DEBUGGING] Padded active batch size: {padded_active_batch.batch['input_ids'].shape}")
        padded_output = self.generator.generate_sequences(padded_active_batch)

        # Remove padding from output
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        
        # Handle meta_info if present
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta
            
        padded_output.batch = trimmed_batch
        return padded_output

    def _get_instructions_and_gui_instructions(self, initial_input_ids: torch.Tensor):
        """Get instructions and gui_instructions from the inital input_ids."""
        inputs_str = self.tokenizer.batch_decode(initial_input_ids)
        instructions = []
        gui_instructions = []
        for input_str in inputs_str:
            messages = prompt_to_messages(input_str)
            instruction = messages[1]["content"]
            gui_instruction = generate_gui_agent_instruction(instruction, self.config.vlm_model)
            instructions.append(instruction)
            gui_instructions.append(gui_instruction)
        return instructions, gui_instructions

    def run_llm_loop(self, gen_batch, initial_input_ids: torch.Tensor) -> DataProto:
        """Run main LLM generation loop."""

        from .feedback.timestamp import current_timestamp
        self.dir_timestamp = current_timestamp()
        
        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        original_right_side = {'responses': initial_input_ids[:, []], 'responses_with_info_mask': initial_input_ids[:, []]}
        
        instructions, gui_instructions = self._get_instructions_and_gui_instructions(gen_batch.batch['input_ids'])
        
        # import json
        # with open("/mnt/cache/luzimu/code_agent/WebGen-RL/src/tests/instruction.json", "w", encoding="utf-8") as f:
        #     json.dump({"instructions": instructions, "gui_instructions": gui_instructions}, f, ensure_ascii=False)

        active_mask = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.bool)
        exceed_length_mask = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.bool)
        turns_stats = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        valid_action_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        valid_search_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        active_num_list = [active_mask.sum().item()]
        rollings = gen_batch

        orig_prompt_lengths = gen_batch.batch['attention_mask'].sum(-1)

        # scores would be [[{"screenshot_score": s1, "webvoyager_score": s2, "position": p}, ...], [...], [...]]
        scores = [[] for _ in range(gen_batch.batch['input_ids'].shape[0])]

        # Main generation loop
        for step in range(self.config.max_turns):
            logging.info(f"run_llm_loop::STEP]: Step {step}, gen_batch size: {gen_batch.batch['input_ids'].shape[0]}")
            start_step = perf_counter()

            if not active_mask.sum():
                break
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            
            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })

            start = perf_counter()
            gen_output = self._generate_with_gpu_padding(rollings_active)
            end = perf_counter()
            logging.info(f"run_llm_loop::generate_with_gpu_padding]: vLLM generation in {end - start:.2f} seconds")
            
            start = perf_counter()
            meta_info = gen_output.meta_info            
            logging.info(f"device after gen -> {gen_output.batch['responses'].device}")
            exceed_length_tag = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.bool)
            responses_ids, responses_str, inputs_str, exceed_length_mask, exceed_length_tag = self._postprocess_responses(rollings.batch["input_ids"], gen_output.batch['responses'], exceed_length_mask, exceed_length_tag)
            responses_ids = responses_ids.to(rollings_active.batch["input_ids"].device)
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)
            end = perf_counter()
            logging.info(f"run_llm_loop::_postprocess_responses]: Execution in {end - start:.2f} seconds")

            # Execute in environment and process observations
            # NOTE(shu): execute predictions here, where to truncate only first response? ^ postprogess?
            start = perf_counter()
            next_obs, dones, scores = self.execute_predictions(
                inputs_str, responses_str, gui_instructions, instructions, scores, active_mask, step, exceed_length_tag
            )
            end = perf_counter()
            
            logging.info(f"run_llm_loop::execute_predictions]: Execution in {end - start:.2f} seconds")
        
            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            turns_stats[curr_active_mask] += 1

            start = perf_counter()
            next_obs_ids = self._process_next_obs(next_obs).to(rollings_active.batch["input_ids"].device)
            end = perf_counter()
            logging.info(f"run_llm_loop::process_next_obs]: Execution in {end - start:.2f} seconds")
            
            # Update states
            rollings = self._update_rolling_state(
                rollings,
                responses_ids,
                next_obs_ids
            )
            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
                next_obs_ids
            )

            reward_positions = rollings.batch['attention_mask'].sum(-1) - orig_prompt_lengths
            for i, reward_position in enumerate(reward_positions):
                if len(scores[i]) > 0:
                    scores[i][-1]["position"] = reward_position

            end = perf_counter()
            logging.info(f"run_llm_loop::STEP]: STEP finishes in {end - start_step:.2f} seconds")
        
        meta_info['turns_stats'] = turns_stats.tolist()
        meta_info['active_mask'] = active_mask.tolist()
        
        print("ACTIVE_TRAJ_NUM:", active_num_list)
        
        return self._compose_final_output(original_left_side, original_right_side, meta_info, scores, exceed_length_mask)

    def _compose_final_output(self, left_side: Dict,
                            right_side: Dict,
                            meta_info: Dict,
                            scores: list,
                            exceed_length_mask: torch.Tensor) -> DataProto:
        """Compose final generation output."""
        final_output = right_side.copy()
        final_output['prompts'] = left_side['input_ids']
        # Combine input IDs
        final_output['input_ids'] = torch.cat([
            left_side['input_ids'],
            right_side['responses']
        ], dim=1)
        
        # Create attention mask and position ids
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses'])
        ], dim=1)
        final_output['info_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses_with_info_mask'])
        ], dim=1)
        
        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            final_output['attention_mask']
        )

        for i in range(final_output['input_ids'].shape[0]):
            if exceed_length_mask[i]:
                # final_output["attention_mask"][i] = torch.zeros_like(final_output["attention_mask"][i], dtype=final_output["attention_mask"][i].dtype)
                final_output["position_ids"][i] = torch.zeros_like(final_output["position_ids"][i], dtype=final_output["position_ids"][i].dtype)
                final_output["info_mask"][i] = torch.zeros_like(final_output["info_mask"][i], dtype=final_output["info_mask"][i].dtype)
                scores[i] = []
        
        print("exceed_length_mask.tolist(): ", exceed_length_mask.tolist())
        final_output = DataProto.from_dict(final_output, non_tensors={"scores": scores})
        final_output.meta_info.update(meta_info)

        print("======== final_output: ", final_output)
        
        return final_output

    def execute_predictions(self, inputs_str: List[str], 
                        responses_str: List[str], 
                        gui_instructions: List[str], 
                        instructions: List[str],
                        scores = None,
                        active_mask=None,
                        step=None,
                        exceed_length_tag=None) -> List[str]:
        """
        Execute predictions across multiple environments.
        NOTE: the function is the actual `step` function in the environment
        NOTE penalty_for_invalid is not included in observation shown to the LLM
            
        Returns:
            List of observation strings
        """
        next_obs, dones = [], []
        
        results = get_feedback(inputs_str, responses_str, gui_instructions, instructions, self.config.vlm_model)

        for i, (result, active) in enumerate(zip(results, active_mask)):
            if not active:
                next_obs.append('')
                dones.append(1)
            else:
                if result["is_finished"] or exceed_length_tag[i]:
                    next_obs.append('')
                    dones.append(1)
                else:
                    append_obs_str = f'\n<|im_start|>user\n{result["feedback_str"]}<|im_end|>'
                    next_obs.append(append_obs_str)
                    dones.append(0)

                if result["is_webvoyager"]:
                    # if current step is webvoyager, then place the webvoyager score in the last position
                    scores[i][-1]["webvoyager_score"] = result["webvoyager_grade"]
                    if not result["is_good_format"]:
                        new_score = copy.deepcopy(scores[i][-1])
                        new_score["is_good_format"] = False
                        scores[i].append(new_score)
                elif not result["is_finished"]:
                    scores[i].append({"webvoyager_score": result["webvoyager_grade"], "screenshot_score": result["screenshot_grade"], "position": -1, "is_good_format": result["is_good_format"]})
                else:
                    # if current step is finish step
                    if not result["is_good_format"]:
                        new_score = copy.deepcopy(scores[i][-1])
                        new_score["is_good_format"] = False
                        scores[i].append(new_score)

        from .feedback.timestamp import current_timestamp
        import json
        output_dir = f"/mnt/cache/luzimu/code_agent/WebGen-RL/log/{self.config.project_name}/{self.config.experiment_name}/debug_rollout_{self.dir_timestamp}"
        # output_dir = f"/mnt/cache/luzimu/code_agent/WebGen-RL/log/debug_rollout_training"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(os.path.join(output_dir, f"execute_pred_{step}_{current_timestamp()}.json"), "w", encoding="utf-8") as f:
            json.dump({"inputs_str": inputs_str, "responses_str": responses_str, "results": results}, f, ensure_ascii=False)
           
        return next_obs, dones, scores