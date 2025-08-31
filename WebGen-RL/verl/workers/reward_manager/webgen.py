
import os 
from verl import DataProto
import torch
import statistics
# from verl.utils.reward_score.synsql_verifier import sql_compute_score

class WebGenRewardManager:
    """The WebGen reward manager.
    """

    def __init__(self, tokenizer, num_examine, config, compute_score) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  
        self.compute_score = compute_score
        self.config = config

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""
        reward_metrics={}
        scores = data.non_tensor_batch["scores"]

        total = 0
        num = 0
        for i, score in enumerate(scores):
            for step_score in score:
                if not step_score.get("is_good_format", True):
                    step_score["reward"] = -1
                else:
                    step_score["reward"] = step_score["screenshot_score"] + step_score["webvoyager_score"]
                position = step_score["position"]
                total += step_score["reward"]
                num += 1

        reward_metrics = {"all": total / num}

        score_dict = {"all": scores}
        
        return score_dict, reward_metrics


class WebGenRewardManagerScreenshot:
    """The WebGen reward manager.
    """

    def __init__(self, tokenizer, num_examine, config, compute_score) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  
        self.compute_score = compute_score
        self.config = config

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""
        reward_metrics={}
        scores = data.non_tensor_batch["scores"]

        total = 0
        num = 0
        for i, score in enumerate(scores):
            for step_score in score:
                if not step_score.get("is_good_format", True):
                    step_score["reward"] = -1
                else:
                    step_score["reward"] = step_score["screenshot_score"]
                position = step_score["position"]
                total += step_score["reward"]
                num += 1

        reward_metrics = {"all": total / num}

        score_dict = {"all": scores}
        
        return score_dict, reward_metrics


class WebGenRewardManagerGUIAgent:
    """The WebGen reward manager.
    """

    def __init__(self, tokenizer, num_examine, config, compute_score) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  
        self.compute_score = compute_score
        self.config = config

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""
        reward_metrics={}
        scores = data.non_tensor_batch["scores"]

        total = 0
        num = 0
        for i, score in enumerate(scores):
            for step_score in score:
                if not step_score.get("is_good_format", True):
                    step_score["reward"] = -1
                else:
                    step_score["reward"] = step_score["webvoyager_score"]
                position = step_score["position"]
                total += step_score["reward"]
                num += 1

        reward_metrics = {"all": total / num}

        score_dict = {"all": scores}
        
        return score_dict, reward_metrics
