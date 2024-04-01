import math
import copy
import os.path
from abc import ABC
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

import ray
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from dataclasses import dataclass
from openrlhf.models import Actor, GPTLMLoss, PolicyLoss, SwitchBalancingLoss, ValueLoss
from openrlhf.models.utils import masked_mean
from openrlhf.utils.deepspeed import RunningMoments

def compute_reward(
    r: Union[torch.Tensor, float],
    kl_coef: float,
    log_probs: torch.Tensor,
    log_probs_base: torch.Tensor,
    action_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if kl_coef <= 0.0:
        kl_coef = 0.0
    response_log_pred = torch.sum(torch.mul(log_probs,action_mask),dim=1)
    base_log_pred = torch.sum(torch.mul(log_probs_base,action_mask),dim=1)
    kl = response_log_pred - base_log_pred
    r = (r - kl_coef * kl).clamp(min=-10, max=10)
    return r.detach(), kl  

@dataclass
class Experience:
    sequences: torch.Tensor
    action_log_probs: torch.Tensor
    r: torch.Tensor  # rl reward
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    base_action_log_probs: Optional[torch.Tensor]
    info: Optional[dict]

    @torch.no_grad()
    def to_device(self, device: torch.device) -> None:
        self.sequences = self.sequences.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.r = self.r.to(device)
        if self.attention_mask is not None:
            self.attention_mask = self.attention_mask.to(device)
        if self.action_mask is not None:
            self.action_mask = self.action_mask.to(device)

    def pin_memory(self):
        self.sequences = self.sequences.pin_memory()
        self.action_log_probs = self.action_log_probs.pin_memory()
        self.r = self.r.pin_memory()
        if self.attention_mask is not None:
            self.attention_mask = self.attention_mask.pin_memory()
        if self.action_mask is not None:
            self.action_mask = self.action_mask.pin_memory()
        return self
    

class NaiveExperienceMaker(ABC):
    """
    Naive experience maker.
    """

    def __init__(
        self,
        actor: Actor,
        reward_model: nn.Module,
        initial_model: Actor,
        tokenizer,
        prompt_max_len: int,
        kl_controller,
        strategy=None,
        reward_fn=None,
    ) -> None:
        super().__init__()
        self.actor = actor
        self.reward_model = reward_model
        self.initial_model = initial_model
        self.tokenizer = tokenizer
        self.prompt_max_len = prompt_max_len
        self.kl_ctl = kl_controller
        self.strategy = strategy
        self.reward_fn = reward_fn
        # self.ori_running_moments = RunningMoments()
        self.running_moments = RunningMoments()

    # tokenizer
    def tokenize_fn(self, texts, max_length, device):
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            max_length=max_length,
            padding=True,
            truncation=True,
        )
        return {k: v.to(device) for k, v in batch.items()}

    @torch.no_grad()
    def make_experience(
        self, 
        prompts: Union[str, List[str]], 
        responses: Union[str, List[str]]=None, 
        baselines: torch.Tensor = None,  
        relative_reward: str = "",
        reward_coff: float = 0.5,
        rollout_repeat: int = 1,
        obj_with_kl: bool = False,
        **generate_kwargs
    ) -> List[Experience]:
        self.actor.eval()
        self.initial_model.eval()
        if self.reward_model:
            self.reward_model.eval()

        snapshots = []
        ret = []
        inputs = self.tokenize_fn(prompts, self.prompt_max_len, device="cuda")  # encode prompts
        if relative_reward == "v3":
            greedy_generate_kwargs = copy.deepcopy(generate_kwargs)
            greedy_generate_kwargs["num_beams"] = 1
            greedy_generate_kwargs["do_sample"] = False
            sequences, _, _ = self.actor.generate(**inputs, **greedy_generate_kwargs) # generation by argmax sampling
            preds = self.tokenizer.batch_decode(sequences, skip_special_tokens=True)
            r_greedy, status = zip(*self.reward_fn(prompts, preds, responses))
            r_greedy = torch.tensor(r_greedy, dtype=torch.float) 

        for _ in range(rollout_repeat):
            sequences, attention_mask, action_mask = self.actor.generate(**inputs, **generate_kwargs)
            num_actions = action_mask.size(1)
            # log probs
            action_log_probs = self.actor(sequences, num_actions, attention_mask)
            # init log probs
            base_action_log_probs = self.initial_model(sequences, num_actions, attention_mask)

            # rewards
            if self.reward_model:
                r = self.reward_model(sequences, attention_mask)
            else:
                preds = self.tokenizer.batch_decode(sequences, skip_special_tokens=True)
                r, status = zip(*self.reward_fn(prompts, preds, responses))
                r = torch.tensor(r, dtype=torch.float, device='cuda')
                if relative_reward == "v1":  # offline fixed baseline from SFT model
                    r = r - reward_coff*baselines.to(r.device)
                elif relative_reward == "v2":  # the dynamic of true reward
                    r = r - self.running_moments.mean
                    self.running_moments.update(r) 
                elif relative_reward == "v3":  # Remax
                    r = r - r_greedy.to(r.device)
            snapshots.append([sequences, r, status, action_log_probs, base_action_log_probs, attention_mask, action_mask])

        if relative_reward == "v4":  # RLOO
            assert rollout_repeat>1, "rollout_repeat should be greater than 1"
            prompts_size = len(prompts)
            for i in range(prompts_size):
                group_r = torch.stack([s[1][i] for s in snapshots]).unsqueeze(0) 
                group_r = group_r.repeat(rollout_repeat, 1)  
                mask = torch.ones_like(group_r).to(group_r.device)  
                mask.diagonal().fill_(0)
                group_baseline = torch.sum(group_r*mask,dim=1)/(rollout_repeat-1)  
                for idx,s in enumerate(snapshots):
                    s[1][i] -= group_baseline[idx]

 
        for i in range(rollout_repeat):
            sequences, r, status, action_log_probs, base_action_log_probs, attention_mask, action_mask = snapshots[i]
            reward, kl = compute_reward(  
                r,
                self.kl_ctl.value,
                action_log_probs,  
                base_action_log_probs,
                action_mask=action_mask,
            )
            # info: (B,) the info of each sequence
            info = {
                "kl":kl/action_mask.float().sum(dim=-1),
                "ppl": masked_mean(-action_log_probs, action_mask, dim=-1),
                "reward": r,  # true reward
                "returns": reward,  # rl reward
                "reward_status": status,
                "response_length": action_mask.float().sum(dim=-1),
                "total_length": attention_mask.float().sum(dim=-1),
            }
            ret.append(Experience(
                sequences,
                action_log_probs,
                r if obj_with_kl else reward,  # if obj with kl, rl reward equals to true reward
                attention_mask,
                action_mask,
                base_action_log_probs,
                info,
            ))
        # reset model state
        self.actor.train()
        return ret

    
