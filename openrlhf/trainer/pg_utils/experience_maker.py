from abc import ABC
from typing import List, Optional, Union, Tuple
import torch
import torch.nn as nn
from dataclasses import dataclass
from openrlhf.models import Actor
from openrlhf.models.utils import masked_mean
from openrlhf.utils.deepspeed import RunningMoments

def compute_reward(
    r: Union[torch.Tensor, float],
    kl_coef: float,
    log_probs: torch.Tensor,
    log_probs_base: torch.Tensor,
    action_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if any(kl_coef <= 0.0):
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
    returns: torch.Tensor  # rl reward
    baselines: torch.Tensor
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    base_action_log_probs: Optional[torch.Tensor]
    info: Optional[dict]

    @torch.no_grad()
    def to_device(self, device: torch.device) -> None:
        self.sequences = self.sequences.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.returns = self.returns.to(device)
        self.baselines = self.baselines.to(device)
        if self.attention_mask is not None:
            self.attention_mask = self.attention_mask.to(device)
        if self.action_mask is not None:
            self.action_mask = self.action_mask.to(device)

    def pin_memory(self):
        self.sequences = self.sequences.pin_memory()
        self.action_log_probs = self.action_log_probs.pin_memory()
        self.returns = self.returns.pin_memory()
        self.baselines = self.baselines.pin_memory()
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
        relative_reward: torch.Tensor = None,  
        relative_reward_type: str = "",
        reward_coff: float = 0.5,
        baseline_type: str = "",
        prompts_difficulity: torch.Tensor = None,
        rollout_repeat: int = 1,
        objective_with_kl: bool = False,
        **generate_kwargs
    ) -> List[Experience]:
        self.actor.eval()
        self.initial_model.eval()
        if self.reward_model:
            self.reward_model.eval()

        snapshots = []
        ret = []
        inputs = self.tokenize_fn(prompts, self.prompt_max_len, device="cuda")  # encode prompts
        for _ in range(rollout_repeat):
            sequences, attention_mask, action_mask = self.actor.generate(**inputs, **generate_kwargs)
            num_actions = action_mask.size(1)
            # log probs
            action_log_probs = self.actor(sequences, num_actions, attention_mask)
            # init log probs
            base_action_log_probs = self.initial_model(sequences, num_actions, attention_mask)
            # rewards
            if self.reward_model:
                true_r = self.reward_model(sequences, attention_mask)
            else:
                preds = self.tokenizer.batch_decode(sequences, skip_special_tokens=True)
                true_r, status = zip(*self.reward_fn(prompts, preds, responses))
                true_r = torch.tensor(true_r, dtype=torch.float, device='cuda')
                if relative_reward_type == "v1":  # offline fixed baseline from SFT model
                    true_r = true_r - reward_coff*relative_reward.to(true_r.device)
                if relative_reward_type == "v2":  # the dynamic of true reward
                    self.running_moments.update(true_r) 
                    relative_reward = torch.ones_like(true_r,device=true_r.device)*self.running_moments.mean
                    true_r = true_r - reward_coff*relative_reward
            if self.strategy.args.use_dynamic_kl:
                kl_coff  = 0.01 + 0.04 * prompts_difficulity  # (base acc=0, kl coff=0.01) -> (base acc=1, kl coff=0.05)  linear increase
                kl_coff = kl_coff.to(true_r.device)
            else:
                kl_coff = self.kl_ctl.value
            reward, kl = compute_reward(true_r,kl_coff,action_log_probs,  base_action_log_probs,action_mask=action_mask)
            snapshots.append([sequences, true_r, reward, status, action_log_probs, base_action_log_probs, attention_mask, action_mask])

        baselines = []
        if baseline_type == "rloo":  # RLOO
            assert rollout_repeat>1, "rollout_repeat should be greater than 1"
            group_baselines = []
            prompts_size = len(prompts)
            for i in range(prompts_size):
                group_reward = torch.stack([s[2][i] for s in snapshots]).unsqueeze(0) 
                group_reward = group_reward.repeat(rollout_repeat, 1)  
                mask = torch.ones_like(group_reward).to(group_reward.device)  
                mask.diagonal().fill_(0)
                group_baseline = torch.sum(group_reward*mask,dim=1)/(rollout_repeat-1)  
                group_baselines.append(group_baseline)
            for i in range(rollout_repeat):
                baselines.append(torch.stack([group_baselines[j][i] for j in range(prompts_size)]))
 
        for i in range(rollout_repeat):
            sequences, true_r, reward, status, action_log_probs, base_action_log_probs, attention_mask, action_mask = snapshots[i]
            baseline = baselines[i] if baselines else torch.zeros_like(reward)
            # info: (B,) the info of each sequence
            info = {
                "kl":kl/action_mask.float().sum(dim=-1),
                "ppl": masked_mean(-action_log_probs, action_mask, dim=-1),
                "returns": reward,  # reward has kl penalty
                "true_reward": true_r,  # true reward
                "reward_status": status,
                "response_length": action_mask.float().sum(dim=-1),
                "total_length": attention_mask.float().sum(dim=-1),
            }
            ret.append(Experience(
                sequences,
                action_log_probs,
                true_r if objective_with_kl else reward,  # if objective with kl, rl reward equals to true reward
                baseline,
                attention_mask,
                action_mask,
                base_action_log_probs,
                info,
            ))
        # reset model state
        self.actor.train()
        return ret

    
