import copy
import os.path
from abc import ABC
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from openrlhf.models import Actor, GPTLMLoss
from openrlhf.models.utils import masked_mean
from .pg_utils import Experience, NaiveExperienceMaker,NaiveReplayBuffer, AdaptiveKLController, FixedKLController


def compute_log_probs(log_probs,mask):
    mask_log_probs = torch.mul(log_probs,mask)  # (B,A)
    seq_log_probs = torch.sum(mask_log_probs,dim=1)  # (B,)
    return seq_log_probs

class PGLoss(nn.Module):
    def forward(self,
                log_probs: torch.Tensor,
                base_log_probs: torch.Tensor,
                returns: torch.Tensor,
                baselines: torch.Tensor,
                action_mask: Optional[torch.Tensor] = None,
                objective_with_kl: bool = False,
                beta: float = 0.04,
                return_info: bool = False):
        """
        log_probs: (B, A)
        base_log_probs: (B, A)
        returns: (B)
        baselines: (B)
        action_mask: (B, A)
        """
        response_log_probs = compute_log_probs(log_probs,action_mask)  # (B,)
        pg_objective = torch.mean(response_log_probs*(returns-baselines))
        if objective_with_kl:
            # estimate kl divergence of pi and base
            token_kl = base_log_probs.exp()/log_probs.exp()-base_log_probs+log_probs-1
            mean_kl = masked_mean(token_kl,action_mask)
            loss = -(pg_objective - beta*mean_kl)
            info = {'kl_loss':mean_kl}
        else:
            loss = -pg_objective
            info = {}
        if return_info:
            return loss,info
        else:
            return loss

class PGTrainer(ABC):
    """
        Trainer for PG algorithm.

    Args:
        strategy (Strategy): the strategy to use for training
        actor (Actor): the actor model in pg algorithm
        reward_model (nn.Module): the reward model in rlhf algorithm to make reward of sentences
        initial_model (Actor): the initial model in rlhf algorithm to generate reference logits to limit the update of actor
        actor_optim (Optimizer): the optimizer to use for actor model
        kl_coef (float, defaults to 0.1): the coefficient of kl divergence loss
        train_batch_size (int, defaults to 8): the batch size to use for training
        buffer_limit (int, defaults to 0): the max_size limitaiton of replay buffer
        buffer_cpu_offload (bool, defaults to True): whether to offload replay buffer to cpu
        rollout_repeat (int, default to 1)
        relative_reward_type (str, default to "") : used to reshape the true reward
        baseline_type (str, default to ""): the type of baseline to use
        experience_batch_size (int, defaults to 8): the batch size to use for experience generation
        max_epochs (int, defaults to 1): the number of epochs of training process
        tokenier (Callable, optional): the tokenizer to use for tokenizing the input
        sample_replay_buffer (bool, defaults to False): whether to sample from replay buffer
        dataloader_pin_memory (bool, defaults to True): whether to pin memory for data loader
        callbacks (List[Callback], defaults to []): the callbacks to call during training process
        generate_kwargs (dict, optional): the kwargs to use while model generating
    """

    def __init__(
        self,
        strategy,
        actor: Actor,
        reward_model: nn.Module,
        initial_model: Actor,
        ema_model: Actor,
        actor_optim: Optimizer,
        actor_scheduler,
        ema_beta: float = 0.992,
        ptx_coef: float = 0,
        micro_train_batch_size: int = 8,
        buffer_limit: int = 0,
        buffer_cpu_offload: bool = True,
        micro_rollout_batch_size: int = 8,
        rollout_repeat: int = 1,
        relative_reward_type: str = "",
        baseline_type: str = "",
        gradient_checkpointing: bool = False,
        max_epochs: int = 1,
        max_norm: float = 1.0,
        tokenizer: Optional[Callable[[Any], dict]] = None,
        init_kl_coef: float = 0.001,
        kl_target: float = None,
        kl_horizon: int = 10000,
        prompt_max_len: int = 128,
        dataloader_pin_memory: bool = True,
        reward_fn: Callable[[List[torch.Tensor]], torch.Tensor] = None,
        **generate_kwargs,
    ) -> None:
        assert (
            not isinstance(reward_model, List) or len(reward_model) == 1 or reward_fn is not None
        ), "reward_fn must be specified if using multiple reward models"

        super().__init__()
        self.strategy = strategy
        self.args = strategy.args
        self.micro_rollout_batch_size = micro_rollout_batch_size
        self.max_epochs = max_epochs
        self.tokenizer = tokenizer
        self.generate_kwargs = generate_kwargs
        self.dataloader_pin_memory = dataloader_pin_memory
        self.max_norm = max_norm
        self.ptx_coef = ptx_coef
        self.micro_train_batch_size = micro_train_batch_size
        self.rollout_repeat = rollout_repeat
        self.relative_reward_type = relative_reward_type
        self.baseline_type = baseline_type
        self.prompt_max_len = prompt_max_len
        self.ema_beta = ema_beta
        self.gradient_checkpointing = gradient_checkpointing
        self.reward_fn = reward_fn
        self.kl_target = kl_target
        self.actor = actor
        self.reward_model = reward_model
        self.initial_model = initial_model
        self.ema_model = ema_model
        self.actor_optim = actor_optim
        self.actor_scheduler = actor_scheduler
        self.actor_loss_fn = PGLoss()
        self.ptx_loss_fn = GPTLMLoss()

        self.objective_with_kl = self.args.objective_with_kl
        self.beta = self.args.beta  # the coff of kl part in loss

        # # Mixtral 8x7b
        self.aux_loss = self.args.aux_loss_coef > 1e-8
        
        if self.kl_target:
            self.kl_ctl = AdaptiveKLController(init_kl_coef, kl_target, kl_horizon)
        else:
            self.kl_ctl = FixedKLController(init_kl_coef)

        self.experience_maker = NaiveExperienceMaker(  
            actor, reward_model, initial_model, tokenizer, prompt_max_len, self.kl_ctl, strategy, reward_fn
        )
        self.replay_buffer = NaiveReplayBuffer(micro_train_batch_size, buffer_limit, buffer_cpu_offload)

        # for eval
        self.best_score = -10000

        self._wandb = None
        if self.strategy.args.use_wandb and self.strategy.is_rank_0():
            import wandb

            self._wandb = wandb
            if not wandb.api.api_key:
                wandb.login(key=strategy.args.use_wandb)
            wandb.init(
                entity=strategy.args.wandb_org,
                project=strategy.args.wandb_project,
                group=strategy.args.wandb_group,
                name=strategy.args.wandb_run_name,
                config=strategy.args.__dict__,
                reinit=True,
            )

            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
            wandb.define_metric("eval/global_step")
            wandb.define_metric("eval/*", step_metric="eval/global_step", step_sync=True)

    def fit(
        self,
        prompts_dataloader,
        pretrain_dataloader,
        eval_dataloader,
        args,
    ) -> None:
        self.prompts_dataloader = prompts_dataloader
        self.pretrain_dataloader = pretrain_dataloader
        self.eval_dataloader = eval_dataloader

        update_timesteps = args.rollout_batch_size // (self.strategy.world_size * self.micro_rollout_batch_size * self.rollout_repeat)
        assert update_timesteps >= 1
        global_step = 1

        # get eval and save steps
        if args.eval_steps == -1:
            args.eval_steps = prompts_dataloader.__len__() // update_timesteps  # Evaluate once per epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")  # do not save ckpt

        for episode in range(args.num_episodes):
            if isinstance(self.prompts_dataloader.sampler, DistributedSampler):
                self.prompts_dataloader.sampler.set_epoch(episode)
            pbar = tqdm(
                range(self.prompts_dataloader.__len__()),
                desc=f"Episode [{episode + 1}/{args.num_episodes}]",
                disable=not self.strategy.is_rank_0(),
            )

            for rand_prompts in self.prompts_dataloader:
                if isinstance(rand_prompts[0], str):
                    prompts, responses = rand_prompts, None
                else:
                    prompts, responses, expand_keys_data = rand_prompts
                relative_reward = expand_keys_data['relative_key'] if "relative_key" in expand_keys_data else None
                prompts_difficulity = expand_keys_data['difficulty_key'] if "difficulty_key" in expand_keys_data else None
                list_experience = self.experience_maker.make_experience(prompts, responses,relative_reward, 
                                                                        self.relative_reward_type, args.reward_coff,self.baseline_type,
                                                                        prompts_difficulity,self.rollout_repeat,self.objective_with_kl, 
                                                                        **self.generate_kwargs)
                for e in list_experience:
                    self.replay_buffer.append(e)
                
                # generate_num = micro_rollout_batch_size * rollout_repeat samples
                if global_step % update_timesteps == 0:
                    # print prompt/answer in each update step
                    status ={}
                    experience = list_experience[0]
                    total_num = experience.sequences.size(0)
                    for attr in type(experience.info['reward_status'][0]).__members__:  
                        select = [s.name == attr for s in experience.info['reward_status']]
                        output = self.tokenizer.batch_decode(experience.sequences[select], skip_special_tokens=True)
                        if output:  # output one case for each attr
                            self.strategy.print(attr)
                            self.strategy.print(output[0])
                        status[attr] = len(output) / total_num  # calculate the CORRECT/INCORRECT/BADFORMAT rate in replay_buffer
                    status = self.strategy.all_reduce(status)  # reduce the information from all ranks 

                    torch.cuda.empty_cache()
                    if args.normalize_reward:
                        self.replay_buffer.normalize("returns", self.strategy)
                    status.update(self.pg_train())
                    self.replay_buffer.clear()
                    torch.cuda.empty_cache()
                    self.kl_ctl.update(status["kl"], args.rollout_batch_size)
                    # logs/checkpoints
                    self.save_logs_and_checkpoints(args, global_step // update_timesteps, pbar, status)

                pbar.update()
                global_step = global_step + 1

    def pg_train(self):
        # replay buffer may be empty at first, we should rebuild at each training
        dataloader = DataLoader(
            self.replay_buffer,
            batch_size=self.replay_buffer.sample_batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=self.dataloader_pin_memory,
            collate_fn=self.replay_buffer.collate_fn,
        )
        device = torch.cuda.current_device()

        status_list = []
        status_mean = {}
        for epoch in range(self.max_epochs):  # must be 1
            pbar = tqdm(
                dataloader,
                desc=f"Train epoch [{epoch + 1}/{self.max_epochs}]",
                disable=not self.strategy.is_rank_0(),
            )
            for experience in pbar:
                experience.to_device(device)
                status = self.training_step(experience)

                # for DP
                # weighted mean for kl, ppl
                status["kl"] *= status["response_length"]  # sequence kl
                status["ppl"] *= status["response_length"] # sequence ppl
                status = self.strategy.all_reduce(status)
                status["kl"] /= status["response_length"]  # average kl on action
                status["ppl"] /= status["response_length"] # average ppl on action

                status_list.append(status)
                short_status = {
                    "pg": status["actor/pg_loss"],
                    "rm": status["true_reward"],
                    "ret": status["returns"],
                    "glen": status["response_length"],
                    "tlen": status["total_length"],
                    "kl": status["kl"],
                    "ppl": status["ppl"],
                }
                if "ptx_loss" in status:
                    short_status["ptx"] = status["actor/ptx_loss"]
                pbar.set_postfix(short_status)

        if status_list:
            status_mean = status_list[0]
            for m in status_list[1:]:
                for k, v in m.items():
                    status_mean[k] += v
            for k in status_mean.keys():
                status_mean[k] /= len(status_list)
        return status_mean

    def training_step(self, experience: Experience) -> Dict[str, float]:
        status = self.training_step_actor(experience)
        return status

    def training_step_actor(self, experience: Experience) -> Dict[str, float]:
        self.actor.train()
        num_actions = experience.action_mask.size(1)
        # actor loss
        action_log_probs, output = self.actor(
            experience.sequences, num_actions, attention_mask=experience.attention_mask, return_output=True
        )  
        action_probs = F.softmax(output["logits"][:, :-1, :], dim=-1)[:, -num_actions:, :]  # used to calculate entropy


        # loss function
        actor_loss, pg_info = self.actor_loss_fn(
            action_log_probs,
            experience.base_action_log_probs,
            experience.returns,
            experience.baselines,
            action_mask=experience.action_mask,
            objective_with_kl=self.objective_with_kl,
            beta = self.beta,
            return_info=True,
        )
        # mixtral
        if self.aux_loss:
            aux_loss = output.aux_loss
        else:
            aux_loss = 0
        loss = actor_loss + aux_loss * self.args.aux_loss_coef  
        self.strategy.backward(loss, self.actor, self.actor_optim)

        # ptx loss
        if self.pretrain_dataloader is not None:
            data = next(self.pretrain_dataloader)
            inputs = data[1].squeeze(1).to(torch.cuda.current_device())
            attention_mask = data[2].squeeze(1).to(torch.cuda.current_device())
            label = torch.where(
                attention_mask.bool(),
                inputs,
                self.ptx_loss_fn.IGNORE_INDEX,
            )

            output = self.actor(inputs, attention_mask=attention_mask, return_output=True)
            ptx_log_probs = output["logits"]

            # loss function
            ptx_loss = self.ptx_loss_fn(ptx_log_probs, label)
            # mixtral
            if self.aux_loss:
                aux_loss = output.aux_loss
            else:
                aux_loss = 0
            loss = ptx_loss + aux_loss * self.args.aux_loss_coef
            self.strategy.backward(self.ptx_coef * loss, self.actor, self.actor_optim)

        # if not warmup:
        self.strategy.optimizer_step(self.actor_optim, self.actor, self.actor_scheduler, name="actor")
        if self.ema_model:
            self.strategy.moving_average(self.actor, self.ema_model, self.ema_beta, "cpu")

        # status
        entropy = masked_mean(-torch.sum(action_probs * torch.log(action_probs + 1e-9), dim=-1), experience.action_mask)
        status = {
            "actor/pg_loss": actor_loss.item(),
            "actor/lr": self.actor_optim.param_groups[0]['lr'],
            "actor/entropy": entropy.item(),
        }
        if self.pretrain_dataloader is not None:
            status["actor/ptx_loss"] = ptx_loss.item()
        for k, v in experience.info.items():
            if k in ["kl", "ppl"]:
                status[k] = (
                    (v * experience.info["response_length"]).sum() / experience.info["response_length"].sum()
                ).item()
            else:
                status[k] = v.mean().item()
        for k, v in pg_info.items():
            status[k] = (
                (v.to("cpu") * experience.info["response_length"]).sum() / experience.info["response_length"].sum()
            ).item()
        return status

  
    
    def evaluate(self, dataloader):
        # eval
        status = {}
        all_exp = []
        pbar = tqdm(
            self.eval_dataloader,
            desc=f"Eval epoch",
            disable=not self.strategy.is_rank_0(),
        )
        for data in pbar:
            if isinstance(data[0], str):
                prompts, responses = data, None
            else:
                prompts, responses = data
            eval_generate = copy.deepcopy(self.generate_kwargs)
            eval_generate.update({
                'do_sample': False,
                'temperature': 0.0,
                'topp': 1.0,
                'repetition_penalty': 1.0,
                })
            list_experience = self.experience_maker.make_experience(prompts, responses, **eval_generate)
            all_exp.extend(list_experience)
            pbar.update()   
        
        for attr in type(all_exp[0].info['reward_status'][0]).__members__:
            total_num = 0
            status[attr] = 0
            for experience in all_exp:
                total_num += experience.sequences.size(0)
                select = [s.name == attr for s in experience.info['reward_status']]
                status[attr] += sum(select)
            status[attr] /= total_num
        status = self.strategy.all_reduce(status)

        if status['CORRECT'] > self.best_score:
            self.best_score = status['CORRECT']
            # TODO save better model

        status['best_score'] = self.best_score
        return status


    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}):
        if global_step % args.logging_steps == 0:
            # step bar
            step_bar.set_postfix(logs_dict)
            # wandb
            if self._wandb is not None and self.strategy.is_rank_0():
                logs = {
                    "train/%s" % k: v
                    for k, v in {
                        **logs_dict,
                        "global_step": global_step,
                    }.items()
                }
                self._wandb.log(logs)

        # TODO: Add evaluation mechanism for PPO
        if global_step % args.eval_steps == 0:
            eval_logs ={}
            if self.eval_dataloader:
                eval_logs.update(self.evaluate(self.eval_dataloader))
            # wandb
            if self._wandb is not None and self.strategy.is_rank_0():
                logs = {
                    "eval/%s" % k: v
                    for k, v in {
                        **eval_logs,
                        "global_step": global_step,
                    }.items()
                }
                self._wandb.log(logs)
        # save ckpt
        # TODO: save best model on dev, use loss/perplexity/others on whole dev dataset as metric
        if global_step % args.save_steps == 0:
            tag = f"global_step{global_step}"
            self.strategy.save_ckpt(
                self.actor.model, os.path.join(args.ckpt_path, "_actor"), tag, args.max_ckpt_num, args.max_ckpt_mem
            )
            # self.strategy.save_ckpt(
            #     self.critic, os.path.join(args.ckpt_path, "_critic"), tag, args.max_ckpt_num, args.max_ckpt_mem
            # )
