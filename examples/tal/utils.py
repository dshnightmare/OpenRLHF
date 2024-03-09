import re
import math
from enum import Enum
from functools import partial
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

INVALID_ANS = "[invalid]"


class gsm8kResult(Enum):
    CORRECT = 0
    INCORRECT = 1
    BADFORMAT = 2

compare_answer_fn_mapper = {
    'gsm8k': lambda pred_ans, gold_ans: abs(pred_ans - gold_ans) <= 1e-4,
    # 'svamp': lambda pred_ans, gold_ans: abs(pred_ans - gold_ans) <= 1e-4,
    # 'mathqa': lambda pred_ans, gold_ans: pred_ans == gold_ans,
}


gold_ans_re_mapper = {
    'gsm8k': re.compile(r"#### (\-?[0-9\.\,]+)"),
}

pred_ans_re_mapper = {
    'gsm8k': re.compile(r"answer is <(\-?[0-9\.\,]+)>"),
}

def extract_answer(completion, re):
    if completion.find('\u0000') >= 0:
        completion = completion[0:completion.find('\u0000')]
    match = re.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        try:
            float(match_str)
        except BaseException:
            return INVALID_ANS
        return match_str
    else:
        return INVALID_ANS

def reward_gsm8k(pred_line, gold_line):
    pred_ans = extract_answer(pred_line, pred_ans_re_mapper['gsm8k'])
    gold_ans = extract_answer(gold_line, gold_ans_re_mapper['gsm8k'])
    assert gold_ans != INVALID_ANS, gold_ans
    if pred_ans == INVALID_ANS:
        return -0.1, gsm8kResult.BADFORMAT
    elif compare_answer_fn_mapper['gsm8k'](float(pred_ans), float(gold_ans)):
        return 1., gsm8kResult.CORRECT
    else:
        return 0., gsm8kResult.INCORRECT
    

def get_scheduler(
    name,
    optimizer: Optimizer,
    num_warmup_steps,
    num_training_steps,
):
    if name == 'constant_with_warmup':
        return get_constant_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps)

    if name == 'cosine':
        return get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

    raise NotImplementedError



def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: float
):
    if current_step < num_warmup_steps:
        return 0.1 + 0.9 * float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return 0.1 + 0.9 * max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))


def get_cosine_schedule_with_warmup(optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1):
    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def _get_constant_schedule_with_warmup_lr_lambda(current_step: int, *, num_warmup_steps: int):
    if current_step < num_warmup_steps:
        return 0.1 + 0.9 * float(current_step) / float(max(1.0, num_warmup_steps))
    return 1.0


def get_constant_schedule_with_warmup(optimizer: Optimizer, num_warmup_steps: int, last_epoch: int = -1):
    lr_lambda = partial(
        _get_constant_schedule_with_warmup_lr_lambda, 
        num_warmup_steps=num_warmup_steps
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)