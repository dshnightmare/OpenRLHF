import re
import math, random, requests, concurrent
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
    # 'gsm8k': re.compile(r"The answer is: (\-?[0-9\.\,]+)"),
    'gsm8k': re.compile(r"#### (\-?[0-9\.\,]+)"),
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

def reward_gsm8k(prompts, pred_lines, gold_lines):
    ret = []
    for gold_line, pred_line in zip(gold_lines, pred_lines):
        pred_ans = extract_answer(pred_line, pred_ans_re_mapper['gsm8k'])
        gold_ans = extract_answer(gold_line, gold_ans_re_mapper['gsm8k'])
        assert gold_ans != INVALID_ANS, gold_ans
        if pred_ans == INVALID_ANS:
            ret.append((-0.1, gsm8kResult.BADFORMAT))
        elif compare_answer_fn_mapper['gsm8k'](float(pred_ans), float(gold_ans)):
            ret.append((1., gsm8kResult.CORRECT))
        else:
            ret.append((0., gsm8kResult.INCORRECT))
    return ret

_ip="10.202.196.174"
extract_endpoints = [f"http://{_ip}:100{i}/generate" for i in range(8)]
match_endpoints = [f"http://{_ip}:200{i}/generate" for i in range(8)]
def reward_tal(prompts, pred_lines, gold_lines):
    hits = [{"problem": problem, "response": response, "ref_answer": ref_answer} for problem, response, ref_answer in zip(prompts, pred_lines, gold_lines)]
    def format_prompt(hit, run_mode, model_type: str = "qwen"):
        BAICHUAN2_USER_TOKEN = '<reserved_106>'
        BAICHUAN2_ASSISTANT_TOKEN = '<reserved_107>'
        QWEN_IM_START = '<|im_start|>'
        QWEN_IM_END = '<|im_end|>'
        QWEN_IM_START_ID = 151644
        QWEN_IM_END_ID = 151645
        QWEN_SUCCESS = 'Y'
        QWEN_SUCCESS_ID = 56
        QWEN_FAIL = 'N'
        QWEN_FAIL_ID = 45
        if run_mode == "match":
            problem = hit["_source"].get("content", "") or ""
            ref_answer = hit["_source"].get("answer", "") or ""
            gen_answer = hit.get("extract", "") or ""
            prompt_template = "下面有一道数学题和两份答案，判断这两份答案是否完全匹配。\n\n### 题目：\n{question}### 答案一：\n{answer1}### 答案二：\n{answer2}\n\n### 回答："
            input_ = prompt_template.format(question=problem, answer1=ref_answer, answer2=gen_answer)
        elif run_mode == "extract":
            problem = hit["_source"].get("content", "") or ""
            response = hit.get("response", "") or ""
            prompt_template = "下面给出一道数学题以及对应的解答过程，请你简化解答过程，将计算得到的最终答案抽取出来。\n\n### 题目：\n{question}### 解答过程：\n{response}\n\n### 答案："
            input_ = prompt_template.format(question=problem, response=response)
        else:
            raise NotImplementedError
        if model_type == "baichuan2":
            prompt = f'{BAICHUAN2_USER_TOKEN}{input_}{BAICHUAN2_ASSISTANT_TOKEN}'
        elif model_type == "qwen":
            prompt = f"{QWEN_IM_START}system\n{QWEN_IM_END}\n{QWEN_IM_START}user\n{input_}{QWEN_IM_END}\n{QWEN_IM_START}assistant\n"
        else:
            raise NotImplementedError
        return prompt
    
    def do_call(idx, prompt, run_mode):
        endpoint = random.choice({"extract": extract_endpoints, "match": match_endpoints}[run_mode])
        headers = {"Content-Type": "application/json"}
        kws = dict(
            prompt=prompt,
            top_p=1.0, top_k=-1, temperature=0, max_tokens=2048,
            n=1, presence_penalty=0.0, frequency_penalty=0.0, logprobs=5,
            # stop=[QWEN_IM_END, QWEN_IM_START],
            # stop_token_ids=[QWEN_IM_END_ID, QWEN_IM_START_ID],
        )
        retry = 1
        ret = None
        while retry > 0:
            try:
                ret = requests.post(endpoint, json=kws, headers=headers).json()
                break
            except:
                retry -= 1
        assert ret is not None
        x = ret["text"][0][len(prompt):]
        return idx, x

    for run_mode in ["extract", "match"]:
        prompts = []
        for idx, hit in enumerate(hits):
            ref_answer = hit["ref_answer"]
            obj = {
                "_id": hit.get("_id"),
                "_source": {"content": hit["problem"], "answer": ref_answer},
                "response": hit["response"],
                "extract": hit.get("extract_response")
            }
            prompt = format_prompt(obj, run_mode)
            prompts.append((idx, prompt))

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            futures = [executor.submit(do_call, idx, prompt, run_mode) for idx, prompt in enumerate(prompts)]
            futures = concurrent.futures.as_completed(futures)
            for future in futures:
                idx, output = future.result()
                hits[idx][f"{run_mode}_response"] = output
    ret = []
    for hit in hits:
        if hit["extract_response"] == "无效答案":
            ret.append((-0.1, gsm8kResult.BADFORMAT))
        elif hit["match_response"] == "Y":
            ret.append((1., gsm8kResult.CORRECT))
        else:
            ret.append((0., gsm8kResult.INCORRECT))
    return ret
    

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