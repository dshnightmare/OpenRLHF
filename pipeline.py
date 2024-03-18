#!/usr/bin/env python
# -*- coding: utf-8 -*-

import concurrent.futures
import json
import os
import random
import sys
import traceback
from typing import Optional

import fire
import ipdb
from tqdm import tqdm


def answer_extract_match_throughput_benchmark(
        input_file: str, ip: str, run_mode: str, output_file: Optional[str] = None, model_type: str = "qwen",
        problem_field: str = "prompt", answer_field: str = "answer",
        max_workers: int = 64,
):

    import openai
    import requests

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

    def format_prompt(hit):
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

    prompts = []
    hits = [json.loads(_) for _ in open(input_file)]
    for idx, hit in enumerate(hits):
        ref_answer = hit.get(answer_field, "") or ""
        obj = {
            "_id": hit.get("_id"),
            "_source": {"content": hit[problem_field], "answer": ref_answer},
            "response": hit["response"],
            "extract": hit.get("extract_response")
        }
        prompt = format_prompt(obj)
        prompts.append(prompt)

    def do_call(idx, prompt):
        api = f"http://{ip}:100{random.randint(0, 7)}/generate"
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
                ret = requests.post(api, json=kws, headers=headers).json()
                break
            except:
                traceback.print_exc()
                retry -= 1
        assert ret is not None
        # ipdb.set_trace()
        x = ret["text"][0][len(prompt):]
        return idx, x

    # f"{system_mesage}\nHuman:{problem}\n\nAssistant:"

    def do_call_openai(idx, prompt):
        client = openai.OpenAI(api_key="EMPTY", base_url=f"http://{ip}:100{random.randint(0, 7)}/v1")
        # openai.api_key = "EMPTY"
        # openai.api_base = f"http://{ip}:100{random.randint(0, 7)}/v1"
        completion = client.completions.create(
            model="/mnt/pfs/jinfeng_team/SFT/wanqian/yq9/models/math-policy/qwen_14b_answer_extract_gpt4_clean_english_4096_0116",
            prompt=prompt, temperature=0, # logprobs=5,  # stream=False, echo=False,
        )
        # client.chat.completions.create()
        ipdb.set_trace()

    # print(do_call_openai(0, prompts[0]))
    # return

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(do_call, idx, prompt) for idx, prompt in enumerate(prompts)]
        futures = tqdm(concurrent.futures.as_completed(futures), total=len(futures))
        for future in futures:
            idx, output = future.result()
            assert hits[idx][f"{run_mode}_response"] == output

    if output_file:
        with open(output_file, "w") as fp:
            for hit in hits:
                fp.write(json.dumps(hit, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    fire.Fire()
