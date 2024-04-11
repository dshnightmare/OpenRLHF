import re
import math, random, requests, concurrent
from enum import Enum
from functools import partial
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
import sympy as sp
from sympy import sympify, simplify, expand, cancel, apart
from sympy.parsing.latex import parse_latex

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
        retry = 5
        ret = None
        while retry > 0:
            try:
                ret = requests.post(endpoint, json=kws, headers=headers).json()
                break
            except:
                retry -= 1
        assert ret is not None, prompt
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
            prompts.append(prompt)

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


def contains_chinese(text):
    return re.search("[\u4e00-\u9fff]", text) is not None

def fix_sympy_expr(expr):
    functions = [
        "sign",
        "sqrt",
        "exp",
        "ln",
        "sin",
        "cos",
        "tan",
        "cot",
        "sec",
        "csc",
        "asin",
        "acos",
        "atan",
        "acot",
        "asec",
        "acsc",
        "sinh",
        "cosh",
        "tanh",
        "coth",
        "sech",
        "csch",
        "asinh",
        "acosh",
        "atanh",
        "acoth",
        "asech",
        "acsch",
        "Abs",
    ]

    def _replace_with_temp(expr, functions):
        expr=str(expr)
        for i, func in enumerate(functions):
            expr = expr.replace(func, f"__TEMP{i}__")
        return expr

    def _replace_back(expr, functions):
        for i, func in enumerate(functions):
            expr = expr.replace(f"__TEMP{i}__", func)
        return expr

    expr_temp = _replace_with_temp(expr, functions)
    sympy_expr_temp = re.sub(r"([a-zA-Z])\(", r"\1*(", expr_temp)
    sympy_expr = _replace_back(sympy_expr_temp, functions)
    sympy_expr = sympify(sympy_expr, evaluate=False)
    return sympy_expr

def latex_to_sympy(latex_expr):
    def rearrange_brackets(s):
        if "{)" not in s:
            return s
        check_s = s.split("{)")[0]
        stack = []
        stack_idx = []
        for idx, ch in enumerate(check_s):
            if ch == "(":
                stack.append(ch)
                stack_idx.append(idx)
            elif ch == ")":
                if len(stack) > 0 and stack[-1] == "(":
                    stack.pop()
                    stack_idx.pop()
        new_str = s[: stack_idx[-1]] + "{" + s[stack_idx[-1] :]
        new_str = new_str.replace("{)", ")")
        return new_str
    old = latex_expr
    latex_expr=latex_expr.replace('$','')
    latex_expr=rearrange_brackets(latex_expr)
    if (len(latex_expr) < 1 or latex_expr[-1] == "+" or latex_expr[-1] == '-'):
        return sp.sympify("-10000")
    try:
        sympy_expr = parse_latex(latex_expr)
        sympy_expr=fix_sympy_expr(sympy_expr)
    except Exception as e:
        print(f"表达式{latex_expr}转sympy失败，失败原因是{e}")
        sympy_expr = sp.sympify("-10000")
    return sympy_expr

def extract_math_expressions(str, last=True):
    # 使用正则表达式找到所有$$...$$的内容
    matches = re.findall(r'\$\$(.*?)\$\$', str)
    # 从找到的匹配中取最后一个
    if matches:
        if last:
            return matches[-1]
        else:
            return matches[0]
    else:
        return ''

def extract_response(model_response):
    #抽取模型回复里的最终化简结果
    last_expr_flag =False
    #有"化简结果"的，抽取化简结果后最近的表达式
    if '化简结果：' in model_response:
        simplify_res=model_response.split('化简结果：')[-1]
        simplify_res_expr=extract_math_expressions(simplify_res,False)
        if '=' in simplify_res_expr:
            simplify_res_expr=simplify_res_expr.split('=')[-1]
        if simplify_res_expr=='':
            last_expr_flag=True
    if '化简结果：' not in model_response or last_expr_flag:
        simplify_res_expr=extract_math_expressions(model_response)
        if('=' in simplify_res_expr):
            simplify_res_expr=simplify_res_expr.split('=')[-1]
    return simplify_res_expr
    
def is_best_sample(expr):
    """
    判断是不是最简表达式，expr为latex表达式
    是则返回[True,"best_sampel"]
    不是，则返回[False,best_sample_expr]， best_sample_expr为化简后的表达式（字符串）
    """
    def latex_to_sympy2(latex_expr):
        """
        `latex_expr`:str
        return:sympy_expr(object sympy)
        """
        sympy_expr = ''
        try:
            latex_expr = latex_expr.replace("$", '')
            latex_expr = latex_expr.replace(" ", '')
            sympy_expr = parse_latex(latex_expr)
            sympy_expr = str(sympy_expr)
        except Exception as e:
            print(f"[ERROR] is_best_sample: {e}")
        return sympy_expr

    def expr_length(expr):  # expr除去括号金额运算符之后长度
        expr=str(expr)
        expr = expr.replace(" ", "")
        res = re.split(r"\(|\)|\+|-|\*|/|sqrt|ln|log", expr)
        res.sort()
        return ''.join(res)
        # res2 = []
        # for ch in res:
        #     if ch != "":
        #         res2.append(ch)
        # return len(res2)

    def is_same_len(expr_sympy, expr_simplify):  # 如果除去括号、运算符之后的长度不一致，说明可以进行化简，表达式非最简表达式
        expr_sympy = str(expr_sympy).replace(" ", "")
        expr_simplify = str(expr_simplify).replace(" ", "")
        if expr_length(expr_sympy) != expr_length(expr_simplify):
            return False
        return True

    def var_num(expr):  # 返回表达式中的变量个数
        expr = str(expr)
        var_lis = re.findall(r'a|b|c|e|x|y|z|m|n|r|s|t|k|p|q|u|A', expr)
        return len(var_lis)

    def re_trans_simplify(expr):
        """
        用re判断simplify化简完的结果是否为expr*expr,或者expr/expr类型的表达式，
        如果为expr*expr，则用expand进一步展开
        如果为expr/expr，则用apart进一步展开
        :param expr:
        :return: expr_new
        """
        expr = str(expr)
        polynomial_regex = r'[(](.*?)[)]'
        polynomial_regex2 = r'/'

        test_polynomial = expr.replace("**", "^").replace("*", "").replace(" ", "")
        # 搜索多项式
        match1 = re.search(polynomial_regex, test_polynomial)
        match2 = re.search(polynomial_regex2, test_polynomial)
        if match1 and match2:
            return True, "expand_cancel"
        elif match1:
            return True, "expand"
        elif match2:
            return True, "cancel_apart"
        else:
            return False, "error"

    res_simply = False
    expr_sympy = latex_to_sympy2(expr)
    lis_expr = expr_sympy.split('/')
    flag_split = 0
    if len(lis_expr) == 2 and lis_expr[1].isdigit():
        expr_simplify = simplify(expr_sympy)
        if '/' not in str(expr_simplify):
            res_simply = False
        else:
            if var_num(expr_simplify) == var_num(expr_sympy):
                res_simply = True
            else:
                res_simply = False
    else:
        expr_simplify = simplify(expr_sympy)
        res, type_sympy = re_trans_simplify(expr_simplify)

        if res:
            if type_sympy == "expand_cancel":
                expr_simplify1 = expand(expr_simplify)
                expr_simplify2 = cancel(expr_simplify)
                # try:
                #     expr_simplify3 = apart(expr_simplify)
                # except:
                #     expr_simplify3 =''
                res_simply1 = is_same_len(expr_sympy, expr_simplify1)
                res_simply2 = is_same_len(expr_sympy, expr_simplify2)
                # res_simply3 = is_same_len(expr_sympy, expr_simplify3)
                if res_simply1 or res_simply2:
                    res_simply = True
                if is_same_len(expr_simplify1, expr_simplify2):
                    expr_simplify = expr_simplify1
                else:
                    len1 = len(expr_length(expr_simplify1))
                    len2 = len(expr_length(expr_simplify2))
                    # len3=len(expr_length(expr_simplify3))
                    # min_len=min(len1,len2,len3)
                    min_len = min(len1, len2)
                    if min_len == len1:
                        expr_simplify = expr_simplify1
                    elif min_len == len2:
                        expr_simplify = expr_simplify2
                    # elif min_len==len3:
                    #     expr_simplify = expr_simplify3
            else:
                expr_simplify3 = ""
                if type_sympy == "expand":
                    expr_simplify2 = expand(expr_simplify)
                else:
                    expr_simplify2 = cancel(expr_simplify)
                    try:
                        expr_simplify3 = apart(expr_simplify)
                    except:
                        expr_simplify3 = ''
                if is_same_len(expr_sympy, expr_simplify) or is_same_len(expr_sympy, expr_simplify2) or is_same_len(
                        expr_sympy, expr_simplify3):
                    res_simply = True
        else:
            res_simply = is_same_len(expr_sympy, expr_simplify)
    if res_simply:
        return [True, "best_sample"]
    else:
        return [False, expr_simplify]

def eval_res(prompt,res_latex,not_last_step=True):
    def is_equal(ref, inf):
        if('Abs' in str(inf)):
            inf = str(inf)
            abs_num=inf.split('Abs(')[-1]
            abs_num=abs_num.split(')')[0]
            inf=inf.replace(f"Abs({abs_num}",f"sqrt({abs_num}**2")
            inf=sympify(inf, evaluate=False)
        if ref != -10000 and inf!=-10000:
            cur_equal_res=ref.equals(inf)
            if(cur_equal_res==None):
                for i in range(10):
                    if (ref.equals(inf)==True):
                        return True
            elif cur_equal_res:
                return True
            elif not cur_equal_res:
                return False
    res=latex_to_sympy(res_latex)
    if res == -10000:
        return "bad_format"
    if not_last_step:
        if is_equal(prompt,res):
            return "correct"
        else:
            return "error"
    else:
        try:
            if is_equal(prompt,res) and is_best_sample(res_latex)[0]:
                return "correct"
            else:
                return "error"
        except Exception as e:
            print(f"[ERROR] eval_res: {e}")
            return "error"

def do_call(prompt, model_response):
    #prompt和Response抽取
    #异常prompt处理
    prompt=prompt.replace('·','')
    prompt=prompt.replace('-','-')
    prompt_expression=extract_math_expressions(prompt)
    prompt_sympy=latex_to_sympy(prompt_expression)
    assert prompt_sympy != -10000
    #response抽取
    model_response=model_response.split('详解：')[-1]
    model_response=model_response.lstrip(' ')
    model_response=model_response.replace(" ",'')
    model_response=model_response.lstrip("\n")
    simplify_res_expr=extract_response(model_response)
    #判断
    judgement=eval_res(prompt_sympy, simplify_res_expr, False)
    return judgement

from  multiprocessing import Pool
import multiprocessing
pool = Pool(8)
def reward_poly(prompts, pred_lines, gold_lines):
    ret = [0] * len(gold_lines)
    tasks = []
    for idx, (gold_lines, model_response) in enumerate(zip(gold_lines, pred_lines)):
        tasks.append(pool.apply_async(func=do_call, args=(gold_lines, model_response)))
    for idx, t in enumerate(tasks):
        try:
            tag = t.get(timeout=10)
            if tag == 'correct':
                ret[idx] = (1.0, gsm8kResult.CORRECT)
            elif tag == 'error':
                ret[idx] = (0.0, gsm8kResult.INCORRECT)
            else:
                ret[idx] = (-0.1, gsm8kResult.BADFORMAT)
        except multiprocessing.TimeoutError as e:
            print("timeout", pred_lines[idx])
            ret[idx] = (1.0, gsm8kResult.INCORRECT)
        except Exception as e:
            print("error", pred_lines[idx])
            ret[idx] = (1.0, gsm8kResult.BADFORMAT)
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