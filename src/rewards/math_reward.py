import re
from math_verify import parse, verify


def correctness_reward_func(prompts, completions, solution, completion_ids, **kwargs) -> list[float]:
    """奖励函数，检查响应是否与正确答案匹配。（用于训练）

    Args:
        prompts: 输入的 prompts
        completions: 模型生成的回复
        solution: 正确答案
        completion_ids: 生成的 token ids
        **kwargs: 其他参数

    Returns:
        list[float]: 每个样本的奖励值 (1.0 正确, 0.0 错误)
    """
    responses = [completion[0]['content'] for completion in completions]
    solutions = solution

    extracted_responses = [parse(r) for r in responses]
    extracted_solutions = [parse(a) for a in solutions]
    is_correct_list = [1.0 if verify(r, a) else 0.0 for r, a in zip(extracted_responses, extracted_solutions)]

    return is_correct_list


# ============ 以下为 eval 专用函数 ============

def _process_mmlu_answer(pred: str) -> str:
    """处理 MMLU 类型答案，提取选项字母。"""
    pred = pred.strip("\n").rstrip(".").rstrip("/").strip(" ")

    tmp = re.findall(r"\b(A|B|C|D)\b", pred.upper())

    if tmp:
        pred = tmp
    else:
        pred = [pred.strip().strip(".")]

    if len(pred) == 0:
        pred = ""
    else:
        pred = pred[0].rstrip(".").rstrip("/")
    return pred


def _get_last_option(text: str) -> str:
    """获取字符串中最后一个在 ['A', 'B', 'C', 'D'] 中的字符。"""
    targets = {'A', 'B', 'C', 'D'}
    for char in reversed(text):
        if char in targets:
            return char
    return None


def _safe_str_conversion(obj) -> str:
    """安全地将 parse 结果转换为字符串。"""
    try:
        return obj[1]
    except:
        return "None"


def eval_correctness_check(responses: list[str], solutions: list, dataset_name: str) -> tuple:
    """评估时的正确性检查函数，支持多种数据集类型。

    Args:
        responses: 模型生成的回复列表
        solutions: 正确答案列表
        dataset_name: 数据集名称 (用于确定验证逻辑)

    Returns:
        tuple: (is_correct_list, extracted_responses_str, extracted_solutions_str)
            - is_correct_list: 每个样本是否正确 (1.0/0.0)
            - extracted_responses_str: 提取的回复字符串
            - extracted_solutions_str: 提取的答案字符串
    """
    extracted_responses = [parse(r) for r in responses]

    # 处理 solutions
    if dataset_name == 'mmlust':
        # mmlust 的 solution 是 [选项, 具体数值] 的列表
        extracted_solutions = [[parse(a[0]), parse(a[1])] for a in solutions]
    else:
        extracted_solutions = [parse(a) for a in solutions]

    # 处理选择题类型数据集的 responses
    if dataset_name in ['arcc', 'mmlust', 'gpqa']:
        new_extracted_responses = []
        for er, r in zip(extracted_responses, responses):
            lst = []
            if er:
                lst.append(parse(f"\\boxed{{ {_process_mmlu_answer(er[-1])} }}"))
            else:
                lst.append(er)

            lst.append(parse(f"\\boxed{{ {_get_last_option(r)} }}"))
            new_extracted_responses.append(lst)

        extracted_responses = new_extracted_responses

    # 计算正确性
    if dataset_name == 'mmlust':
        is_correct_list = [
            1.0 if (verify(r[0], a[0]) or verify(r[0], a[1]) or
                    verify(r[1], a[0]) or verify(r[1], a[1])) else 0.0
            for r, a in zip(extracted_responses, extracted_solutions)
        ]
    elif dataset_name in ['arcc', 'gpqa']:
        is_correct_list = [
            1.0 if (verify(r[0], a) or verify(r[1], a)) else 0.0
            for r, a in zip(extracted_responses, extracted_solutions)
        ]
    else:
        is_correct_list = [
            1.0 if verify(r, a) else 0.0
            for r, a in zip(extracted_responses, extracted_solutions)
        ]

    # 转换为字符串用于日志记录
    if dataset_name in ['arcc', 'mmlust', 'gpqa']:
        extracted_responses_str = [
            [_safe_str_conversion(r[0]), _safe_str_conversion(r[1])]
            for r in extracted_responses
        ]
    else:
        extracted_responses_str = [_safe_str_conversion(r) for r in extracted_responses]

    if dataset_name == 'mmlust':
        extracted_solutions_str = [
            [_safe_str_conversion(a[0]), _safe_str_conversion(a[1])]
            for a in extracted_solutions
        ]
    else:
        extracted_solutions_str = [_safe_str_conversion(a) for a in extracted_solutions]

    return is_correct_list, extracted_responses_str, extracted_solutions_str
