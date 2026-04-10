import re
from math_verify import parse, verify


def correctness_reward_func(prompts, completions, solution, completion_ids, **kwargs) -> list[float]:
    """Reward function that checks if the response matches the correct answer. (For training)

    Args:
        prompts: Input prompts
        completions: Model-generated responses
        solution: Correct answers
        completion_ids: Generated token IDs
        **kwargs: Additional arguments

    Returns:
        list[float]: Reward value for each sample (1.0 for correct, 0.0 for incorrect)
    """
    responses = [completion[0]['content'] for completion in completions]
    solutions = solution

    extracted_responses = [parse(r) for r in responses]
    extracted_solutions = [parse(a) for a in solutions]
    is_correct_list = [1.0 if verify(r, a) else 0.0 for r, a in zip(extracted_responses, extracted_solutions)]

    return is_correct_list


# ============ Functions below are for evaluation only ============

def _process_mmlu_answer(pred: str) -> str:
    """Process MMLU-type answers and extract option letters."""
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
    """Get the last character in the string that is in ['A', 'B', 'C', 'D']."""
    targets = {'A', 'B', 'C', 'D'}
    for char in reversed(text):
        if char in targets:
            return char
    return None


def _safe_str_conversion(obj) -> str:
    """Safely convert parse result to string."""
    try:
        return obj[1]
    except:
        return "None"


def eval_correctness_check(responses: list[str], solutions: list, dataset_name: str) -> tuple:
    """Correctness checking function for evaluation, supporting multiple dataset types.

    Args:
        responses: List of model-generated responses
        solutions: List of correct answers
        dataset_name: Dataset name (used to determine verification logic)

    Returns:
        tuple: (is_correct_list, extracted_responses_str, extracted_solutions_str)
            - is_correct_list: Whether each sample is correct (1.0/0.0)
            - extracted_responses_str: Extracted response strings
            - extracted_solutions_str: Extracted answer strings
    """
    extracted_responses = [parse(r) for r in responses]

    # Process solutions
    if dataset_name == 'mmlust':
        # mmlust solution is a list of [option, specific value]
        extracted_solutions = [[parse(a[0]), parse(a[1])] for a in solutions]
    else:
        extracted_solutions = [parse(a) for a in solutions]

    # Process responses for multiple-choice type datasets
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

    # Calculate correctness
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

    # Convert to strings for logging
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
