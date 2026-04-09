from datasets import load_dataset, Dataset, concatenate_datasets

SHOTS = [
    [
        {'role': 'user', 'content': "Kevin Kangaroo begins hopping on a number line at 0. He wants to get to 1, but he can hop only $\\frac{1}{3}$ of the distance. Each hop tires him out so that he continues to hop $\\frac{1}{3}$ of the remaining distance. How far has he hopped after five hops? Express your answer as a common fraction."},
        {'role': 'assistant', 'content': "Let's think step by step\nKevin hops $1/3$ of the remaining distance with every hop.\nHis first hop takes $1/3$ closer.\nFor his second hop, he has $2/3$ left to travel, so he hops forward $(2/3)(1/3)$.\nFor his third hop, he has $(2/3)^2$ left to travel, so he hops forward $(2/3)^2(1/3)$.\nIn general, Kevin hops forward $(2/3)^{k-1}(1/3)$ on his $k$th hop.\nWe want to find how far he has hopped after five hops.\nThis is a finite geometric series with first term $1/3$, common ratio $2/3$, and five terms.\nThus, Kevin has hopped $\\frac{\\frac{1}{3}\\left(1-\\left(\\frac{2}{3}\\right)^5\\right)}{1-\\frac{2}{3}} = \\boxed{\\frac{211}{243}}$.\nThe answer is \\frac{211}{243}}"}
    ],
    [
        {'role': 'user', 'content': "What is the area of the region defined by the equation $x^2+y^2 - 7 = 4y-14x+3$?"},
        {'role': 'assistant', 'content': "Let's think step by step\nWe rewrite the equation as $x^2 + 14x + y^2 - 4y = 10$ and then complete the square,\nresulting in  $(x+7)^2-49 + (y-2)^2-4=10$,\nor $(x+7)^2+(y-2)^2=63$.\nThis is the equation of a circle with center $(-7, 2)$ and radius $\\sqrt{63},$\nso the area of this region is $\\pi r^2 = \\boxed{63\\pi}$.\nThe answer is 63\\pi"}
    ],
    [
        {'role': 'user', 'content': "If $x^2+y^2=1$, what is the largest possible value of $|x|+|y|$?"},
        {'role': 'assistant', 'content': "Let's think step by step\nIf $(x,y)$ lies on the circle,\nso does $(x,-y),$ $(-x,-y),$ and $(-x,-y),$ (which all give the same value of $|x| + |y|$),\nso we can assume that $x \\ge 0$ and $y \\ge 0.$\nThen $|x| + |y| = x + y.$  Squaring, we get\n\\[(x + y)^2 = x^2 + 2xy + y^2 = 1 + 2xy.\\]\nNote that $(x - y)^2 \\ge 0.$\nExpanding, we get $x^2 - 2xy + y^2 \\ge 0,$ so $2xy \\le x^2 + y^2 = 1.$\nHence,\\[1 + 2xy \\le 2,\\]which means $x + y \\le \\sqrt{2}.$\nEquality occurs when $x = y = \\frac{1}{\\sqrt{2}},$\nso the maximum value of $|x| + |y|$ is $\\boxed{\\sqrt{2}}.$\nThe answer is \\sqrt{2}"}
    ],
    [
        {'role': 'user', 'content': "If $f(x)=\\frac{ax+b}{cx+d}, abcd\\not=0$ and $f(f(x))=x$ for all $x$ in the domain of $f$, what is the value of $a+d$?"},
        {'role': 'assistant', 'content': "Let's think step by step\nThe condition $f(f(x))$ means that $f$ is the inverse of itself,\nso its graph is symmetrical about the line $y = x$.\nWith a rational function of this form, we will have two asymptotes:\na vertical one at $x=-d/c$ if $cx+d$ does not divide $ax+b$,\nand a horizontal one at $y=a/c$,\nif we take the limit of $f(x)$ as $x$ goes to $\\pm\\infty$.\nIn order for $f$ to be its own inverse, the intersection of the asymptotes must lie on the line $y=x$\nso that it and its asymptotes reflect onto themselves.\nThis means that $-d/c=a/c$,\nand therefore $-d=a$ and $a+d=\\boxed{0}$.\nThe answer is 0"}
    ]
]

SYSTEM_PROMPT = """Please reason step by step, and put your final answer within \\boxed{}."""
SYSTEM_PROMPT2 = """Please reason step by step, and put your final answer within \\boxed{} with (A, B, C or D)."""


def get_prompt_shot(question, shot=0):
    prompt = [{'role': 'system', 'content': SYSTEM_PROMPT}]
    for i in range(shot):
        prompt.extend(SHOTS[i])
    prompt.extend([{'role': 'user', 'content': question + '\n' + SYSTEM_PROMPT}])
    return prompt


def get_gsm8k_questions(split="train", shot=0, **kwargs) -> Dataset:
    def extract_hash_answer(text: str) -> str | None:
        if "####" not in text:
            return None
        return text.split("####")[1].strip().replace(",", "").replace("$", "")

    data = load_dataset("openai/gsm8k", 'main')[split]
    data = data.map(lambda x: {
        'prompt': get_prompt_shot(x['question'], shot),
        'solution': f"{extract_hash_answer(x['answer'])}",
        'question': x['question']
    })
    return data


def get_math500_questions(shot=0, **kwargs) -> Dataset:
    dataset = load_dataset("HuggingFaceH4/MATH-500")['test']
    dataset = dataset.map(lambda x: {
        'prompt': get_prompt_shot(x['problem'], shot),
        'solution': x['solution'],
        'question': x['problem']
    })
    return dataset


def get_math_questions(split="train", subset='algebra', shot=0, **kwargs) -> Dataset:
    if subset == 'all':
        name_list = ['algebra', 'counting_and_probability', 'geometry', 'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus']
    elif isinstance(subset, str):
        name_list = [subset]
    elif isinstance(subset, list):
        name_list = subset
    else:
        raise ValueError('subset must be a string or a list of strings')

    data = Dataset.from_dict({})
    for name in name_list:
        data0 = load_dataset("EleutherAI/hendrycks_math", name)[split]
        data = concatenate_datasets([data, data0])

    data = data.map(lambda x: {
        'prompt': get_prompt_shot(x['problem'], shot),
        'solution': x['solution'],
        'question': x['problem']
    })
    return data


def get_dapo_questions(split="train", subset='all', shot=0, **kwargs) -> Dataset:
    data = load_dataset("open-r1/DAPO-Math-17k-Processed", subset)["train"]
    data = data.map(lambda x: {
        'prompt': get_prompt_shot(x['prompt'], shot),
        'solution': x['solution'],
        'question': x['prompt']
    })
    train_test_split = data.train_test_split(test_size=0.1, seed=42)

    return train_test_split[split]


def get_aime2024_questions(shot=0, **kwargs) -> Dataset:
    data = load_dataset("HuggingFaceH4/aime_2024")["train"]
    data = data.map(lambda x: {
        'prompt': get_prompt_shot(x['problem'], shot),
        'solution': x['answer'],
        'question': x['problem']
    })
    return data


def get_aime2025_questions(shot=0, **kwargs) -> Dataset:
    data1 = load_dataset("opencompass/AIME2025", "AIME2025-I")["test"]
    data2 = load_dataset("opencompass/AIME2025", "AIME2025-II")['test']
    data = concatenate_datasets([data1, data2])
    data = data.map(lambda x: {
        'prompt': get_prompt_shot(x['question'], shot),
        'solution': x['answer'],
        'question': x['question']
    })
    return data


def get_minervamath_questions(shot=0, **kwargs) -> Dataset:
    dataset = load_dataset("math-ai/minervamath")['test']
    dataset = dataset.map(lambda x: {
        'prompt': get_prompt_shot(x['question'], shot),
        'solution': f"\\boxed{{ {x['answer']} }}",
        'question': x['question'],
    })
    return dataset


def get_amc23_questions(shot=0, **kwargs) -> Dataset:
    dataset = load_dataset("zwhe99/amc23")['test']
    dataset = dataset.map(lambda x: {
        'prompt': get_prompt_shot(x['question'], shot),
        'solution': str(x['answer']),
        'question': x['question'],
    })
    return dataset


def get_arcc_questions(split="test", shot=0, **kwargs) -> Dataset:
    if split == "none":
        split = "test"

    def get_prompt(question, choices):
        prompt = f"Question: {question}\nOptions:\n"
        for i, choice in enumerate(choices):
            prompt += f"{chr(65 + i)}. {choice}\n"
        return prompt

    def my_char(c):
        mapping = {'1': 'A', '2': 'B', '3': 'C', '4': 'D'}
        return mapping.get(c, c)

    dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge")[split]

    dataset = dataset.map(lambda x: {
        'prompt': [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": get_prompt(x['question'], x['choices']['text']).strip() + '\n' + SYSTEM_PROMPT},
        ],
        'solution': f"\\boxed{{ {my_char(x['answerKey'])} }}",
        'answerKey': my_char(x['answerKey']),
        'question': get_prompt(x['question'], x['choices']['text']).strip(),
    })
    return dataset


def get_mmlust_questions(shot=0, **kwargs) -> Dataset:
    def get_prompt(question, choices):
        prompt = f"Question: {question}\nOptions:\n"
        for i, choice in enumerate(choices):
            prompt += f"{chr(65 + i)}. {choice}\n"
        return prompt

    dataset = load_dataset("TIGER-Lab/MMLU-STEM")['test']

    dic = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}

    dataset = dataset.map(lambda x: {
        'prompt': [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": get_prompt(x['question'], x['choices']).strip() + '\n' + SYSTEM_PROMPT2},
        ],
        'solution': [f"\\boxed{{ {dic[x['answer']]} }}", f"\\boxed{{ {x['choices'][x['answer']]} }}"],
        'question': get_prompt(x['question'], x['choices']).strip(),
    })
    return dataset


# 数据集注册表
DATASET_REGISTRY = {
    "gsm8k": get_gsm8k_questions,
    "math500": get_math500_questions,
    "math": get_math_questions,
    "dapo": get_dapo_questions,
    "aime2024": get_aime2024_questions,
    "aime2025": get_aime2025_questions,
    "minervamath": get_minervamath_questions,
    "amc23": get_amc23_questions,
    "arcc": get_arcc_questions,
    "mmlust": get_mmlust_questions,
}


def get_dataset(name: str, split: str = "train", subset: str = "all", shot: int = 0) -> Dataset:
    """统一的数据集加载接口。

    Args:
        name: 数据集名称 (gsm8k, math500, math, dapo, aime2024, aime2025, etc.)
        split: 数据集划分 (train, test, none)
        subset: 子集名称 (用于 math, dapo 等数据集)
        shot: few-shot 数量

    Returns:
        Dataset: 加载的数据集
    """
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_REGISTRY.keys())}")

    return DATASET_REGISTRY[name](split=split, subset=subset, shot=shot)
