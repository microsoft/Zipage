from datasets import load_dataset
import os
import random

INPUT_KEY = {
    "math-ai/aime24": "problem",
    "zwhe99/amc23": "question",
    "juanlrdc/gmsk8": "problem",
    "HuggingFaceH4/MATH-500": "problem",
}
TARGET_KEY = {
    "math-ai/aime24": "solution",
    "zwhe99/amc23": "answer",
    "juanlrdc/gmsk8": "solution",
    "HuggingFaceH4/MATH-500": "answer",
}
SPLIT = {
    "math-ai/aime24": "test",
    "zwhe99/amc23": "test",
    "juanlrdc/gmsk8": "test",
    "HuggingFaceH4/MATH-500": "test",
}


def dataset2list(
    dataset,
    n_sample,
    input_key,
    target_key,
    system_prompt,
    tokenizer,
    non_thinking=False,
):
    prompts = []
    data = []
    for index, item in enumerate(dataset):
        question = item[input_key]
        if system_prompt is not None:
            prompt = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ]
        else:
            prompt = [
                {"role": "user", "content": question},
            ]
        prompt = tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )
        if non_thinking:
            prompt = prompt + "</think>"
        for _ in range(n_sample):
            prompts.append(prompt)
            if not isinstance(item[target_key], str):
                item[target_key] = str(item[target_key])
            data.append(
                {"question": question, "answer": item[target_key], "index": index}
            )
    return prompts, data


def load_math_data(
    dataset_name_or_path,
    tokenizer,
    n_sample=1,
    input_key=None,
    target_key=None,
    split_len=None,
    system_prompt=None,
):

    if dataset_name_or_path == "mixed":
        gmsk8 = load_dataset("juanlrdc/gmsk8", split="test")
        amc23 = load_dataset("zwhe99/amc23", split="test")
        prompts = []
        data = []

        p, d = dataset2list(
            amc23,
            n_sample,
            INPUT_KEY["zwhe99/amc23"],
            TARGET_KEY["zwhe99/amc23"],
            system_prompt,
            tokenizer,
        )
        prompts.extend(p)
        data.extend(d)
        p, d = dataset2list(
            gmsk8,
            n_sample,
            INPUT_KEY["juanlrdc/gmsk8"],
            TARGET_KEY["juanlrdc/gmsk8"],
            system_prompt,
            tokenizer,
            non_thinking=True,
        )
        prompts.extend(p)
        data.extend(d)
        combined = list(zip(prompts, data))
        random.seed(42)
        random.shuffle(combined)
        prompts, data = zip(*combined)
        prompts = list(prompts)
        data = list(data)
        return prompts, data

    else:
        split = "test"
        if dataset_name_or_path in INPUT_KEY:
            input_key = INPUT_KEY[dataset_name_or_path]
            target_key = TARGET_KEY[dataset_name_or_path]
            split = SPLIT[dataset_name_or_path]
        else:
            assert input_key is not None and target_key is not None
        dataset = load_dataset(dataset_name_or_path, split=split)
        if split_len is not None:
            dataset = dataset.select(range(0, split_len))
        prompts, data = dataset2list(
            dataset,
            n_sample,
            input_key,
            target_key,
            system_prompt,
            tokenizer,
        )

    return prompts, data
