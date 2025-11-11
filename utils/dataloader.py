from datasets import load_dataset

INPUT_KEY = {
    "math-ai/aime24": "problem",
    "zwhe99/amc23": "question",
    "agentica-org/DeepScaleR-Preview-Dataset": "problem",
    "THUDM/LongBench-v2": "question",
}
TARGET_KEY = {
    "math-ai/aime24": "solution",
    "zwhe99/amc23": "answer",
    "agentica-org/DeepScaleR-Preview-Dataset": "answer",
    "THUDM/LongBench-v2": "answer",
}
SPLIT = {
    "math-ai/aime24": "test",
    "zwhe99/amc23": "test",
    "THUDM/LongBench-v2": "train",
    "agentica-org/DeepScaleR-Preview-Dataset": "train",
}


def load_data(
    dataset_name_or_path,
    tokenizer,
    n_sample=1,
    input_key=None,
    target_key=None,
    split_len=None,
    system_prompt=None,
):
    split = "test"
    if dataset_name_or_path in INPUT_KEY:
        input_key = INPUT_KEY[dataset_name_or_path]
        target_key = TARGET_KEY[dataset_name_or_path]
        split = SPLIT[dataset_name_or_path]
    else:
        assert (
            input_key is not None
        ), "input_key is not provided for dataset: {dataset_name_or_path}"
        assert (
            target_key is not None
        ), "target_key is not provided for dataset: {dataset_name_or_path}"

    prompts = []
    data = []
    dataset = load_dataset(dataset_name_or_path, split=split)

    if split_len is not None:
        dataset = dataset.select(range(0, split_len))

    for index, item in enumerate(dataset):
        if dataset_name_or_path == "THUDM/LongBench-v2":
            context = item["context"]
            question = item["question"]
            choices = [
                "A. " + item["choice_A"],
                "B. " + item["choice_B"],
                "C. " + item["choice_C"],
                "D. " + item["choice_D"],
            ]
            question = (
                context
                + "\n"
                + question
                + "\n"
                + "\n".join(choices)
                + "\n"
                + "Answer: "
            )
            if len(question)>64*1024:
                continue
        else:
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
        for _ in range(n_sample):
            prompts.append(prompt)
            if not isinstance(item[target_key], str):
                item[target_key] = str(item[target_key])
            if not dataset_name_or_path == "THUDM/LongBench-v2":
                data.append(
                    {"question": question, "answer": item[target_key], "index": index}
                )
            else:
                data.append({"answer": item[target_key], "index": index})
    return prompts, data
