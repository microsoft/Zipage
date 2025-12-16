import datasets
from lcb_runner.benchmarks.code_generation import CodeGenerationProblem
from lcb_runner.lm_styles import LMStyle
from transformers import AutoTokenizer

# from lcb_runner.prompts.code_generation import format_prompt_generation

FORMATTING_MESSAGE_WITH_STARTER_CODE = "You will use the following starter code to write the solution to the problem and enclose your code within delimiters."
FORMATTING_WITHOUT_STARTER_CODE = "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT."


def format_prompt_generation(
    question: CodeGenerationProblem, LanguageModelStyle: LMStyle
) -> str:
    prompt = "You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.\n\n"
    prompt += f"Question: {question.question_content}\n\n"
    if question.starter_code:
        prompt += f"{FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
        prompt += f"```python\n{question.starter_code}\n```\n"
    else:
        prompt += f"{FORMATTING_WITHOUT_STARTER_CODE}\n"
        prompt += f"```python\n# YOUR CODE HERE\n```\n"
    return prompt


def load_code_generation_dataset_not_fast() -> list[CodeGenerationProblem]:
    dataset = datasets.load_dataset(
        "json", data_files="datasets/livecodebench/test.jsonl", split="train"
    )
    dataset = [CodeGenerationProblem(**p) for p in dataset]  # type: ignore
    print(f"Loaded {len(dataset)} problems")
    return dataset


def format_message(
    prompt: str, tokenizer: AutoTokenizer, system_prompt: str | None = None
) -> str:
    if system_prompt:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
    else:
        messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        truncation=False,
        padding=False,
    )


def load_lcb_codegeneration_dataset(
    args, tokenizer: AutoTokenizer, system_prompt: str | None = None
) -> list[CodeGenerationProblem]:

    benchmark = load_code_generation_dataset_not_fast()
    benchmark = sorted(benchmark, key=lambda x: x.question_id)
    format_prompt = lambda x, y: format_message(
        format_prompt_generation(x, y), tokenizer, system_prompt
    )
    return benchmark, format_prompt
