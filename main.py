import os

from transformers import AutoTokenizer
from utils.dataloader import load_data


def main(args):

    path = os.path.expanduser(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(path)

    if args.compress:
        # from compressvllm import LLM, SamplingParams

        # llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)
        # sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
        pass
    else:
        from nanovllm import LLM, SamplingParams

        llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)
        sampling_params = SamplingParams(temperature=0.6, max_tokens=256)

    prompts, data = load_data(
        args.dataset,
        tokenizer,
        n_sample=args.n_sample,
        system_prompt=args.system_prompt,
        split_len=args.split_len,
    )
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    # sampling related
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--n_sample", type=int, default=1)
    # dataset related
    parser.add_argument("--split_len", type=int, default=None)
    parser.add_argument("--input_key", type=str, default=None)
    parser.add_argument("--target_key", type=str, default=None)
    parser.add_argument("--system_prompt", type=str, default=None)

    # compress related

    parser.add_argument("--compress", action="store_true")
    args = parser.parse_args()
    main(args)
