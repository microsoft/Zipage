import os
import json
from transformers import AutoTokenizer
from utils.dataloader import load_data


def main(args):

    path = os.path.expanduser(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(path)

    if args.compress:
        from zipvllm import ZipLLM as LLM, SamplingParams

        llm = LLM(
            path,
            enforce_eager=True,
            tensor_parallel_size=args.tensor_parallel_size,
            max_blocks_per_seq=args.max_blocks_per_seq,
            query_cache_len=args.query_cache_len,
            port=args.port,
            enable_log=True,
            decay_factor=args.decay_factor,
            use_score_cache=args.use_score_cache,
            repetition_penalty=args.repetition_penalty,
            query_selection_mode=args.query_selection_mode,
        )
        sampling_params = SamplingParams(temperature=0.6, max_tokens=args.max_tokens)
    else:
        from nanovllm import SamplingParams
        from zipvllm import NanoLLM as LLM

        llm = LLM(
            path,
            enforce_eager=True,
            tensor_parallel_size=args.tensor_parallel_size,
            enable_log=True,
        )
        sampling_params = SamplingParams(temperature=0.6, max_tokens=args.max_tokens)

    prompts, data = load_data(
        args.dataset,
        tokenizer,
        n_sample=args.n_sample,
        system_prompt=args.system_prompt,
        split_len=args.split_len,
    )
    outputs = llm.generate(prompts, sampling_params)

    for d, output in zip(data, outputs):
        d["output"] = output["text"]
    if args.output_path:
        with open(args.output_path, "w") as f:
            json.dump({"data": data, "logger": llm.logger}, f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--port", type=int, default=2333)
    # sampling related
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--n_sample", type=int, default=1)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    # dataset related
    parser.add_argument("--split_len", type=int, default=None)
    parser.add_argument("--input_key", type=str, default=None)
    parser.add_argument("--target_key", type=str, default=None)
    parser.add_argument("--system_prompt", type=str, default=None)

    # compress related
    parser.add_argument("--compress", action="store_true")
    parser.add_argument("--max_blocks_per_seq", type=int, default=8)
    parser.add_argument("--query_cache_len", type=int, default=16)
    parser.add_argument("--decay_factor", type=float, default=0.6)
    parser.add_argument("--use_score_cache", action="store_true")
    parser.add_argument("--query_selection_mode", type=str, default="recent")
    args = parser.parse_args()
    main(args)
