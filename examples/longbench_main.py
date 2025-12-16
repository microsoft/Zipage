import os
import json
from transformers import AutoTokenizer
from .dataloader.longbench_dataloader import load_longbench_data


def main(args):

    path = os.path.expanduser(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(path)

    if args.compress:
        from zipage import ZipLLM as LLM, SamplingParams

        llm = LLM(
            path,
            enforce_eager=True,
            tensor_parallel_size=args.tensor_parallel_size,
            max_cache_blocks_per_seq=args.max_cache_blocks_per_seq,
            query_cache_len=args.query_cache_len,
            port=args.port,
            enable_log=True,
            layer_stride=args.layer_stride,
            decay_factor=args.decay_factor,
            use_global_score=args.use_global_score,
            max_norm=args.max_norm,
            repetition_penalty=args.repetition_penalty,
            query_selection_mode=args.query_selection_mode,
            enable_async_compress=args.enable_async_compress,
            enable_hybrid_engine=args.enable_hybrid_engine,
            strict_max_blocks=args.strict_max_blocks,
            enable_prefix_cache=args.enable_prefix_cache,
            similarity_lambda=args.similarity_lambda,
            use_similarity=args.use_similarity,
            similarity_temperature=args.similarity_temperature,
            lightning_similarity=args.lightning_similarity,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_num_batched_tokens=args.max_num_batched_tokens,
            enable_pooling=args.enable_pooling,
            continues_pooling=args.continues_pooling,
            pooling_size=args.pooling_size,
            log_interval=args.log_interval,
        )
    else:
        from nanovllm import SamplingParams
        from zipage import NanoLLM as LLM

        llm = LLM(
            path,
            enforce_eager=True,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_num_batched_tokens=args.max_num_batched_tokens,
            enable_log=True,
        )

    if args.dataset is None:
        datasets = [
            "qasper",
            "multifieldqa_en",
            "hotpotqa",
            "2wikimqa",
            "gov_report",
            "multi_news",
            "trec",
            "triviaqa",
            "samsum",
            "passage_count",
            "passage_retrieval_en",
            "lcc",
            "repobench-p",
        ]
    else:
        datasets = [args.dataset]

    sampling_params = []
    prompts = []
    data = []

    for dataset in datasets:
        p, d, maxlen = load_longbench_data(
            dataset,
            tokenizer,
            n_sample=args.n_sample,
            system_prompt=args.system_prompt,
            non_thinking=args.non_thinking,
        )
        prompts.extend(p)
        data.extend(d)
        if args.max_tokens is not None:
            maxlen = args.max_tokens
        sampling_params.extend(
            [SamplingParams(temperature=args.temperature, max_tokens=maxlen)]
            * len(p)
        )
    outputs = llm.generate(prompts, sampling_params)

    for d, output in zip(data, outputs):
        d["output"] = output["text"]
        d["output_len"] = len(output["token_ids"])

    if args.output_path:
        with open(args.output_path, "w") as f:
            info = {
                "data": data,
                "logger": llm.logger,
            }
            if args.compress:
                for key in llm.time_record:
                    if key.endswith("_sum"):
                        info[key] = llm.time_record[key]
                for key in llm.model_runner.time_record:
                    if key.endswith("_sum"):
                        info[key] = llm.model_runner.time_record[key]
            json.dump(info, f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--non_thinking", action="store_true")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_num_batched_tokens", type=int, default=63840)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--port", type=int, default=2333)
    parser.add_argument("--log_interval", type=int, default=2)
    # sampling related
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max_tokens", type=int, default=None)
    parser.add_argument("--n_sample", type=int, default=1)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    # dataset related
    parser.add_argument("--split_len", type=int, default=None)
    parser.add_argument("--input_key", type=str, default=None)
    parser.add_argument("--target_key", type=str, default=None)
    parser.add_argument("--system_prompt", type=str, default=None)

    # compress related
    parser.add_argument("--compress", action="store_true")
    parser.add_argument("--layer_stride", type=int, default=1)
    parser.add_argument("--max_cache_blocks_per_seq", type=int, default=8)
    parser.add_argument("--query_cache_len", type=int, default=16)
    parser.add_argument("--decay_factor", type=float, default=0.6)
    parser.add_argument("--use_global_score", action="store_true")
    parser.add_argument("--max_norm", action="store_true")
    parser.add_argument("--query_selection_mode", type=str, default="recent")
    parser.add_argument("--enable_async_compress", action="store_true")
    parser.add_argument("--enable_hybrid_engine", action="store_true")
    parser.add_argument("--strict_max_blocks", action="store_true")
    parser.add_argument("--use_attention_sink", action="store_true")
    parser.add_argument("--sink_len", type=int, default=4)
    parser.add_argument("--similarity_lambda", type=float, default=0.4)
    parser.add_argument("--lightning_similarity", action="store_true")
    parser.add_argument("--similarity_temperature", type=float, default=1.0)
    parser.add_argument("--use_similarity", action="store_true")
    parser.add_argument("--enable_prefix_cache", action="store_true")
    parser.add_argument("--enable_pooling", action="store_true")
    parser.add_argument("--continues_pooling", action="store_true")
    parser.add_argument("--pooling_size", type=int, default=5)
    args = parser.parse_args()
    main(args)
