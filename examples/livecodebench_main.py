import os
import json
import sys
from pathlib import Path

from .dataloader.livecodebench_dataloader import load_lcb_codegeneration_dataset

# lcb_runner (third_party) reads few-shot files via relative paths like
# "lcb_runner/prompts/...", so we temporarily chdir into repo_root/third_party
# during import to avoid FileNotFoundError, without modifying third_party code.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_THIRD_PARTY = _REPO_ROOT / "third_party"
_ORIG_CWD = Path.cwd()
sys.path.insert(0, str(_THIRD_PARTY))
os.chdir(_THIRD_PARTY)
try:
    from lcb_runner.lm_styles import LanguageModel, LMStyle
    from lcb_runner.runner.scenario_router import Scenario
    from lcb_runner.runner.scenario_router import (
        combine_results,
        sort_and_extract_save_results,
        get_metrics,
    )
    from lcb_runner.evaluation import extract_instance_results
finally:
    os.chdir(_REPO_ROOT)

from transformers import AutoTokenizer, AutoConfig
from collections import defaultdict


class LCBRunner:
    def __init__(self, args, model):
        self.model = model
        self.args = args
        if args.compress:
            from zipage import ZipLLM as LLM, SamplingParams

            self.llm = LLM(
                args.model_path,
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
            )
        else:
            from nanovllm import SamplingParams
            from zipage import NanoLLM as LLM

            self.llm = LLM(
                args.path,
                enforce_eager=True,
                tensor_parallel_size=args.tensor_parallel_size,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_num_batched_tokens=args.max_num_batched_tokens,
                enable_log=True,
                port=args.port,
            )
        self.sampling_params = SamplingParams(
            temperature=args.temperature, max_tokens=args.max_tokens
        )
        self.sample_n = args.n_sample
        self.loginfo = defaultdict(list)

    def _run_single(self, prompt: str) -> list[str]:
        pass

    def run(self, prompts: list[str]) -> list[list[str]]:
        all_prompts = []
        for prompt in prompts:
            for _ in range(self.sample_n):
                all_prompts.append(prompt)

        outputs = self.llm.generate(all_prompts, self.sampling_params)

        outputs_text = []

        for i in range(len(outputs)):
            if i % self.sample_n == 0:
                outputs_text.append([])
            outputs_text[-1].append(outputs[i]["text"])
            self.loginfo["output_len"].append(len(outputs[i]["token_ids"]))
        return outputs_text

    def run_main(self, benchmark: list, format_prompt: callable) -> list[list[str]]:
        if self.args.scenario == Scenario.selfrepair:
            return self.run_main_repair(benchmark, format_prompt)

        prompts = [
            format_prompt(problem, self.model.model_style) for problem in benchmark
        ]
        outputs = self.run(prompts)
        return outputs


def main(args):

    args.path = os.path.expanduser(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.path)

    model = LanguageModel(
        model_name=args.path,
        model_repr=args.path,
        model_style=LMStyle.DeepSeekR1,
        release_date=None,
        link=None
    )

    benchmark, format_prompt = load_lcb_codegeneration_dataset(
        args, tokenizer, args.system_prompt
    )
    lcb_runner = LCBRunner(args, model)
    results = lcb_runner.run_main(benchmark, format_prompt)

    # eval
    combined_results = combine_results(args.scenario, results, model)
    save_results = [
        instance.insert_output(outputs_list, extracted_list)
        for instance, (outputs_list, extracted_list) in zip(benchmark, combined_results)
    ]

    save_results, combined_results = sort_and_extract_save_results(
        args.scenario, save_results
    )
    metrics = get_metrics(args.scenario, args, benchmark, combined_results)
    graded = extract_instance_results(metrics[1])
    metrics_log = {
        "graded": graded,
        "pass@1": metrics[0],
        "difficulty": [instance.difficulty.value for instance in benchmark],
    }

    if args.output_path:
        with open(args.output_path, "w") as f:
            info = {
                "logger": lcb_runner.llm.logger,
                "output": results,
                "output_len": lcb_runner.loginfo["output_len"],
                "metrics": metrics_log,
            }
            if args.compress:
                for key in lcb_runner.llm.time_record:
                    if key.endswith("_sum"):
                        info[key] = lcb_runner.llm.time_record[key]
                for key in lcb_runner.llm.model_runner.time_record:
                    if key.endswith("_sum"):
                        info[key] = lcb_runner.llm.model_runner.time_record[key]
            json.dump(info, f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_num_batched_tokens", type=int, default=63840)
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

    args.scenario = Scenario.codegeneration
    args.start_date = None
    args.end_date = None
    args.stop = None
    args.num_process_evaluate = 32
    args.timeout = 6
    args.local_model_path = None
    main(args)
