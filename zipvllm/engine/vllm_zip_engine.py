import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed

from zipvllm.config import Config
from zipvllm.sampling_params import SamplingParams
from zipvllm.engine.sequence import Sequence
from zipvllm.engine.scheduler import Scheduler
from zipvllm.engine.model_runner import ModelRunner


class LLMEngine:
    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)

        port = kwargs.get("port", 2333)
        self.model_runner = ModelRunner(config, 0, self.events, port)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        config.pad = self.tokenizer.pad_token_id
        self.scheduler = Scheduler(config)

        enable_log = kwargs.get("enable_log", False)
        if enable_log:
            self.logger = {}
            for key in [
                "throughput",
                "running_seqs",
                "waiting_seqs",
                "block_occupancy",
                "time",
            ]:
                self.logger[key] = []
        else:
            self.logger = None
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.compress_future = None
        self.run_future = None

        atexit.register(self.exit)

    def exit(self):
        self.executor.shutdown(wait=True)
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def _compress_task(self, compress_seqs: list[Sequence]):
        self.model_runner.call("compress", compress_seqs)

    def _run_task(self, run_seqs: list[Sequence], is_prefill: bool):
        token_ids, entropy = self.model_runner.call("run", run_seqs, is_prefill)
        self.scheduler.postprocess(run_seqs, token_ids, entropy)
        return token_ids, entropy

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        if not is_prefill:
            compress_seqs: list[Sequence] = []
            for seq in seqs:
                if seq.require_compress:
                    compress_seqs.append(seq)
            if compress_seqs:
                self.model_runner.call("compress", compress_seqs)
        token_ids, entropy = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids, entropy)
        outputs = [
            (seq.request_id, seq.completion_token_ids)
            for seq in seqs
            if seq.is_finished
        ]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens

    def async_step(self):
        seqs, is_prefill = self.scheduler.schedule()
        compress_seqs: list[Sequence] = []
        run_seqs: list[Sequence] = []

        for seq in seqs:
            if seq.require_compress:
                compress_seqs.append(seq)
            else:
                run_seqs.append(seq)

        self.compress_future = self.executor.submit(
            self._compress_task, compress_seqs if compress_seqs else None
        )
        self.run_future = (
            self.executor.submit(self._run_task, run_seqs, is_prefill)
            if run_seqs
            else None
        )

        if self.compress_future is not None:
            self.compress_future.result()
        if self.run_future is not None:
            self.run_future.result()

        outputs = [
            (seq.request_id, seq.completion_token_ids)
            for seq in run_seqs
            if seq.is_finished
        ]
        num_tokens = sum(len(seq) for seq in run_seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def reset_logger(self):
        if self.logger:
            for key in self.logger:
                self.logger[key].clear()

    def _append_log_entry(
        self, time_from_start, running_seqs, waiting_seqs, decode_throughput
    ):
        self.logger["throughput"].append(decode_throughput)
        self.logger["running_seqs"].append(running_seqs)
        self.logger["waiting_seqs"].append(waiting_seqs)
        self.logger["block_occupancy"].append(
            self.scheduler.block_manager.block_occupancy
        )
        self.logger["time"].append(time_from_start)

    def log_step(self, time_from_start, running_seqs, waiting_seqs, decode_throughput):
        if not self.logger:
            return
        should_log = (
            not self.logger["time"] or time_from_start - self.logger["time"][-1] > 5
        )
        if should_log:
            self._append_log_entry(
                time_from_start, running_seqs, waiting_seqs, decode_throughput
            )

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.0
        start_time = perf_counter()
        self.reset_logger()
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            current_time = perf_counter()
            if num_tokens > 0:
                prefill_throughput = num_tokens / (current_time - t)
            else:
                decode_throughput = -num_tokens / (current_time - t)
            pbar.set_postfix(
                {
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                }
            )
            self.log_step(
                current_time - start_time,
                len(self.scheduler.running),
                len(self.scheduler.waiting),
                decode_throughput,
            )
            for request_id, token_ids in output:
                outputs[request_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        outputs = [outputs[request_id] for request_id in sorted(outputs.keys())]
        outputs = [
            {"text": self.tokenizer.decode(token_ids), "token_ids": token_ids}
            for token_ids in outputs
        ]
        if use_tqdm:
            pbar.close()
        return outputs
