import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner

<<<<<<< HEAD
from collections import defaultdict
=======
>>>>>>> 2aaa790 (init commit)

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
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
<<<<<<< HEAD
        self.enable_log = kwargs.get("enable_log", False)
        self.logger = defaultdict(list)
=======
        enable_log = kwargs.get("enable_log", False)
        if enable_log:
            self.logger = {
                "throughput": [],
                "running_seqs": [],
                "waiting_seqs": [],
                "block_occupancy": [],
                "time": [],
            }
        else:
            self.logger = None
>>>>>>> 2aaa790 (init commit)
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids)
        outputs = [
            (seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished
        ]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def reset_logger(self):
<<<<<<< HEAD
        self.logger.clear()

    def log_step(self, time_from_start, running_seqs, waiting_seqs, decode_throughput):
        if self.enable_log:
=======
        if self.logger:
            self.logger["throughput"] = []
            self.logger["running_seqs"] = []
            self.logger["waiting_seqs"] = []
            self.logger["block_occupancy"] = []
            self.logger["time"] = []

    def log_step(self, time_from_start, running_seqs, waiting_seqs, decode_throughput):
        if self.logger:
>>>>>>> 2aaa790 (init commit)
            block_occupancy = len(self.scheduler.block_manager.used_block_ids) / (
                len(self.scheduler.block_manager.used_block_ids)
                + len(self.scheduler.block_manager.free_block_ids)
            )
            if len(self.logger["throughput"]) == 0:
                self.logger["throughput"].append(decode_throughput)
                self.logger["running_seqs"].append(running_seqs)
                self.logger["waiting_seqs"].append(waiting_seqs)
                self.logger["block_occupancy"].append(block_occupancy)
                self.logger["time"].append(time_from_start)
            else:
                prev_time = self.logger["time"][-1]
                if time_from_start - prev_time > 5:
                    self.logger["throughput"].append(decode_throughput)
                    self.logger["running_seqs"].append(running_seqs)
                    self.logger["waiting_seqs"].append(waiting_seqs)
                    self.logger["block_occupancy"].append(block_occupancy)
                    self.logger["time"].append(time_from_start)

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
<<<<<<< HEAD
        dt = start_time
=======
>>>>>>> 2aaa790 (init commit)
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            current_time = perf_counter()
            if num_tokens > 0:
                prefill_throughput = num_tokens / (current_time - t)
            else:
<<<<<<< HEAD
                decode_throughput = -num_tokens / (current_time - dt)
                dt = current_time
=======
                decode_throughput = -num_tokens / (current_time - t)
>>>>>>> 2aaa790 (init commit)
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
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [
            {"text": self.tokenizer.decode(token_ids), "token_ids": token_ids}
            for token_ids in outputs
        ]
        if use_tqdm:
            pbar.close()
        return outputs
