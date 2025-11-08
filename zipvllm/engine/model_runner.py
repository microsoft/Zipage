import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from time import perf_counter
from collections import defaultdict

from zipvllm.layers.sampler import Sampler
from zipvllm.config import Config
from zipvllm.engine.sequence import Sequence
from zipvllm.models.qwen3 import Qwen3ForCausalLM
from zipvllm.utils.context import set_context, get_context, reset_context
from zipvllm.utils.loader import load_model

from zipvllm.kernel.compress_kv import compress_kv
from zipvllm.kernel.compress_kv_out_order import compress_kv_out_order
from zipvllm.kernel.compress_score import compress_score
from zipvllm.kernel.compress_score_out_order import compress_score_out_order
from zipvllm.kernel.attention_score import attention_score
from zipvllm.kernel.raw_similarity_score import raw_similarity_score
from zipvllm.kernel.global_score import global_score
from zipvllm.kernel.utils import topk_mask, get_compress_slot_indices


class ModelRunner:

    def __init__(
        self, config: Config, rank: int, event: Event | list[Event], port: int = 2333
    ):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.max_blocks_per_seq = config.max_blocks_per_seq
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        self.query_cache_len = config.query_cache_len
        self.query_selection_mode = config.query_selection_mode
        self.query_interval = self.block_size // self.query_cache_len
        self.use_score_cache = config.use_score_cache
        self.decay_factor = config.decay_factor
        self.use_similarity = config.use_similarity
        self.similarity_factor = config.similarity_factor
        self.use_attention_sink = config.use_attention_sink
        self.sink_len = config.sink_len
        self.keep_order = config.keep_order

        assert self.sink_len < self.block_size * (self.max_blocks_per_seq - 2)

        dist.init_process_group(
            "nccl", f"tcp://localhost:{port}", world_size=self.world_size, rank=rank
        )
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler(config.repetition_penalty)
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)


        self.time_record = defaultdict(int)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4 : n + 4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4 : n + 4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = (
            self.config.max_num_batched_tokens,
            self.config.max_model_len,
        )
        num_seqs = min(
            max_num_batched_tokens // max_model_len, self.config.max_num_seqs
        )
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self, query_cache_len=16, max_block_perseq=8):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size

        block_bytes = (
            hf_config.num_hidden_layers
            * self.block_size
            * num_kv_heads
            * (2 * hf_config.head_dim + 1)
            * hf_config.torch_dtype.itemsize
        )

        query_cache_bytes = (
            hf_config.num_hidden_layers
            * query_cache_len
            * hf_config.num_attention_heads
            * hf_config.head_dim
            * hf_config.torch_dtype.itemsize
        )

        config.num_kvcache_blocks = int(
            total * config.gpu_memory_utilization - used - peak + current
        ) // (block_bytes + query_cache_bytes // max_block_perseq)

        config.max_num_seqs = min(
            config.max_num_seqs, config.num_kvcache_blocks // max_block_perseq
        )

        self.query_cache = torch.empty(
            hf_config.num_hidden_layers,
            config.max_num_seqs,
            query_cache_len,
            hf_config.num_attention_heads,
            hf_config.head_dim,
        )

        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(
            2,
            hf_config.num_hidden_layers,
            config.num_kvcache_blocks,
            self.block_size,
            num_kv_heads,
            hf_config.head_dim,
        )
        self.score_cache = torch.empty(
            hf_config.num_hidden_layers,
            config.num_kvcache_blocks,
            self.block_size,
            num_kv_heads,
        )

        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                module.q_cache = self.query_cache[layer_id]
                module.score_cache = self.score_cache[layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [
            seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs
        ]
        block_tables = torch.tensor(
            block_tables, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        return block_tables

    def get_query_slot_mapping(
        self, seq: Sequence, is_prefill: bool = False, cu_len: int = 0, seqlen: int = 0
    ):
        query_slot_mapping = []
        if seq.seq_id == -1:
            return query_slot_mapping
        if is_prefill:
            if self.query_selection_mode == "recent" or "entropy":
                if seqlen % self.block_size > self.block_size - self.query_cache_len:
                    start = seq.num_blocks * self.block_size - self.query_cache_len
                    for idx, pos in enumerate(range(start, seqlen)):
                        query_slot_mapping.append((cu_len + pos, seq.seq_id, idx))
            elif self.query_selection_mode == "interval":
                for idx, pos in enumerate(
                    range(
                        seqlen - seq.last_block_num_tokens + self.query_interval,
                        seqlen,
                        self.query_interval,
                    )
                ):
                    query_slot_mapping.append((cu_len + pos, seq.seq_id, idx))

        else:
            if self.query_selection_mode == "recent":
                if seq.last_block_num_tokens > self.block_size - self.query_cache_len:
                    query_slot_mapping.append(
                        (
                            cu_len,
                            seq.seq_id,
                            seq.last_block_num_tokens
                            - (self.block_size - self.query_cache_len)
                            - 1,
                        )
                    )
            elif self.query_selection_mode == "interval":
                if seq.last_block_num_tokens % self.query_interval == 0:
                    query_slot_mapping.append(
                        (
                            cu_len,
                            seq.seq_id,
                            seq.last_block_num_tokens // self.query_interval - 1,
                        )
                    )
            elif self.query_selection_mode == "entropy":
                if seq.last_block_num_tokens == 1:
                    seq.query_entropy_list.clear()
                if len(seq.query_entropy_list) < self.query_cache_len:
                    seq.query_entropy_list.append(seq.last_token_entropy)
                    query_slot_mapping.append(
                        (
                            cu_len,
                            seq.seq_id,
                            self.query_cache_len - len(seq.query_entropy_list),
                        )
                    )
                else:
                    min_entropy_idx = min(
                        range(len(seq.query_entropy_list)),
                        key=lambda i: seq.query_entropy_list[i],
                    )
                    if seq.query_entropy_list[min_entropy_idx] < seq.last_token_entropy:
                        seq.query_entropy_list[min_entropy_idx] = seq.last_token_entropy
                        query_slot_mapping.append(
                            (
                                cu_len,
                                seq.seq_id,
                                self.query_cache_len - min_entropy_idx - 1,
                            )
                        )

        return query_slot_mapping

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        query_slot_mapping = []
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq[:])
            positions.extend(list(range(seqlen)))
            seqlen_q = seqlen
            seqlen_k = seqlen
            query_slot_mapping.extend(
                self.get_query_slot_mapping(
                    seq, True, cu_len=cu_seqlens_q[-1], seqlen=seqlen
                )
            )
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)

            if not seq.block_table:  # warmup
                continue
            for i in range(seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens
                slot_mapping.extend(list(range(start, end)))
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        cu_seqlens_q = torch.tensor(
            cu_seqlens_q, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(
            cu_seqlens_k, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        slot_mapping = torch.tensor(
            slot_mapping, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        if query_slot_mapping:
            query_slot_mapping = torch.tensor(
                query_slot_mapping, dtype=torch.int32, pin_memory=True
            ).cuda(non_blocking=True)
        else:
            query_slot_mapping = None
        set_context(
            True,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            slot_mapping,
            query_slot_mapping=query_slot_mapping,
            context_lens=None,
            block_tables=None,
        )
        return input_ids, positions

    def compress(self, seqs: list[Sequence]):
        # prepare
        max_len_block_table = 0
        block_tables = []
        seq_ids = []
        compressed = []
        for seq in seqs:
            assert seq.require_compress
            seq_ids.append(seq.seq_id)
            max_len_block_table = max(max_len_block_table, len(seq.block_table) - 1)
            block_tables.append(seq.block_table[:-2] + [-seq.block_table[-2] - 2])
            compressed.append(seq.compressed)

        seq_ids = torch.tensor(seq_ids, dtype=torch.int32, pin_memory=True).cuda(
            non_blocking=True
        )
        compressed = torch.tensor(compressed, dtype=torch.bool, pin_memory=True).cuda(
            non_blocking=True
        )
        for i in range(len(block_tables)):
            block_tables[i] = block_tables[i] + [-1] * (
                max_len_block_table - len(block_tables[i])
            )
        block_tables = torch.tensor(
            block_tables, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)

        for layer_id in range(len(self.model.model.layers)):
            k_cache = self.kv_cache[0, layer_id]
            v_cache = self.kv_cache[1, layer_id]
            query_cache = self.query_cache[layer_id]
            
            start_time = perf_counter()
            scores = attention_score(k_cache, query_cache, seq_ids, block_tables)
            end_time = perf_counter()
            self.time_record["attention_score"] += end_time - start_time
            
            bsz, num_kv_heads, num_blocks, block_size = scores.shape
            if self.use_score_cache:
                start_time = perf_counter()
                scores = scores.view(bsz, num_kv_heads, -1)
                scores = scores.div_(scores.max(dim=-1, keepdim=True).values)
                scores = scores.view(bsz, num_kv_heads, num_blocks, block_size)
                scores = global_score(
                    scores,
                    self.score_cache[layer_id],
                    block_tables,
                    compressed,
                    self.decay_factor,
                )
                end_time = perf_counter()
                self.time_record["global_score"] += end_time - start_time

            scores = scores.view(bsz, num_kv_heads, -1)
            if self.use_similarity:
                start_time = perf_counter()
                similarity = raw_similarity_score(k_cache, block_tables).view(
                    bsz, num_kv_heads, -1
                )
                if self.use_score_cache:
                    similarity = similarity.div_(
                        similarity.max(dim=-1, keepdim=True).values
                    )
                scores = (
                    scores * (1 - self.similarity_factor)
                    + similarity * self.similarity_factor
                )
                end_time = perf_counter()
                self.time_record["similarity_score"] += end_time - start_time
            if self.use_attention_sink:
                start_time = perf_counter()
                mask = (
                    torch.arange(block_size * num_blocks, device=scores.device)
                    .unsqueeze(0)
                    .unsqueeze(0)
                    < self.sink_len
                )
                scores = scores.masked_fill_(mask, float("inf"))
                end_time = perf_counter()
                self.time_record["attention_sink"] += end_time - start_time
            scores = scores.view(bsz, num_kv_heads, num_blocks, block_size)
            start_time = perf_counter()
            mask = (block_tables == -1).unsqueeze(1).unsqueeze(-1)
            scores = scores.masked_fill_(mask, -float("inf"))
            scores = scores.view(bsz, num_kv_heads, -1)
            keep_flag = topk_mask(
                scores, self.block_size * (self.max_blocks_per_seq - 2)
            )
            keep_flag = keep_flag.view(bsz, num_kv_heads, num_blocks, block_size)
            end_time = perf_counter()
            self.time_record["topk_mask"] += end_time - start_time
            start_time = perf_counter()

            
            start_time = perf_counter()
            if self.keep_order:
                compress_kv(k_cache, v_cache, keep_flag, block_tables)
                if self.use_score_cache:
                    compress_score(self.score_cache[layer_id], keep_flag, block_tables)
            else:
                save_indices, load_indices = get_compress_slot_indices(
                    keep_flag, block_tables, self.max_blocks_per_seq - 2
                )
                compress_kv_out_order(
                    k_cache, v_cache, save_indices, load_indices, block_tables
                )
                if self.use_score_cache:
                    compress_score_out_order(
                        self.score_cache[layer_id],
                        save_indices,
                        load_indices,
                        block_tables,
                    )
            end_time = perf_counter()
            self.time_record["compress_kv"] += end_time - start_time

        for seq in seqs:
            if len(seq.block_table) > self.max_blocks_per_seq:
                seq.block_to_release = seq.block_table[self.max_blocks_per_seq - 1 : -1]
            seq.compressed = True
            seq.block_table = (
                seq.block_table[: self.max_blocks_per_seq - 2]
                + [seq.block_table[-1]]
                + [seq.block_table[self.max_blocks_per_seq - 2]]
            )
        return seqs

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        query_slot_mapping = []
        context_lens = []
        for i, seq in enumerate(seqs):
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(
                (len(seq.block_table) - 1) * self.block_size + seq.last_block_num_tokens
            )
            slot_mapping.append(
                seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
            )
            query_slot_mapping.extend(self.get_query_slot_mapping(seq, False, cu_len=i))
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        slot_mapping = torch.tensor(
            slot_mapping, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        context_lens = torch.tensor(
            context_lens, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        if query_slot_mapping:
            query_slot_mapping = torch.tensor(
                query_slot_mapping, dtype=torch.int32, pin_memory=True
            ).cuda(non_blocking=True)
        else:
            query_slot_mapping = None
        set_context(
            False,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            query_slot_mapping=query_slot_mapping,
        )
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence], recent_token_len: int = 32):
        token_ids = None
        if self.sampler.repetition_penalty_processor is not None:
            token_ids = []
            max_len = 0
            for seq in seqs:
                token_ids.append(seq.last_n_completion_tokens(recent_token_len))
                max_len = max(max_len, len(token_ids[-1]))
            for i in range(len(token_ids)):
                token_ids[i] = token_ids[i] + [self.config.pad] * (
                    max_len - len(token_ids[i])
                )
            token_ids = torch.tensor(
                token_ids, dtype=torch.int64, pin_memory=True
            ).cuda(non_blocking=True)
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(
            temperatures, dtype=torch.float32, pin_memory=True
        ).cuda(non_blocking=True)
        if token_ids is not None:
            assert token_ids.shape[0] == temperatures.shape[0]
        return token_ids, temperatures

    @torch.inference_mode()
    def run_model(
        self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool
    ):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            raise ValueError("Not implemented")
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][
                :bs, : context.block_tables.size(1)
            ] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = (
            self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        )
        recent_token_ids, temperatures = (
            self.prepare_sample(seqs) if self.rank == 0 else (None, None)
        )
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids, entropy = (
            self.sampler(
                logits,
                temperatures,
                recent_token_ids,
                self.query_selection_mode == "entropy",
            )
            if self.rank == 0
            else (None, None)
        )
        token_ids = token_ids.tolist()
        reset_context()
        return token_ids, entropy

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(
                False,
                slot_mapping=slot_mapping[:bs],
                context_lens=context_lens[:bs],
                block_tables=block_tables[:bs],
            )
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
