import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory
import threading

from time import perf_counter
from collections import defaultdict


from zipage.layers.sampler import Sampler
from zipage.config import Config
from zipage.engine.sequence import Sequence
from zipage.models import AutoModelForCausalLM
from zipage.utils.context import set_context, get_context, reset_context
from zipage.utils.loader import load_model

from zipage.kernel.compress_kv import compress_kv
from zipage.kernel.compress_score import compress_score
from zipage.kernel.attention_score import attention_score
from zipage.kernel.raw_similarity_score import raw_similarity_score
from zipage.kernel.lightning_similarity_score import lightning_similarity_score
from zipage.kernel.global_score import global_score
from zipage.kernel.window_mask import window_mask
from zipage.kernel.utils import topk_mask
import torch.nn.functional as F


class ModelRunner:

    def __init__(
        self,
        config: Config,
        rank: int,
        event: Event | list[Event],
        compress_event: Event | list[Event],
        compress_done_event: Event | list[Event],
        port: int = 2333,
    ):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.max_blocks_per_seq = config.max_cache_blocks_per_seq + 1
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event
        self.compress_event = compress_event
        self.compress_done_event = compress_done_event

        self.query_cache_len = config.query_cache_len

        self.use_global_score = config.use_global_score
        self.max_norm = config.max_norm
        self.decay_factor = config.decay_factor
        self.score_cache = None

        self.use_similarity = config.use_similarity
        self.lightning_similarity = config.lightning_similarity
        self.similarity_lambda = config.similarity_lambda
        self.similarity_temperature = config.similarity_temperature

        self.enable_pooling = config.enable_pooling
        self.continues_pooling = config.continues_pooling

        self.use_attention_sink = config.use_attention_sink
        self.sink_len = config.sink_len
        self.layer_stride = config.layer_stride

        self.pooling_size = 5

        assert self.sink_len < self.block_size * (self.max_blocks_per_seq - 1)

        dist.init_process_group(
            "nccl", f"tcp://localhost:{port}", world_size=self.world_size, rank=rank
        )
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        self.model = AutoModelForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler(config.repetition_penalty)
        self.compress_stream = torch.cuda.Stream()
        self.run_stream = torch.cuda.Stream()
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        self.time_record = defaultdict(int)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="zipage", create=True, size=2**20)
                self.compress_shm = SharedMemory(
                    name="compress", create=True, size=2**20
                )
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="zipage")
                self.compress_shm = SharedMemory(name="compress")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            self.compress_shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
                self.compress_shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def run_loop(self):
        torch.cuda.set_device(self.rank)
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def compress_loop(self):
        torch.cuda.set_device(self.rank)
        while True:
            method_name, args = self.read_shm(True)
            if method_name == "exit":
                self.compress_done_event.set()
                break
            self.call("compress", *args)
            self.compress_done_event.set()

    def loop(self):
        run_thread = threading.Thread(target=self.run_loop)
        compress_thread = threading.Thread(target=self.compress_loop)
        run_thread.start()
        compress_thread.start()
        compress_thread.join()
        run_thread.join()

    def read_shm(self, is_compress: bool = False):
        assert self.world_size > 1 and self.rank > 0
        if not is_compress:
            self.event.wait()
            n = int.from_bytes(self.shm.buf[0:4], "little")
            method_name, *args = pickle.loads(self.shm.buf[4 : n + 4])
            self.event.clear()
        else:
            self.compress_event.wait()
            n = int.from_bytes(self.compress_shm.buf[0:4], "little")
            method_name, *args = pickle.loads(self.compress_shm.buf[4 : n + 4])
            self.compress_event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        if method_name == "run":
            self.shm.buf[0:4] = n.to_bytes(4, "little")
            self.shm.buf[4 : n + 4] = data
            for event in self.event:
                event.set()
        elif method_name == "compress":
            self.compress_shm.buf[0:4] = n.to_bytes(4, "little")
            self.compress_shm.buf[4 : n + 4] = data
            for event in self.compress_event:
                event.set()
        else:
            # exit
            self.compress_shm.buf[0:4] = n.to_bytes(4, "little")
            self.compress_shm.buf[4 : n + 4] = data
            for event in self.compress_event:
                event.set()
            for event in self.compress_done_event:
                event.wait()
                event.clear()
            self.shm.buf[0:4] = n.to_bytes(4, "little")
            self.shm.buf[4 : n + 4] = data
            for event in self.event:
                event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        res = method(*args)

        if self.world_size > 1 and self.rank == 0 and method_name == "compress":
            for event in self.compress_done_event:
                event.wait()
                event.clear()
        return res

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

        assert hf_config.num_attention_heads % self.world_size == 0
        num_q_heads = hf_config.num_attention_heads // self.world_size

        block_bytes = (
            hf_config.num_hidden_layers
            * self.block_size
            * num_kv_heads
            * (2 * hf_config.head_dim + self.use_global_score)
            * hf_config.torch_dtype.itemsize
        )

        query_cache_bytes = (
            hf_config.num_hidden_layers
            * query_cache_len
            * num_q_heads
            * hf_config.head_dim
            * hf_config.torch_dtype.itemsize
        )
        available_bytes = int(
            total * config.gpu_memory_utilization - used - peak + current
        )
        max_concurrency = available_bytes // (
            block_bytes * max_block_perseq + query_cache_bytes
        )
        config.max_concurrency = min(config.max_num_seqs, max_concurrency)

        if self.world_size > 1:
            dist.barrier()
            local_max_conc = torch.tensor(
                [config.max_concurrency], dtype=torch.int32, device="cuda"
            )
            dist.all_reduce(local_max_conc, op=dist.ReduceOp.MIN)
            config.max_concurrency = int(local_max_conc.item())

        config.num_kvcache_blocks = (
            available_bytes - config.max_concurrency * query_cache_bytes
        ) // block_bytes

        if self.world_size > 1:
            dist.barrier()
            local_num_kvcache_blocks = torch.tensor(
                [config.num_kvcache_blocks], dtype=torch.int32, device="cuda"
            )
            dist.all_reduce(local_num_kvcache_blocks, op=dist.ReduceOp.MIN)
            config.num_kvcache_blocks = int(local_num_kvcache_blocks.item())

        assert config.max_concurrency > 0
        assert config.num_kvcache_blocks > 0

        self.query_cache = torch.empty(
            hf_config.num_hidden_layers,
            config.max_concurrency,
            query_cache_len,
            num_q_heads,
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
        if self.use_global_score:
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
        self,
        seq: Sequence,
        is_prefill: bool = False,
        cu_len: int = 0,
        seqlen_q: int = 0,
    ):
        query_slot_mapping = []
        if seq.seq_id == -1:
            return query_slot_mapping
        if is_prefill:
            if seqlen_q % self.block_size > self.block_size - self.query_cache_len:
                start = (
                    (seqlen_q + self.block_size - 1) // self.block_size
                ) * self.block_size - self.query_cache_len
                for idx, pos in enumerate(range(start, seqlen_q)):
                    query_slot_mapping.append((cu_len + pos, seq.seq_id, idx))

        else:
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
        block_tables = None
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens :])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            query_slot_mapping.extend(
                self.get_query_slot_mapping(
                    seq, True, cu_len=cu_seqlens_q[-1], seqlen_q=seqlen_q
                )
            )
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)

            if not seq.block_table:  # warmup
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens
                slot_mapping.extend(list(range(start, end)))
            seq.num_cached_tokens = seqlen
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:  # prefix cache
            block_tables = self.prepare_block_tables(seqs)
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
            block_tables=block_tables,
        )
        return input_ids, positions

    def compress(self, seqs: list[Sequence]):
        def record(name):
            st = torch.cuda.Event(enable_timing=True)
            ed = torch.cuda.Event(enable_timing=True)
            st.record(self.compress_stream)
            return st, ed, name

        # prepare
        max_len_block_table = 0
        block_tables = []
        seq_ids = []
        compressed = []
        target_block_tables = []
        for seq in seqs:
            assert seq.seq_id != -1
            seq_ids.append(seq.seq_id)
            max_len_block_table = max(max_len_block_table, len(seq.block_table))
            # < 0 and !=-1 means the last block
            block_tables.append(seq.block_table[:-1] + [-seq.block_table[-1] - 2])
            for block_id in seq.block_table:
                assert block_id >= 0
            compressed.append(seq.compressed)
            target_block_tables.append(seq.new_block_table)
            assert len(seq.new_block_table) == self.max_blocks_per_seq
            for block_id in seq.new_block_table:
                assert block_id >= 0

        times = []

        with torch.cuda.stream(self.compress_stream):
            seq_ids = torch.tensor(seq_ids, dtype=torch.int32, pin_memory=True).cuda(
                non_blocking=True
            )
            compressed = torch.tensor(
                compressed, dtype=torch.bool, pin_memory=True
            ).cuda(non_blocking=True)
            target_block_tables = torch.tensor(
                target_block_tables, dtype=torch.int32, pin_memory=True
            ).cuda(non_blocking=True)
            for i in range(len(block_tables)):
                block_tables[i] = block_tables[i] + [-1] * (
                    max_len_block_table - len(block_tables[i])
                )
            block_tables = torch.tensor(
                block_tables, dtype=torch.int32, pin_memory=True
            ).cuda(non_blocking=True)

            for layer_id in range(0, len(self.model.model.layers), self.layer_stride):
                k_cache = self.kv_cache[0, layer_id : layer_id + self.layer_stride]
                v_cache = self.kv_cache[1, layer_id : layer_id + self.layer_stride]
                query_cache = self.query_cache[layer_id : layer_id + self.layer_stride]

                # attention score
                st, ed, name = record("attention_score")
                scores = attention_score(k_cache, query_cache, seq_ids, block_tables)
                ed.record(self.compress_stream)
                times.append((name, st, ed))

                num_layers, bsz, num_kv_heads, num_blocks, block_size = scores.shape

                # global score
                if self.use_global_score:
                    st, ed, name = record("global_score")
                    if self.max_norm:
                        scores = scores.view(num_layers, bsz, num_kv_heads, -1)
                        scores = scores.div_(scores.max(dim=-1, keepdim=True).values)
                        scores = scores.view(
                            num_layers, bsz, num_kv_heads, num_blocks, block_size
                        )
                    scores = global_score(
                        scores,
                        self.score_cache[layer_id : layer_id + self.layer_stride],
                        block_tables,
                        compressed,
                        self.decay_factor,
                    )
                    ed.record(self.compress_stream)
                    times.append((name, st, ed))

                # seqnence dimension max pooling
                scores = scores.view(num_layers * bsz, num_kv_heads, -1)
                if self.enable_pooling:
                    if not self.continues_pooling and torch.all(compressed):
                        pass
                    else:
                        st, ed, name = record("pooling")
                        pooledscores = F.max_pool1d(
                            scores,
                            kernel_size=self.pooling_size,
                            stride=1,
                            padding=self.pooling_size // 2,
                        )
                        if not self.continues_pooling:
                            compressed_mask = (
                                compressed.unsqueeze(0)
                                .expand(num_layers, bsz)
                                .unsqueeze(-1)
                                .unsqueeze(-1)
                            )
                            compressed_mask = compressed_mask.reshape(
                                num_layers * bsz, 1, 1
                            )

                            scores = torch.where(
                                compressed_mask,
                                scores,
                                pooledscores,
                            )
                        else:
                            scores = pooledscores
                        ed.record(self.compress_stream)
                        times.append((name, st, ed))

                scores = scores.view(
                    num_layers, bsz, num_kv_heads, num_blocks, block_size
                )

                # similarity score
                if self.use_similarity:
                    st, ed, name = record("similarity_score")
                    if self.lightning_similarity:
                        similarity = lightning_similarity_score(
                            k_cache,
                            block_tables,
                            temperature=self.similarity_temperature,
                        )
                    else:
                        similarity = raw_similarity_score(
                            k_cache,
                            block_tables,
                            temperature=self.similarity_temperature,
                        )
                    if self.use_global_score and self.max_norm:
                        similarity = similarity.view(num_layers, bsz, num_kv_heads, -1)
                        similarity = similarity.div_(
                            similarity.max(dim=-1, keepdim=True).values
                        )
                        similarity = similarity.reshape(
                            num_layers, bsz, num_kv_heads, num_blocks, block_size
                        )
                    scores = scores * self.similarity_lambda - similarity * (
                        1 - self.similarity_lambda
                    )
                    ed.record(self.compress_stream)
                    times.append((name, st, ed))

                # attention sink
                if self.use_attention_sink:
                    st, ed, name = record("attention_sink")
                    scores = scores.view(num_layers, bsz, num_kv_heads, -1)
                    mask = (
                        torch.arange(block_size * num_blocks, device=scores.device)
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .unsqueeze(0)
                        < self.sink_len
                    )
                    scores = scores.masked_fill_(mask, float("inf"))
                    ed.record(self.compress_stream)
                    times.append((name, st, ed))

                scores = scores.view(
                    num_layers, bsz, num_kv_heads, num_blocks, block_size
                )

                # window mask
                st, ed, name = record("window_mask")
                scores = window_mask(scores, block_tables, self.query_cache_len)
                ed.record(self.compress_stream)
                times.append((name, st, ed))

                # top-k mask
                st, ed, name = record("topk_mask")
                mask = (block_tables == -1).unsqueeze(1).unsqueeze(-1).unsqueeze(0)
                scores = scores.masked_fill_(mask, -float("inf"))
                scores = scores.view(num_layers, bsz, num_kv_heads, -1)
                keep_flag = topk_mask(
                    scores, self.block_size * (self.max_blocks_per_seq - 1)
                )
                keep_flag = keep_flag.view(
                    num_layers, bsz, num_kv_heads, num_blocks, block_size
                )
                ed.record(self.compress_stream)
                times.append((name, st, ed))

                # compress kv
                st, ed, name = record("compress_kv")
                compress_kv(
                    k_cache, v_cache, keep_flag, block_tables, target_block_tables
                )
                ed.record(self.compress_stream)
                times.append((name, st, ed))

                # compress global score
                if self.use_global_score:
                    st, ed, name = record("compress_global_score")
                    compress_score(
                        self.score_cache[layer_id : layer_id + self.layer_stride],
                        keep_flag,
                        block_tables,
                        target_block_tables,
                    )
                    ed.record(self.compress_stream)
                    times.append((name, st, ed))
        
        times[-1][2].synchronize()
        
        avg_time=defaultdict(list)
        for name, st, ed in times:
            # elapsed_time expects (start, end) and returns milliseconds.
            t = st.elapsed_time(ed)/1000
            avg_time[name].append(t)
            self.time_record[name+'_sum'] += t
        for name in avg_time:
            self.time_record[name] = sum(avg_time[name])/len(avg_time[name])
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
        with torch.cuda.stream(self.run_stream):
            input_ids, positions = (
                self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
            )
            recent_token_ids, temperatures = (
                self.prepare_sample(seqs) if self.rank == 0 else (None, None)
            )
            logits = self.run_model(input_ids, positions, is_prefill)
            token_ids = (
                self.sampler(
                    logits,
                    temperatures,
                    recent_token_ids,
                ).tolist()
                if self.rank == 0
                else None
            )
            reset_context()
        return token_ids

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
