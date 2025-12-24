from collections import deque

from zipage.config import Config
from zipage.engine.sequence import Sequence, SequenceStatus
from zipage.engine.block_manager import BlockManager
from typing import Optional
import time


class Scheduler:
    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(
            config.num_kvcache_blocks,
            config.kvcache_block_size,
            config.max_cache_blocks_per_seq + 1,
            config.enable_prefix_cache,
        )
        self.enable_prefix_cache = config.enable_prefix_cache
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.free_query_ids: deque[int] = deque(range(config.max_concurrency))
        self.used_query_ids: set[int] = set()
        self.enable_hybrid_engine = config.enable_hybrid_engine
        self.query_cache_len = config.query_cache_len
        self.block_size = self.block_manager.block_size

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def _allocate_query_id(self, seq: Sequence):
        query_id = self.free_query_ids.popleft()
        self.used_query_ids.add(query_id)
        seq.query_id = query_id

    def _deallocate_query_id(self, seq: Sequence):
        if seq.query_id != -1:
            self.used_query_ids.remove(seq.query_id)
            self.free_query_ids.append(seq.query_id)
            seq.query_id = -1

    def schedule(self) -> tuple[list[Sequence], bool]:
        # query slots allocation
        if self.enable_hybrid_engine and self.running and self.running[-1].query_id == -1:
            running_seqs = []
            while self.running and len(self.free_query_ids) > 0:
                seq = self.running.popleft()
                running_seqs.append(seq)
                if seq.query_id == -1:
                    self._allocate_query_id(seq)
            self.running.extendleft(reversed(running_seqs))
        
        # prefill
        prefilling_seqs = []
        num_batched_tokens = 0
        while self.waiting and len(prefilling_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            if num_batched_tokens + len(
                seq
            ) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
            if len(self.free_query_ids) == 0:
                if not self.enable_hybrid_engine:
                    break
                if not self.can_decode_without_query_id(seq):
                    break
            if len(self.free_query_ids) > 0:
                self._allocate_query_id(seq)
            self.block_manager.allocate(seq)
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            if not seq.num_cached_tokens == len(seq):
                num_batched_tokens += len(seq) - seq.num_cached_tokens
                prefilling_seqs.append(seq)
        if prefilling_seqs:
            return prefilling_seqs, True

        # decode
        running_seqs = []
        decoding_and_compressing_seqs = []
        while self.running and len(decoding_and_compressing_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            rejoining_seqs = []
            if seq.query_id != -1:
                while not self.block_manager.can_append_or_compress(seq):
                    if self.running and (
                        self.running[-1].query_id == -1 or self.enable_prefix_cache
                    ):
                        last_seq = self.running.pop()
                        if not (last_seq.compressed or last_seq.require_compress):
                            self.preempt(last_seq)
                        else:
                            rejoining_seqs.append(last_seq)
                    else:
                        # suspend with out preemption
                        running_seqs.append(seq)
                        break
                else:
                    self.block_manager.may_append(seq)
                    running_seqs.append(seq)
                    decoding_and_compressing_seqs.append(seq)
            else:
                not_blocking = self.can_decode_without_query_id(seq)
                while not_blocking and not self.block_manager.can_append(seq):
                    if self.running:
                        self.preempt(self.running.pop())
                    else:
                        running_seqs.append(seq)
                        break
                else:
                    running_seqs.append(seq)
                    if not_blocking:
                        self.block_manager.may_append(seq)
                        decoding_and_compressing_seqs.append(seq)
            self.running.extend(reversed(rejoining_seqs))
        self.running.extendleft(reversed(running_seqs))
        return decoding_and_compressing_seqs, False

    def can_decode_without_query_id(self, seq: Sequence) -> bool:
        if seq.num_blocks < self.block_manager.max_blocks_per_seq:
            return True
        else:
            return seq.last_block_num_tokens <= self.block_size - self.query_cache_len

    def can_allocate_query_id(self, seq: Sequence, is_prefill: bool) -> bool:
        return len(self.free_query_ids) > 0 and (
            len(seq.block_table) < self.block_manager.max_blocks_per_seq
            or seq.last_block_num_tokens <= self.block_size - self.query_cache_len + 1
        )

    def preempt(self, seq: Sequence):
        assert seq.compressed == False and seq.require_compress == False
        seq.status = SequenceStatus.WAITING
        self._deallocate_query_id(seq)
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(
        self,
        seqs: list[Sequence],
        token_ids: Optional[list[int]] = None,
    ) -> list[bool]:
        for seq in seqs:
            if seq.require_compress:
                self.block_manager.deallocate_block_to_release(seq)
                seq.block_table = seq.new_block_table
                seq.new_block_table = []
                seq.num_cached_tokens = (len(seq.block_table) - 1) * self.block_size + 1
                seq.compressed = True
                seq.require_compress = False
            if token_ids is not None:
                token_id = token_ids.pop(0)
                seq.append_token(token_id)
                if (
                    not seq.ignore_eos and token_id == self.eos
                ) or seq.num_completion_tokens == seq.max_tokens:
                    seq.time_finished = time.time()
                    seq.status = SequenceStatus.FINISHED
                    self.block_manager.deallocate(seq)
                    self.running.remove(seq)
                    self._deallocate_query_id(seq)
