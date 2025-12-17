from collections import deque

from zipage.config import Config
from zipage.engine.sequence import Sequence, SequenceStatus
from zipage.engine.block_manager import BlockManager
from typing import Optional
import threading


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
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.free_seq_ids: deque[int] = deque(range(config.max_concurrency))
        self.used_seq_ids: set[int] = set()
        self.enable_hybrid_engine = config.enable_hybrid_engine
        self.strict_max_blocks = config.strict_max_blocks
        self.query_cache_len = config.query_cache_len
        self.block_size = self.block_manager.block_size

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def _allocate_seq_id(self, seq: Sequence):
        seq_id = self.free_seq_ids.popleft()
        self.used_seq_ids.add(seq_id)
        seq.seq_id = seq_id

    def _deallocate_seq_id(self, seq: Sequence):
        if seq.seq_id != -1:
            self.used_seq_ids.remove(seq.seq_id)
            self.free_seq_ids.append(seq.seq_id)
            seq.seq_id = -1

    def schedule(self) -> tuple[list[Sequence], bool]:
        # prefill
        prefilling_seqs = []
        num_batched_tokens = 0
        while self.waiting and len(prefilling_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            if num_batched_tokens + len(
                seq
            ) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
            if len(self.free_seq_ids) == 0:
                if not self.enable_hybrid_engine:
                    break
                if (
                    seq.num_blocks >= self.block_manager.max_blocks_per_seq
                    and self.strict_max_blocks
                ):
                    break
            if len(self.free_seq_ids) > 0:
                self._allocate_seq_id(seq)
            self.block_manager.allocate(seq)
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            prefilling_seqs.append(seq)
        if prefilling_seqs:
            return prefilling_seqs, True

        # decode
        running_seqs = []
        decoding_and_compressing_seqs = []
        while self.running and len(decoding_and_compressing_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            if seq.seq_id == -1 and self.can_allocate_seq_id(seq):
                self._allocate_seq_id(seq)
            rejoining_seqs = []
            if seq.seq_id != -1:
                while not self.block_manager.can_append_or_compress(
                    seq, (not self.enable_hybrid_engine) or self.strict_max_blocks
                ):
                    if self.running and self.enable_hybrid_engine:
                        last_seq = self.running.pop()
                        if not (last_seq.compressed or last_seq.require_compress):
                            self.preempt(last_seq)
                        else:
                            rejoining_seqs.append(last_seq)
                    else:
                        # suspend
                        running_seqs.append(seq)
                        break
                else:
                    self.block_manager.may_append(seq)
                    running_seqs.append(seq)
                    decoding_and_compressing_seqs.append(seq)
            else:
                condition = self.block_manager.can_append(seq, self.strict_max_blocks)
                while condition == -1:
                    if self.running:
                        self.preempt(self.running.pop())
                    else:
                        running_seqs.append(seq)
                        break
                    condition = self.block_manager.can_append(
                        seq, self.strict_max_blocks
                    )
                else:
                    running_seqs.append(seq)
                    if condition == 1:
                        self.block_manager.may_append(seq)
                        decoding_and_compressing_seqs.append(seq)
            self.running.extend(reversed(rejoining_seqs))
        self.running.extendleft(reversed(running_seqs))
        return decoding_and_compressing_seqs, False

    def can_allocate_seq_id(self, seq: Sequence) -> bool:
        return len(self.free_seq_ids) > 0 and (
            len(seq.block_table) < self.block_manager.max_blocks_per_seq
            or seq.last_block_num_tokens <= self.block_size - self.query_cache_len + 1
        )

    def preempt(self, seq: Sequence):
        assert seq.compressed == False and seq.require_compress == False
        seq.status = SequenceStatus.WAITING
        self._deallocate_seq_id(seq)
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
                seq.num_cached_tokens = (
                    len(seq.block_table) - 1
                ) * self.block_size + 1
                seq.compressed = True
                seq.require_compress = False
            if token_ids is not None:
                token_id = token_ids.pop(0)
                seq.append_token(token_id)
                if (
                    not seq.ignore_eos and token_id == self.eos
                ) or seq.num_completion_tokens == seq.max_tokens:
                    seq.status = SequenceStatus.FINISHED
                    self.block_manager.deallocate(seq)
                    self.running.remove(seq)
                    self._deallocate_seq_id(seq)
