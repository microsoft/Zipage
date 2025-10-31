from collections import deque

from zipvllm.config import Config
from zipvllm.engine.sequence import Sequence, SequenceStatus
from zipvllm.engine.block_manager import BlockManager
from typing import Optional


class Scheduler:
    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(
            config.num_kvcache_blocks,
            config.kvcache_block_size,
            config.max_blocks_per_seq,
        )
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.free_seq_ids: deque[int] = deque(range(config.max_num_seqs))
        self.used_seq_ids: set[int] = set()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def _allocate_seq_id(self, seq: Sequence):
        seq_id = self.free_seq_ids.popleft()
        self.used_seq_ids.add(seq_id)
        seq.seq_id = seq_id

    def _deallocate_seq_id(self, seq: Sequence):
        self.used_seq_ids.remove(seq.seq_id)
        self.free_seq_ids.append(seq.seq_id)

    def schedule(self) -> tuple[list[Sequence], bool]:
        # prefill
        prefilling_seqs = []
        num_batched_tokens = 0
        while self.waiting and len(self.free_seq_ids) > 0:
            seq = self.waiting[0]
            if num_batched_tokens + len(
                seq
            ) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
            self._allocate_seq_id(seq)
            self.block_manager.allocate(seq)
            num_batched_tokens += len(seq)
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            prefilling_seqs.append(seq)
        if prefilling_seqs:
            return prefilling_seqs, True

        # decode
        decoding_seqs = []
        for seq in self.running:
            if self.block_manager.can_append_or_compress(seq):
                decoding_seqs.append(seq)
        return decoding_seqs, False

    def postprocess(
        self,
        seqs: list[Sequence],
        token_ids: list[int],
        entropies: Optional[list[float]] = None,
    ) -> list[bool]:
        for seq, token_id in zip(seqs, token_ids):
            if entropies is not None:
                seq.last_token_entropy = entropies.pop(0)
            seq.append_token(token_id)
            self.block_manager.deallocate_block_to_release(seq)
            seq.require_compress = False
            if (
                not seq.ignore_eos and token_id == self.eos
            ) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
                self._deallocate_seq_id(seq)
