from collections import deque
import xxhash
import numpy as np

from zipvllm.engine.sequence import Sequence


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int, max_blocks_per_seq: int = 4):
        self.block_size = block_size
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()
        self.max_blocks_per_seq = max_blocks_per_seq

    @property
    def block_occupancy(self):
        return len(self.used_block_ids) / (
            len(self.used_block_ids) + len(self.free_block_ids)
        )

    def _allocate_block(self, block_id: int):
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return

    def _deallocate_block(self, block_id: int):
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        assert not seq.block_table
        for _ in range(seq.num_blocks):
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def deallocate_block_to_release(self, seq: Sequence):
        for block_id in reversed(seq.block_to_release):
            self._deallocate_block(block_id)
        seq.block_to_release.clear()

    def can_append_or_compress(self, seq: Sequence):
        block_table = seq.block_table
        if len(seq) % self.block_size == 1:
            if len(block_table) < self.max_blocks_per_seq:
                if len(self.free_block_ids) > 0:
                    block_id = self.free_block_ids[0]
                    self._allocate_block(block_id)
                    block_table.append(block_id)
                else:
                    # suspend
                    return False
            else:
                seq.require_compress = True
        return True
