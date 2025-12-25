from collections import deque
import threading
import xxhash
import numpy as np

from zipage.engine.sequence import Sequence


class Block:
    def __init__(self, block_id: int):
        self.block_id = block_id
        self.ref_count = 0
        self.token_ids = []
        self.hash = -1

    def update(self, hash: int, token_ids: list[int]):
        self.token_ids = token_ids
        self.hash = hash

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        max_blocks_per_seq: int = 4,
        enable_prefix_cache: bool = False,
    ):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

        self.max_blocks_per_seq = max_blocks_per_seq
        self.enable_prefix_cache = enable_prefix_cache
        self.block_lock = threading.Lock()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    @property
    def block_occupancy(self):
        return len(self.used_block_ids) / (
            len(self.used_block_ids) + len(self.free_block_ids)
        )

    def _allocate_block(self, block_id: int) -> Block:
        with self.block_lock:
            self.free_block_ids.remove(block_id)
            self.used_block_ids.add(block_id)
            block = self.blocks[block_id]
            assert block.ref_count == 0
            block.reset()
        return block

    def _deallocate_block(self, block_id: int):
        with self.block_lock:
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self.used_block_ids.remove(block_id)
                self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        assert not seq.block_table
        h = -1
        cache_miss = False
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            h = (
                self.compute_hash(token_ids, h)
                if (len(token_ids) == self.block_size and self.enable_prefix_cache)
                else -1
            )
            block_id = self.hash_to_block_id.get(h, -1)
            may_save_query = (
                i == seq.num_blocks - 1 and seq.num_blocks >= self.max_blocks_per_seq
            )
            if (
                block_id == -1
                or self.blocks[block_id].token_ids != token_ids
                or may_save_query
            ):
                cache_miss = True
            if cache_miss:
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    block = self._allocate_block(block_id)
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            seq.block_table.append(block_id)

    def deallocate_blocks(self, block_ids: list[int]):
        for block_id in block_ids:
            self._deallocate_block(block_id)

    def deallocate(self, seq: Sequence):
        self.deallocate_blocks(reversed(seq.block_table))
        seq.num_cached_tokens = 0
        seq.prefilled = False
        seq.block_table.clear()

    def deallocate_block_to_release(self, seq: Sequence):
        self.deallocate_blocks(reversed(seq.block_to_release))
        seq.block_to_release.clear()

    def may_compress(self, seq: Sequence) -> bool:
        if seq.compressed:
            seq.new_block_table = seq.block_table[: self.max_blocks_per_seq]
            seq.block_to_release = seq.block_table[self.max_blocks_per_seq :]
            seq.require_compress = True
        else:
            # preserve share prefix
            non_prefix_start = 0
            for block_id in seq.block_table:
                if self.blocks[block_id].ref_count > 1:
                    non_prefix_start += 1
                else:
                    break
            assert len(seq.block_table) - non_prefix_start >= 1
            num_new_blocks = min(non_prefix_start, self.max_blocks_per_seq - 1)
            if num_new_blocks > len(self.free_block_ids):
                return False
            seq.new_block_table = []
            for _ in range(num_new_blocks):
                block_id = self.free_block_ids[0]
                self._allocate_block(block_id)
                seq.new_block_table.append(block_id)
            seq.new_block_table += seq.block_table[
                non_prefix_start : non_prefix_start
                + self.max_blocks_per_seq
                - num_new_blocks
            ]
            assert len(seq.new_block_table) == self.max_blocks_per_seq
            seq.block_to_release = seq.block_table[:non_prefix_start]
            seq.block_to_release += seq.block_table[
                non_prefix_start + self.max_blocks_per_seq - num_new_blocks :
            ]
            for block_id in seq.block_table[
                non_prefix_start : non_prefix_start
                + self.max_blocks_per_seq
                - num_new_blocks
            ]:
                self.blocks[block_id].reset()

            seq.require_compress = True
        return True

    def can_append_or_compress(self, seq: Sequence) -> bool:
        if seq.require_compress:
            # asynchronous compression not finished, this step's compression will be skipped
            return True
        block_table = seq.block_table
        if len(seq) % self.block_size == 1 and (
            seq.num_cached_tokens > self.block_size * len(block_table)
        ):
            if len(block_table) < self.max_blocks_per_seq:
                if not len(self.free_block_ids) > 0:
                    return False
            else:
                return self.may_compress(seq)
        return True

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        if seq.require_compress:
            return
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        if len(seq) % self.block_size == 1 and (
            seq.num_cached_tokens > self.block_size * len(seq.block_table)
        ):
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            seq.block_table.append(block_id)
        elif len(seq) % self.block_size == 0:
            if not seq.compressed:
                token_ids = seq.block(seq.num_blocks - 1)
                prefix = (
                    self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
                )
                h = self.compute_hash(token_ids, prefix)
                last_block.update(h, token_ids)
                self.hash_to_block_id[h] = last_block.block_id
        else:
            assert last_block.hash == -1
