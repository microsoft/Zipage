from collections import deque
import threading
import xxhash
import numpy as np

from zipvllm.engine.sequence import Sequence


class Block:
    def __init__(self, block_id: int):
        self.block_id = block_id
        self.ref_count = 1
        self.token_ids = []
        self.hash = -1

    def update(self, hash: int, token_ids: list[int]):
        self.token_ids = token_ids
        self.hash = hash


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int, max_blocks_per_seq: int = 4):
        self.block_size = block_size
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()
        self.max_blocks_per_seq = max_blocks_per_seq
        self.hash_to_block: dict[int, Block] = dict()
        self.block_id_to_blcok: dict[int, Block] = dict()
        self.ref_lock = threading.Lock()
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

    def _allocate_block(self):
        with self.block_lock:
            block_id = self.free_block_ids[0]
            self.free_block_ids.remove(block_id)
            self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int):
        with self.block_lock:
            self.used_block_ids.remove(block_id)
            self.free_block_ids.append(block_id)

    def _remove_ref(self, block_id: int):
        with self.ref_lock:
            block = self.block_id_to_blcok[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
                del self.block_id_to_blcok[block_id]
                del self.hash_to_block[block.hash]

    def _find_block_add_ref(self, h: int):
        block = None
        with self.ref_lock:
            if h in self.hash_to_block:
                block = self.hash_to_block[h]
                block.ref_count += 1
        return block

    def _add_block(self, block: Block):
        with self.ref_lock:
            self.block_id_to_blcok[block.block_id] = block
            self.hash_to_block[block.hash] = block

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
                if len(token_ids) == self.block_size
                else -1
            )
            block = self._find_block_add_ref(h)
            if block is None:
                cache_miss = True
            elif block is not None and block.token_ids != token_ids:
                # hash collision
                cache_miss = True
                self._remove_ref(block.block_id)
            else:
                block_id = block.block_id
                seq.num_cached_tokens += self.block_size

            if cache_miss:
                block_id = self._allocate_block()

            if h != -1 and block is None:
                block = Block(block_id)
                block.update(h, token_ids)
                self._add_block(block)

            seq.block_table.append(block_id)

    def deallocate_blocks(self, block_ids: list[int]):
        for block_id in block_ids:
            if block_id in self.block_id_to_blcok:
                self._remove_ref(block_id)
            else:
                self._deallocate_block(block_id)

    def deallocate(self, seq: Sequence):
        self.deallocate_blocks(reversed(seq.block_table))
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def deallocate_block_to_release(self, seq: Sequence):
        self.deallocate_blocks(seq.block_to_release)
        seq.block_to_release = None

    def may_compress(self, seq: Sequence) -> bool:
        if seq.compressed:
            seq.new_block_table = (
                seq.block_table[: self.max_blocks_per_seq - 2]
                + [seq.block_table[-1]]
                + [seq.block_table[-2]]
            )
            if len(seq.block_table) > self.max_blocks_per_seq:
                seq.block_to_release = seq.block_table[self.max_blocks_per_seq - 2 : -2]
            else:
                seq.block_to_release = []
            seq.require_compress = True
        else:
            non_prefix_start = 0
            for block_id in seq.block_table:
                if (
                    block_id in self.block_id_to_blcok
                    and self.block_id_to_blcok[block_id].ref_count > 1
                ):
                    non_prefix_start += 1
                else:
                    break
            non_prefix_blocks = len(seq.block_table) - non_prefix_start
            assert non_prefix_blocks >= 1
            num_new_blocks = min(non_prefix_start, self.max_blocks_per_seq - 2)
            if num_new_blocks + (non_prefix_blocks == 1) >= len(self.free_block_ids):
                return False
            # new block table after compression
            seq.new_block_table = []
            for _ in range(num_new_blocks):
                block_id = self._allocate_block()
                seq.new_block_table.append(block_id)
            if num_new_blocks < self.max_blocks_per_seq - 2:
                seq.new_block_table += seq.block_table[
                    non_prefix_start : self.max_blocks_per_seq - 2
                ]
            seq.new_block_table.append(seq.block_table[-1])
            if non_prefix_blocks == 1:
                block_id = self._allocate_block()
                seq.new_block_table.append(block_id)
            else:
                seq.new_block_table.append(seq.block_table[-2])
            # block to release after compression
            seq.block_to_release = seq.block_table[:non_prefix_start]
            if non_prefix_blocks == 1:
                seq.block_to_release += seq.block_table[
                    max(non_prefix_start, self.max_blocks_per_seq - 2) : -1
                ]
            else:
                seq.block_to_release += seq.block_table[
                    max(non_prefix_start, self.max_blocks_per_seq - 2) : -2
                ]
            seq.require_compress = True
        return True

    def can_append_or_compress(self, seq: Sequence, strict: bool = False) -> bool:
        if seq.require_compress:
            # asynchronous compression not finished, this step's compression will be skipped
            return True
        block_table = seq.block_table
        if len(seq) % self.block_size == 1 and (
            seq.num_cached_tokens > self.block_size * len(block_table)
        ):
            if strict:
                if len(block_table) < self.max_blocks_per_seq:
                    if not len(self.free_block_ids) > 0:
                        # suspend
                        return False
                else:
                    return self.may_compress(seq)
            else:
                if not len(self.free_block_ids) > 0:
                    if len(block_table) < self.max_blocks_per_seq:
                        return False
                    else:
                        return self.may_compress(seq)
        return True

    def can_append(self, seq: Sequence, strict: bool = False):
        if len(seq) % self.block_size == 1:
            if strict:
                return (
                    len(self.free_block_ids) > 0
                    and len(seq.block_table) < self.max_blocks_per_seq - 1
                )
            else:
                return len(self.free_block_ids) > 0
        return True

    def may_append(self, seq: Sequence):
        if len(seq) % self.block_size == 1 and (
            seq.num_cached_tokens > self.block_size * len(seq.block_table)
        ):
            if not seq.require_compress:
                block_id = self._allocate_block()
                seq.block_table.append(block_id)
