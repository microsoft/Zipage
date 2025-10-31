from collections import deque
import xxhash
import numpy as np

from zipvllm.engine.sequence import Sequence


<<<<<<< HEAD
class Block:
    def __init__(self, block_id: int):
        self.block_id = block_id
        self.ref_count = 1
        self.token_ids = []
        self.hash = -1

    def update(self, hash: int, token_ids: list[int]):
        self.token_ids = token_ids
        self.hash = hash


=======
>>>>>>> 2aaa790 (init commit)
class BlockManager:

    def __init__(self, num_blocks: int, block_size: int, max_blocks_per_seq: int = 4):
        self.block_size = block_size
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()
        self.max_blocks_per_seq = max_blocks_per_seq
<<<<<<< HEAD
        self.hash_to_block: dict[int, Block] = dict()
        self.block_id_to_blcok: dict[int, Block] = dict()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()
=======
>>>>>>> 2aaa790 (init commit)

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
<<<<<<< HEAD
        h = -1
        cache_miss = False
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            h = (
                self.compute_hash(token_ids, h)
                if len(token_ids) == self.block_size
                else -1
            )
            block = self.hash_to_block.get(h, None)
            if block is None or block.token_ids != token_ids:
                cache_miss = True
                block_id = -1
            else:
                block_id = block.block_id
                seq.num_cached_tokens += self.block_size

            if cache_miss:
                block_id = self.free_block_ids[0]
                self._allocate_block(block_id)
                block = Block(block_id)
            else:
                block.ref_count += 1

            if h != -1:
                self.block_id_to_blcok[block_id] = block
                self.hash_to_block[h] = block
                block.update(h, token_ids)
=======
        for _ in range(seq.num_blocks):
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
>>>>>>> 2aaa790 (init commit)
            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
<<<<<<< HEAD
            if block_id in self.block_id_to_blcok:
                block = self.block_id_to_blcok[block_id]
                block.ref_count -= 1
                if block.ref_count == 0:
                    self._deallocate_block(block_id)
                    del self.block_id_to_blcok[block_id]
                    del self.hash_to_block[block.hash]
=======
            self._deallocate_block(block_id)
>>>>>>> 2aaa790 (init commit)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def deallocate_block_to_release(self, seq: Sequence):
        for block_id in reversed(seq.block_to_release):
<<<<<<< HEAD
            if block_id in self.block_id_to_blcok:
                block = self.block_id_to_blcok[block_id]
                block.ref_count -= 1
                if block.ref_count == 0:
                    self._deallocate_block(block_id)
                    del self.block_id_to_blcok[block_id]
                    del self.hash_to_block[block.hash]
            else:
                self._deallocate_block(block_id)
        seq.block_to_release = None

    def may_compress(self, seq: Sequence) -> bool:
        if seq.require_compress:
            return True
        if seq.compressed:
            seq.new_block_table = (
                seq.block_table[: self.max_blocks_per_seq - 2]
                + [seq.block_table[-1]]
                + [seq.block_table[self.max_blocks_per_seq - 2]]
            )
            if len(seq.block_table) > self.max_blocks_per_seq:
                seq.block_to_release = seq.block_table[self.max_blocks_per_seq - 1 : -1]
            seq.require_compress = True
        else:
            prefix_cache_block_ids = []
            compressable_block_ids = []
            for block_id in seq.block_table:
                if block_id in self.block_id_to_blcok:
                    block = self.block_id_to_blcok[block_id]
                    if block.ref_count > 1:
                        prefix_cache_block_ids.append(block_id)
                else:
                    compressable_block_ids.append(block_id)
            assert len(compressable_block_ids) > 0

            num_new_blocks = min(
                len(prefix_cache_block_ids), self.max_blocks_per_seq - 2
            )
            if num_new_blocks + (len(compressable_block_ids) <= 1) > len(
                self.free_block_ids
            ):
                return False
            seq.new_block_table = []
            for i in range(num_new_blocks):
                block_id = self.free_block_ids[0]
                self._allocate_block(block_id)
                seq.new_block_table.append(block_id)
            seq.new_block_table += compressable_block_ids[
                : self.max_blocks_per_seq - 2 - num_new_blocks
            ]
            seq.block_to_release = compressable_block_ids[num_new_blocks:]
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
                    seq.require_compress = True
            else:
                if not len(self.free_block_ids) > 0:
                    if len(block_table) < self.max_blocks_per_seq:
                        return False
                    else:
                        seq.require_compress = True
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
                block_id = self.free_block_ids[0]
                self._allocate_block(block_id)
                seq.block_table.append(block_id)
                if not seq.compressed:
                    self.block_id_to_blcok[block_id] = Block(block_id)
=======
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
>>>>>>> 2aaa790 (init commit)
