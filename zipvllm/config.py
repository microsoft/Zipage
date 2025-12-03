import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 63840
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    pad: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    # compress config
    max_cache_blocks_per_seq: int = 8
    # query cache config
    query_cache_len: int = 16
    # global score config
    use_global_score: bool = False
    decay_factor: float = 0.6
    # similarity score config
    use_similarity: bool = False
    lightning_similarity: bool = False
    similarity_lambda: float = 0.1
    # pooling config
    enable_pooling: bool = False
    continues_pooling: bool = False
    pooling_size: int = 5
    # engine config
    enable_async_compress: bool = False
    enable_hybrid_engine: bool = False
    strict_max_blocks: bool = False
    enable_prefix_cache: bool = False
    # others
    layer_stride: int = 1
    use_attention_sink: bool = False
    sink_len: int = 4
    repetition_penalty: float = 1.0

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len
        assert self.max_cache_blocks_per_seq >= 1
        assert self.pooling_size % 2 == 1
