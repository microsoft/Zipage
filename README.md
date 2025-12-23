<p align="center">
<img width="400" src="assets/logo.png">
</p>


# Zipage

A high-concurrency offline LLM inference engine specifically optimized for long-output reasoning tasks.

## Key Features

* **High Concurrency** - This project builds on PagedAttention and supports KV cache compression. The memory required for each request remains **constant**, thereby sustaining high concurrency.
* **Optimization Suite** - Asynchronous decoding and compression, Prefix caching, Tensor Parallelism, etc.

## TODO

- Online engine.
- Support chunked prefilling.

## Installation




## Quick Start

```python
from transformers import AutoTokenizer
from zipage import ZipLLM as LLM, SamplingParams


path = './model/qwen3_8b'
llm = LLM(
    path,
    enforce_eager=True,
    gpu_memory_utilization=0.9,
    max_cache_blocks_per_seq=8,
    query_cache_len=16,
    layer_stride=8,
    enable_async_compress=True,
    enable_hybrid_engine=True,
    enable_prefix_cache=True,
    decay_factor=0.8,
    use_global_score=True,
    use_similarity=True,
    lightning_similarity=True,
    similarity_lambda=0.2,
    similarity_temperature=0.4,
    max_num_batched_tokens=32768,
    enable_pooling=True
)
sampling_params = SamplingParams(temperature=0.6, max_tokens=2048)
prompts=['hello, zipage.']
outputs = llm.generate(prompts, sampling_params)
print(outputs['text'])
```

## Benchmark


## Acknowledgments

We extend our gratitude to the developers of [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm), upon which our project is built.
