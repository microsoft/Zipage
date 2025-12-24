<p align="center">
<img width="400" src="assets/logo.png">
</p>


# Zipage

A high-concurrency LLM inference engine.

## Key Features

* **High Concurrency** - This project builds on PagedAttention and supports KV cache compression. The memory required for each request remains **constant**, thereby sustaining high concurrency.
* **Optimization Suite** - Asynchronous decoding and compression, Prefix caching, Tensor Parallelism, etc.

TODO

- [ ] Online engine.
- [ ] Support chunked prefilling.
- [ ] Adaptive KV cache budget.
- [ ] More sampling algorithms.
- [ ] Support VLM.

## Installation


```
git clone https://xxx.git
cd zipage
pip install -e .
```



## Quick Start

```python
from transformers import AutoTokenizer
from zipage import ZipLLM as LLM, SamplingParams


path = './model/qwen3_8b'
llm = LLM(
    path,
    enforce_eager=True,
    gpu_memory_utilization=0.9,
    max_num_batched_tokens=32768,
    max_cache_blocks_per_seq=8,
    enable_async_compress=True,
    enable_hybrid_engine=True,
    enable_prefix_cache=True,
    use_global_score=True,
    use_similarity=True,
    lightning_similarity=True,
    enable_pooling=True
)
sampling_params = SamplingParams(temperature=0.6, max_tokens=2048)
prompts=['hello, zipage.']
outputs = llm.generate(prompts, sampling_params)
print(outputs['text'])
```

## Benchmark

Evaluate on a math benchmark

```shell
bash scripts/mathbench.sh
```

Evaluate on LongBench

```shell
python examples/longbench_process.py
bash scripts/longbench.sh
```

<!-- ## Acknowledgments

We extend our gratitude to the developers of [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm), upon which our project is built. -->
