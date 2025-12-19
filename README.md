<p align="center">
<img width="400" src="assets/logo.png">
</p>


# Zipage

A high-concurrency offline LLM inference engine specifically optimized for long-output reasoning tasks.

## Key Features

* **High Concurrency** - It builds on PagedAttention and supports KV cache compression. The memory required for each request remains **constant**, thereby sustaining high concurrency.
* **Optimization Suite** - Asynchronous decoding and compression, Prefix caching, Tensor Parallelism, etc.

## TODO

- Online engine.
- Support chunked prefilling.

## Installation




## Quick Start



## Benchmark


## Acknowledgments

We extend our gratitude to the developers of [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm), upon which our project is built.
