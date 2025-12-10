set -x
# MODEL="/local_nvme/liaomengqi/hugingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
MODEL="/local_nvme/liaomengqi/hugingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
# MODEL="/local_nvme/liaomengqi/hugingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/916b56a44061fd5cd7d6a8fb632557ed4f724f60"
# MODEL="/local_nvme/liaomengqi/hugingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Llama-8B/snapshots/6a6f4aa4197940add57724a7707d069478df56b1"


DATASET_NAME="multifieldqa_en"


CUDA_VISIBLE_DEVICES=2 python -m bench.longbench_main \
    --model_path $MODEL \
    --dataset $DATASET_NAME \
    --temperature 0.6 \
    --max_tokens 4096 \
    --max_cache_blocks_per_seq 8 \
    --port 2336 \
    --n_sample 4 \
    --query_cache_len 16 \
    --strict_max_blocks \
    --enable_async_compress \
    --enable_hybrid_engine \
    --use_similarity \
    --lightning_similarity \
    --similarity_lambda 0.2 \
    --similarity_temperature 0.4 \
    --use_global_score \
    --decay_factor 0.8 \
    --enable_pooling \
    --layer_stride 8 \
    --compress \
    --output_path outputs/test_qwen8b_non_shared_longbench.json