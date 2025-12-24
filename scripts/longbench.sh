set -x
MODEL="./models/qwen3_8b"
# MODEL="./models/qwen3_14b"

DATASET_NAME="multifieldqa_en"


CUDA_VISIBLE_DEVICES=0 python -m examples.longbench_main \
    --model_path $MODEL \
    --dataset $DATASET_NAME \
    --temperature 0.6 \
    --max_tokens 4096 \
    --max_cache_blocks_per_seq 8 \
    --port 2333 \
    --n_sample 4 \
    --query_cache_len 16 \
    --strict_max_blocks \
    --enable_async_compress \
    --enable_hybrid_engine \
    --use_redudancy \
    --lightning_redudancy \
    --redudancy_lambda 0.2 \
    --redudancy_temperature 0.4 \
    --use_global_score \
    --decay_factor 0.8 \
    --enable_pooling \
    --layer_stride 8 \
    --output_path outputs/long_qwen8b_full.json