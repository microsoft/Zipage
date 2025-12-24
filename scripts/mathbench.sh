set -x
MODEL="./models/qwen3_8b"
# MODEL="./models/qwen3_14b"

DATASET_PATH="math-ai/aime24"
# DATASET_PATH="juanlrdc/gmsk8"
# DATASET_PATH="mixed"
# DATASET_PATH="zwhe99/amc23"


CUDA_VISIBLE_DEVICES=1 python -m examples.mathbench_main \
    --model_path $MODEL \
    --dataset $DATASET_PATH \
    --system_prompt  "Please reason step by step, and put your final answer within \\boxed{}." \
    --temperature 0.6 \
    --max_tokens 32768 \
    --tensor_parallel_size 1 \
    --max_cache_blocks_per_seq 8 \
    --port 2334 \
    --n_sample 32 \
    --query_cache_len 16 \
    --enable_hybrid_engine \
    --enable_prefix_cache \
    --enable_async_compress \
    --use_redudancy \
    --lightning_redudancy \
    --redudancy_lambda 0.2 \
    --redudancy_temperature 0.4 \
    --use_global_score \
    --decay_factor 0.8 \
    --enable_pooling \
    --layer_stride 8 \
    --compress \
    --output_path outputs/single_qwen8b_zipage.json