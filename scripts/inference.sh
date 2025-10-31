
MODEL="/local_nvme/liaomengqi/hugingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
# DATASET_PATH="math-ai/aime24"
DATASET_PATH="zwhe99/amc23"


python main.py \
    --model_path $MODEL \
    --dataset $DATASET_PATH \
    --n_sample 1 \
    --system_prompt  "Please reason step by step, and put your final answer within \\boxed{}." \
    --temperature 0.6 \
    --max_tokens 1024 \
    --n_sample 1 \
    --split_len 5