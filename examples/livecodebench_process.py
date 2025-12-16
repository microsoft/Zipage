from huggingface_hub import snapshot_download
import os
import zipfile

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
datasets_dir = os.path.join(project_root, "datasets", "livecodebench")

os.makedirs(datasets_dir, exist_ok=True)

dataset_path = snapshot_download(
    repo_id="livecodebench/code_generation", repo_type="dataset", local_dir=datasets_dir
)

