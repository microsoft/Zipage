from huggingface_hub import snapshot_download
import os
import zipfile

current_dir = os.path.dirname(os.path.abspath(__file__))
current_dir = os.path.join(current_dir, "longbench")
dataset_path = snapshot_download(
    repo_id="THUDM/LongBench", repo_type="dataset", local_dir=current_dir
)

data_zip_path = os.path.join(current_dir, "data.zip")

with zipfile.ZipFile(data_zip_path, "r") as zip_ref:
    zip_ref.extractall(current_dir)
