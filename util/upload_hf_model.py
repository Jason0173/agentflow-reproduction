
from huggingface_hub import HfApi

"""
repo_id = "YOUR_ID/YOUR_REPO"
hf_folder_path = "checkpoints/.../actor/huggingface"
"""

api = HfApi(token=YOUR_HF_TOKEN)
api.create_repo(repo_id=repo_id, private=False, exist_ok=True, repo_type="model")
api.upload_folder(folder_path=hf_folder_path, repo_id=repo_id, repo_type="model")