import os
from pathlib import Path
import sys
root = Path(__file__).resolve().parents[1] / ".artifacts/model"
version = os.environ["RELEASE_TAG"]
if sys.argv[1] == "hf":
    from huggingface_hub import HfApi
    repo = os.environ["HF_REPO_ID"]
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo_id=repo, repo_type="model", exist_ok=True)
    commit = api.upload_folder(repo_id=repo, folder_path=root, commit_message=f"Release {version}", ignore_patterns=["__pycache__/*", "*.pyc"])
    api.create_tag(repo_id=repo, tag=version, revision=commit.oid, exist_ok=True)
elif sys.argv[1] == "ms":
    from modelscope.hub.api import HubApi
    repo = os.environ["MODELSCOPE_REPO_ID"]
    api = HubApi()
    api.login(os.environ["MODELSCOPE_API_KEY"])
    api.create_repo(repo_id=repo, repo_type="model", visibility="public", license="MIT", exist_ok=True)
    api.upload_folder(repo_id=repo, repo_type="model", folder_path=str(root), commit_message=f"Release {version}", ignore_patterns=["__pycache__", "*.pyc"])
    api.create_model_tag(model_id=repo, tag_name=version)
else:
    raise SystemExit("Expected hf or ms")
