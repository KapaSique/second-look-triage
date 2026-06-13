"""Deploy the Second Look Gradio demo to a Hugging Face Space.

Run AFTER the user has authenticated (`hf auth login`) and `app/model.pkl` is in place.
Reads the token from the local HF login — this script never handles the token itself.

    python app/deploy_space.py
"""
import os
from huggingface_hub import HfApi

APP_DIR = os.path.dirname(os.path.abspath(__file__))
api = HfApi()
who = api.whoami()
user = who["name"]
repo_id = f"{user}/second-look-triage"
print("authenticated as:", user, "->", repo_id)

api.create_repo(repo_id, repo_type="space", space_sdk="gradio", exist_ok=True)
api.upload_folder(
    folder_path=APP_DIR,
    repo_id=repo_id,
    repo_type="space",
    ignore_patterns=["deploy_space.py", "__pycache__/*", "*.pyc"],
    commit_message="Deploy Second Look triage safety-net demo",
)
print("DONE → https://huggingface.co/spaces/" + repo_id)
