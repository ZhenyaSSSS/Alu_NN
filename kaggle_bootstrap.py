import os


def load_wandb_key(secret_name: str = "WANDB_API_KEY") -> bool:
    try:
        from kaggle_secrets import UserSecretsClient
    except ImportError:
        return False
    try:
        raw = UserSecretsClient().get_secret(secret_name)
    except Exception:
        return False
    key = (raw or "").strip()
    if not key:
        return False
    os.environ["WANDB_API_KEY"] = key
    return True
