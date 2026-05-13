import argparse
import os
import random

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset

import config
from engine import NLAEngine


def _num_workers() -> int:
    w = int(getattr(config, "DATALOADER_NUM_WORKERS", 0))
    if w <= 0:
        return 0
    cpu = os.cpu_count() or 1
    return min(w, max(1, cpu - 1))


class DummyDataset(Dataset):
    def __len__(self):
        return config.STEPS_PER_EPOCH

    def __getitem__(self, idx):
        return torch.tensor(0)


def _accelerator_name() -> str:
    env = os.environ.get("PL_ACCELERATOR", "").strip().lower()
    if env in ("gpu", "tpu", "cpu"):
        return env
    a = str(getattr(config, "ACCELERATOR", "gpu")).lower().strip()
    return a if a in ("gpu", "tpu", "cpu") else "gpu"


def _is_tpu_run() -> bool:
    return _accelerator_name() == "tpu"


def _pl_devices_value():
    ev = os.environ.get("PL_DEVICES", "").strip()
    if ev.isdigit():
        return int(ev)
    w = getattr(config, "PL_DEVICES", 1)
    try:
        return int(w) if not isinstance(w, bool) else 1
    except (TypeError, ValueError):
        return 1


def fix_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    pl.seed_everything(seed, workers=True)


def _maybe_compile(module: torch.nn.Module) -> torch.nn.Module:
    if not getattr(config, "USE_TORCH_COMPILE", False) or not hasattr(torch, "compile"):
        return module
    try:
        return torch.compile(
            module,
            mode=getattr(config, "TORCH_COMPILE_MODE", "max-autotune"),
            fullgraph=getattr(config, "TORCH_COMPILE_FULLGRAPH", False),
            dynamic=getattr(config, "TORCH_COMPILE_DYNAMIC", False),
        )
    except Exception as e:
        print(f"[train] torch.compile disabled: {e}")
        return module


def _checkpoint_dir(project_root: str) -> str:
    d = os.environ.get("ALU_CHECKPOINT_DIR", "").strip()
    if not d:
        c = getattr(config, "CHECKPOINT_DIR", None)
        if isinstance(c, str) and c.strip():
            d = c.strip()
    if not d:
        d = os.path.join(project_root, "checkpoints")
    else:
        d = os.path.normpath(os.path.expanduser(d))
    os.makedirs(d, exist_ok=True)
    return d


def _resolve_checkpoint_path(ckpt_path: str | None, project_root: str) -> str | None:
    if not ckpt_path or ckpt_path in ("last", "hpc"):
        return ckpt_path
    p = os.path.normpath(os.path.expanduser(ckpt_path))
    if os.path.isfile(p):
        return p
    base = _checkpoint_dir(project_root)
    cand = os.path.join(base, os.path.basename(p))
    if os.path.isfile(cand):
        print(f"[train] checkpoint: {cand}")
        return cand
    proj = os.path.basename(os.path.normpath(project_root))
    try:
        rel = os.path.relpath(p, project_root)
    except ValueError:
        rel = ""
    parts = rel.split(os.sep)
    if len(parts) == 3 and parts[1] == "checkpoints":
        run_id, _, fname = parts[0], parts[1], parts[2]
        alt = os.path.join(project_root, proj, run_id, "checkpoints", fname)
        if os.path.isfile(alt):
            print(f"[train] checkpoint: {alt}")
            return alt
    raise FileNotFoundError(
        f"Checkpoint not found: {ckpt_path!r}\n"
        f"  Tried: {p!r}\n"
        f"  If W&B saved under project subfolder, use e.g.:\n"
        f"  {os.path.join(project_root, proj, '<run_id>', 'checkpoints', os.path.basename(p))}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None, metavar="CKPT")
    parser.add_argument("--accelerator", type=str, default=None, metavar="gpu|tpu|cpu")
    args = parser.parse_args()
    if args.accelerator is not None:
        a = args.accelerator.lower().strip()
        if a not in ("gpu", "tpu", "cpu"):
            raise SystemExit("--accelerator must be one of: gpu, tpu, cpu")
        os.environ["PL_ACCELERATOR"] = a
    _project_root = os.path.dirname(os.path.abspath(__file__))
    raw_ckpt = args.resume or os.environ.get("RESUME_CKPT")
    try:
        ckpt_path = _resolve_checkpoint_path(raw_ckpt, _project_root)
    except FileNotFoundError as e:
        raise SystemExit(str(e)) from e
    if ckpt_path:
        print(f"[train] resume ckpt_path={ckpt_path!r}")

    print(f"[train] accelerator={_accelerator_name()}")

    if not os.environ.get("WANDB_API_KEY", "").strip():
        os.environ.setdefault("WANDB_MODE", "offline")
        print("[train] WANDB_API_KEY unset → wandb offline (see kaggle_bootstrap)")

    if (
        hasattr(torch, "set_float32_matmul_precision")
        and torch.cuda.is_available()
        and not _is_tpu_run()
    ):
        torch.set_float32_matmul_precision(getattr(config, "FLOAT32_MATMUL_PRECISION", "high"))

    fix_seeds(config.SEED)

    nw = (
        int(getattr(config, "DATALOADER_NUM_WORKERS_TPU", 0))
        if _is_tpu_run()
        else _num_workers()
    )
    if nw < 0:
        nw = 0
    train_loader = DataLoader(
        DummyDataset(),
        batch_size=1,
        num_workers=nw,
        persistent_workers=nw > 0,
        generator=torch.Generator().manual_seed(config.SEED),
    )
    val_loader = DataLoader(
        DummyDataset(),
        batch_size=1,
        num_workers=nw,
        persistent_workers=nw > 0,
        generator=torch.Generator().manual_seed(config.SEED + 1),
    )

    model = NLAEngine()
    compile_ok = bool(getattr(config, "USE_TORCH_COMPILE", False)) and (
        not _is_tpu_run() or bool(getattr(config, "USE_TORCH_COMPILE_ON_TPU", False))
    )
    if getattr(config, "USE_TORCH_COMPILE", False) and _is_tpu_run() and not getattr(
        config, "USE_TORCH_COMPILE_ON_TPU", False
    ):
        print("[train] torch.compile skipped on TPU (set USE_TORCH_COMPILE_ON_TPU=True to try)")
    if compile_ok:
        model.model = _maybe_compile(model.model)

    wandb_logger = WandbLogger(
        project=config.WANDB_PROJECT,
        name=config.WANDB_RUN_NAME,
        entity=os.environ.get("WANDB_ENTITY"),
        log_model=config.WANDB_LOG_MODEL,
    )

    _root = _project_root
    _ckpt_dir = _checkpoint_dir(_root)
    print(f"[train] checkpoints dir={_ckpt_dir!r}")

    checkpoint_callback = ModelCheckpoint(
        dirpath=_ckpt_dir,
        monitor="val_ema",
        mode="max",
        save_top_k=3,
        save_last=True,
        filename="nla-{epoch:02d}-{val_ema:.4f}",
    )

    acc = _accelerator_name()
    devices = _pl_devices_value()
    if acc == "tpu":
        precision = os.environ.get("PL_PRECISION_TPU", "").strip() or str(
            getattr(config, "PL_PRECISION_TPU", "bf16-true")
        )
    elif acc == "gpu":
        precision = os.environ.get("PL_PRECISION_GPU", "").strip() or str(
            getattr(config, "PL_PRECISION_GPU", "bf16-mixed")
        )
    else:
        precision = "32-true"

    print(f"[train] Trainer: accelerator={acc!r} devices={devices!r} precision={precision!r}")

    trainer = pl.Trainer(
        max_epochs=config.EPOCHS,
        accelerator=acc,
        devices=devices,
        precision=precision,
        gradient_clip_val=config.GRAD_CLIP,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, LearningRateMonitor(logging_interval="step")],
        log_every_n_steps=1,
        val_check_interval=0.5,
        deterministic=False,
        limit_val_batches=getattr(config, "VAL_LIMIT_BATCHES", 50),
    )

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=ckpt_path,
    )
