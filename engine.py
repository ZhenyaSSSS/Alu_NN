import math
import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.trainer.states import TrainerFn

try:
    from schedulefree import AdamWScheduleFree, RAdamScheduleFree
except ImportError:
    AdamWScheduleFree = None  # type: ignore[misc, assignment]
    RAdamScheduleFree = None  # type: ignore[misc, assignment]

_SCHEDULE_FREE_TYPES = tuple(t for t in (AdamWScheduleFree, RAdamScheduleFree) if t is not None)


def _is_schedule_free_optimizer(opt) -> bool:
    return bool(_SCHEDULE_FREE_TYPES) and isinstance(opt, _SCHEDULE_FREE_TYPES)


from architecture import NLA
import config
from data_gen import generate_batch
from bit_utils import bits_to_float


def _special_float_accuracy(target_f: torch.Tensor, pred_f: torch.Tensor) -> torch.Tensor:
    m = ~torch.isfinite(target_f)
    if not m.any():
        return torch.tensor(float("nan"), device=target_f.device, dtype=torch.float32)
    t, p = target_f[m], pred_f[m]
    ok = (torch.isnan(t) & torch.isnan(p)) | (torch.isposinf(t) & torch.isposinf(p)) | (
        torch.isneginf(t) & torch.isneginf(p)
    )
    return ok.float().mean()


def _mape_finite(target_f: torch.Tensor, pred_f: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    fin = torch.isfinite(target_f) & torch.isfinite(pred_f)
    nz = target_f.abs() > eps
    m = fin & nz
    if not m.any():
        return torch.tensor(float("nan"), device=target_f.device, dtype=torch.float32)
    return ((pred_f[m] - target_f[m]).abs() / target_f[m].abs().clamp_min(eps)).mean()


def _serializable_config() -> dict:
    out = {}
    for k, v in vars(config).items():
        if not k.isupper() or k.startswith("_"):
            continue
        if callable(v):
            continue
        if isinstance(v, torch.Tensor):
            out[k] = v.detach().cpu().tolist() if v.numel() <= 128 else f"Tensor{v.shape}"
        elif isinstance(v, dict):
            out[k] = {
                str(a): (list(b) if isinstance(b, tuple) else b) for a, b in v.items()
            }
        else:
            out[k] = v
    return out


def sota_swd_loss(z: torch.Tensor) -> torch.Tensor:
    device = z.device
    zf = z.float()
    b, d = zf.shape
    k = config.SWD_NUM_PROJECTIONS

    mean_loss = zf.mean(dim=0).pow(2).mean()
    var_loss = (zf.var(dim=0, unbiased=False) - 1.0).pow(2).mean()

    projections = torch.randn(d, k, device=device, dtype=torch.float32)
    projections = projections / projections.norm(dim=0, keepdim=True)

    z_proj = torch.matmul(zf, projections).sort(dim=0)[0]

    p = torch.linspace(0.5 / b, 1.0 - 0.5 / b, b, device=device, dtype=torch.float32).unsqueeze(1)
    t = (2.0 * p - 1.0).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    q = torch.erfinv(t) * math.sqrt(2.0)
    target_proj = q.expand(-1, k)

    swd = F.mse_loss(z_proj, target_proj)
    return swd + config.SWD_MOMENT_WEIGHT * (mean_loss + var_loss)


def masked_infonce_loss(
    pred: torch.Tensor, target: torch.Tensor, bits_target: torch.Tensor, tau: Optional[float] = None
):
    if tau is None:
        tau = float(getattr(config, "INFO_NCE_TAU", 0.1))
    pred_f = pred.float()
    target_f = target.float()
    pred_norm = F.normalize(pred_f, dim=-1)
    target_norm = F.normalize(target_f, dim=-1)

    sim = torch.matmul(pred_norm, target_norm.T) / tau

    shifts = torch.arange(31, -1, -1, device=bits_target.device, dtype=torch.int32)
    compact = (bits_target.to(torch.int32) << shifts).sum(dim=-1, dtype=torch.int32)
    mask = (compact.unsqueeze(1) == compact.unsqueeze(0)).float()

    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    loss = -(mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
    return loss.mean()


def _param_groups_weight_decay(model: nn.Module, wd: float):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or name.endswith(".norm.weight") or "bit_embed" in name or "op_table" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": wd},
        {"params": no_decay, "weight_decay": 0.0},
    ]


class NLAEngine(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = NLA(config)
        self.register_buffer("bit_weights", config.get_bit_weights())
        self.automatic_optimization = True

    def _synthetic_batch_size(self) -> int:
        tr = self.trainer
        if tr is None:
            return int(getattr(config, "BATCH_SIZE", 16384))
        acc = getattr(tr, "accelerator", None)
        name = type(acc).__name__ if acc is not None else ""
        if "TPU" in name or "XLA" in name:
            ev = os.environ.get("ALU_BATCH_SIZE_TPU", "").strip()
            if ev.isdigit():
                return max(1, int(ev))
            return int(getattr(config, "BATCH_SIZE_TPU_PER_CORE", config.BATCH_SIZE))
        return int(config.BATCH_SIZE)

    def _optimizer_lr(self) -> float:
        tr = self.trainer
        base = float(getattr(config, "LR", 1e-3))
        if tr is None:
            return base
        acc = getattr(tr, "accelerator", None)
        name = type(acc).__name__ if acc is not None else ""
        if "TPU" not in name and "XLA" not in name:
            return base
        ev = os.environ.get("ALU_LR_TPU", "").strip()
        if ev:
            try:
                return float(ev)
            except ValueError:
                pass
        return float(getattr(config, "LR_TPU", base))

    def _optimizer0(self):
        try:
            opt = self.optimizers()
        except Exception:
            return None
        if isinstance(opt, (list, tuple)):
            opt = opt[0]
        return opt

    def _schedule_free_train(self) -> None:
        opt = self._optimizer0()
        if _is_schedule_free_optimizer(opt):
            opt.train()

    def _schedule_free_eval(self) -> None:
        opt = self._optimizer0()
        if _is_schedule_free_optimizer(opt):
            opt.eval()

    def on_train_start(self):
        if isinstance(self.logger, WandbLogger):
            self.logger.experiment.config.update(_serializable_config(), allow_val_change=True)
            meta: dict = {"torch_version": torch.__version__}
            tr = self.trainer
            if tr is not None:
                acc_cls = type(tr.accelerator).__name__
                meta["lightning_accelerator"] = acc_cls
                if "TPU" in acc_cls or "XLA" in acc_cls:
                    try:
                        import torch_xla.core.xla_model as xm

                        meta["xla_device"] = str(xm.xla_device())
                    except Exception:
                        meta["xla_device"] = "n/a"
                elif torch.cuda.is_available():
                    meta["cuda_version"] = torch.version.cuda
                    meta["cuda_device"] = torch.cuda.get_device_name(torch.cuda.current_device())
            elif torch.cuda.is_available():
                meta["cuda_version"] = torch.version.cuda
                meta["cuda_device"] = torch.cuda.get_device_name(torch.cuda.current_device())
            self.logger.experiment.config.update(meta, allow_val_change=True)
            self.logger.watch(
                self.model,
                log=config.WANDB_WATCH,
                log_freq=config.WANDB_WATCH_LOG_FREQ,
                log_graph=getattr(config, "WANDB_LOG_GRAPH", False),
            )
        self._schedule_free_train()
        if self.trainer is not None and self.trainer.global_rank == 0:
            bs = self._synthetic_batch_size()
            ws = self.trainer.world_size
            lr = self._optimizer_lr()
            print(f"[engine] batch/core={bs} cores={ws} global_samples/step≈{bs * ws} lr={lr}")

    def on_train_batch_start(self, batch, batch_idx):
        self._schedule_free_train()

    def on_validation_start(self):
        self._schedule_free_eval()

    def on_validation_end(self):
        if self.trainer.state.fn == TrainerFn.FITTING:
            self._schedule_free_train()

    def on_test_start(self):
        self._schedule_free_eval()

    def on_predict_start(self):
        self._schedule_free_eval()

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        opt = self._optimizer0()
        if not _is_schedule_free_optimizer(opt):
            return
        opt.eval()
        if self.trainer.state.fn == TrainerFn.FITTING:
            opt.train()

    def on_fit_end(self):
        self._schedule_free_eval()

    def forward(self, bits_A, bits_B, op_ids):
        A_enc = self.model.encoder(bits_A)
        B_enc = self.model.encoder(bits_B)
        z = self.model.solver(A_enc, B_enc, op_ids)
        return self.model.decoder(z)

    def _add_latent_noise(self, z: torch.Tensor):
        noise = torch.randn_like(z) * config.NOISE_LEVEL * z.std(dim=-1, keepdim=True)
        return z + noise

    def training_step(self, batch, batch_idx):
        bs = self._synthetic_batch_size()
        bits_A, bits_B, op_ids, bits_Target = generate_batch(bs, self.device)

        enc_A = self.model.encoder(bits_A)
        enc_B = self.model.encoder(bits_B)
        enc_Target = self.model.encoder(bits_Target)

        enc_A_noisy = self._add_latent_noise(enc_A)
        enc_B_noisy = self._add_latent_noise(enc_B)

        pred_z = self.model.solver(enc_A_noisy, enc_B_noisy, op_ids)

        logits_clean = self.model.decoder(enc_Target)
        logits_solver = self.model.decoder(pred_z)

        target_f32 = bits_Target.float()
        logits_clean_f32 = logits_clean.float()
        logits_solver_f32 = logits_solver.float()
        bce_clean = F.binary_cross_entropy_with_logits(logits_clean_f32, target_f32, reduction="none")
        bce_solver = F.binary_cross_entropy_with_logits(logits_solver_f32, target_f32, reduction="none")

        loss_bce = (bce_clean * self.bit_weights).mean() + config.BETA_DECODER * (
            bce_solver * self.bit_weights
        ).mean()

        if getattr(config, "USE_INFO_NCE", True):
            loss_info = masked_infonce_loss(pred_z, enc_Target, bits_Target)
        else:
            loss_info = pred_z.new_zeros(())

        loss_latent_reg = F.smooth_l1_loss(pred_z.float(), enc_Target.detach().float())

        loss_wae = sota_swd_loss(enc_Target)

        loss = (
            config.LAMBDAS["bce"] * loss_bce
            + (config.LAMBDAS["info_nce"] * loss_info if getattr(config, "USE_INFO_NCE", True) else 0.0)
            + config.LAMBDAS["latent_reg"] * loss_latent_reg
            + config.LAMBDAS["wae"] * loss_wae
        )

        bce_clean_mean = (bce_clean * self.bit_weights).mean()
        bce_solver_mean = (bce_solver * self.bit_weights).mean()

        opt = self.optimizers()
        if isinstance(opt, (list, tuple)):
            opt = opt[0]
        pg0 = opt.param_groups[0]
        lr_log = pg0.get("scheduled_lr", pg0["lr"])

        self.log_dict(
            {
                "train/loss": loss,
                "train/loss_total": loss,
                "train/bce": loss_bce,
                "train/bce_clean": bce_clean_mean,
                "train/bce_solver": bce_solver_mean,
                "train/info_nce": loss_info,
                "train/latent_reg": loss_latent_reg,
                "train/wae_swd": loss_wae,
                "train/lr": lr_log,
            },
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        bs = self._synthetic_batch_size()
        bits_A, bits_B, op_ids, bits_Target = generate_batch(bs, self.device)

        logits = self(bits_A, bits_B, op_ids)
        pred_bits = (logits > 0).long()

        is_correct = (pred_bits == bits_Target).all(dim=-1).float()
        ema = is_correct.mean()

        target_f = bits_to_float(bits_Target)
        pred_f = bits_to_float(pred_bits)

        nan_target = torch.isnan(target_f)
        nan_pred = torch.isnan(pred_f)
        nan_acc = (nan_target == nan_pred).float().mean()

        special_acc = _special_float_accuracy(target_f, pred_f)
        mape_fin = _mape_finite(target_f, pred_f)

        metrics = {
            "val_ema": ema,
            "val/exact_match_bits": ema,
            "val_nan_acc": nan_acc,
            "val/special_float_acc": special_acc,
            "val/mape_finite": mape_fin,
            "val_loss_proxy": 1.0 - ema,
        }
        num_ops = config.NUM_OPS
        oh = F.one_hot(op_ids, num_classes=num_ops).to(is_correct.dtype)
        correct = is_correct.unsqueeze(1)
        num = (correct * oh).sum(dim=0)
        den = oh.sum(dim=0)
        acc_per_op = torch.where(
            den > 0,
            num / den,
            torch.full((num_ops,), float("nan"), device=is_correct.device, dtype=is_correct.dtype),
        )
        for op_name, (op_id, _) in config.OPERATIONS_MAP.items():
            metrics[f"val_acc/{op_name}"] = acc_per_op[op_id]
        logged = {}
        for k, v in metrics.items():
            if torch.is_tensor(v) and v.dim() == 0 and not torch.isfinite(v):
                continue
            logged[k] = v
        self.log_dict(
            logged,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            logger=True,
        )

    def configure_optimizers(self):
        param_groups = _param_groups_weight_decay(self.model, config.WEIGHT_DECAY)
        use_sf = bool(getattr(config, "USE_SCHEDULE_FREE", True))
        if use_sf and _SCHEDULE_FREE_TYPES:
            variant = str(getattr(config, "SCHEDULE_FREE_VARIANT", "radam")).lower().strip()
            if variant not in ("radam", "adamw"):
                variant = "radam"
            betas = tuple(getattr(config, "SCHEDULE_FREE_BETAS", (0.9, 0.999)))
            r = float(getattr(config, "SCHEDULE_FREE_R", 0.0))
            wlp = float(getattr(config, "SCHEDULE_FREE_WEIGHT_LR_POWER", 2.0))
            if variant == "adamw" and AdamWScheduleFree is not None:
                warmup = int(
                    getattr(
                        config,
                        "SCHEDULE_FREE_WARMUP_STEPS",
                        getattr(config, "LR_WARMUP_STEPS", 2500),
                    )
                )
                return {
                    "optimizer": AdamWScheduleFree(
                        param_groups,
                        lr=self._optimizer_lr(),
                        betas=betas,
                        weight_decay=0.0,
                        warmup_steps=max(0, warmup),
                        r=r,
                        weight_lr_power=wlp,
                    )
                }
            if variant == "radam" and RAdamScheduleFree is not None:
                return {
                    "optimizer": RAdamScheduleFree(
                        param_groups,
                        lr=self._optimizer_lr(),
                        betas=betas,
                        weight_decay=0.0,
                        r=r,
                        weight_lr_power=wlp,
                        silent_sgd_phase=bool(getattr(config, "SCHEDULE_FREE_RADAM_SILENT_SGD", True)),
                    )
                }
            if variant == "radam" and RAdamScheduleFree is None:
                print("[engine] RAdamScheduleFree missing, AdamWScheduleFree")
            if AdamWScheduleFree is not None:
                warmup = int(
                    getattr(
                        config,
                        "SCHEDULE_FREE_WARMUP_STEPS",
                        getattr(config, "LR_WARMUP_STEPS", 2500),
                    )
                )
                return {
                    "optimizer": AdamWScheduleFree(
                        param_groups,
                        lr=self._optimizer_lr(),
                        betas=betas,
                        weight_decay=0.0,
                        warmup_steps=max(0, warmup),
                        r=r,
                        weight_lr_power=wlp,
                    )
                }

        if use_sf and not _SCHEDULE_FREE_TYPES:
            print("[engine] schedulefree missing → AdamW + LambdaLR")

        opt = torch.optim.AdamW(param_groups, lr=self._optimizer_lr())
        total = None
        if self.trainer is not None:
            total = getattr(self.trainer, "estimated_stepping_batches", None)
        if total is None or total < 1:
            total = config.EPOCHS * config.STEPS_PER_EPOCH
        total = int(total)
        warmup = min(int(getattr(config, "LR_WARMUP_STEPS", 2500)), max(1, total - 1))
        min_ratio = float(getattr(config, "LR_COSINE_MIN_RATIO", 0.01))

        def lr_lambda(last_epoch: int) -> float:
            if last_epoch < warmup:
                return float(last_epoch + 1) / float(warmup)
            denom = max(1, total - warmup - 1)
            p = float(last_epoch - warmup) / float(denom)
            c = 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))
            return min_ratio + (1.0 - min_ratio) * c

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}
