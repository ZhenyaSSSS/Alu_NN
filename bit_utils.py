import torch
from config import QUIET_NAN


def canonicalize_float(f32_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    f32_tensor = f32_tensor.to(dtype=torch.float32)
    int_repr = f32_tensor.view(torch.int32)
    int_repr = torch.where(int_repr == torch.iinfo(torch.int32).min, torch.zeros_like(int_repr), int_repr)
    f32_tensor = int_repr.view(torch.float32)
    nan_mask = torch.isnan(f32_tensor)
    return f32_tensor, nan_mask


@torch.no_grad()
def float32_to_bits(float_tensor: torch.Tensor) -> torch.Tensor:
    dev = float_tensor.device
    f, nan_mask = canonicalize_float(float_tensor)
    f_clean = f.clone()
    f_clean[nan_mask] = 0.0
    xi = f_clean.view(torch.int32).clone()
    xi[nan_mask] = QUIET_NAN
    shifts = torch.arange(31, -1, -1, device=dev, dtype=torch.int32)
    bits = (xi.unsqueeze(-1) >> shifts) & 1
    return bits.long()


@torch.no_grad()
def bits_to_float(bits_tensor: torch.Tensor) -> torch.Tensor:
    dev = bits_tensor.device
    bits = bits_tensor.to(torch.int32)
    shifts = torch.arange(31, -1, -1, device=dev, dtype=torch.int32)
    shifted = bits << shifts
    xi = shifted.sum(dim=-1, dtype=torch.int32)
    f, _ = canonicalize_float(xi.view(torch.float32))
    return f
