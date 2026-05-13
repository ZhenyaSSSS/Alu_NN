import math
import torch
from config import OPERATIONS_MAP
from bit_utils import float32_to_bits

_ORDERED_OPS = sorted(OPERATIONS_MAP.items(), key=lambda kv: kv[1][0])
_OP_IDS = [oid for _, (oid, _) in _ORDERED_OPS]
assert _OP_IDS == list(range(len(_OP_IDS))), "OPERATIONS_MAP op ids must be 0..N-1"


@torch.no_grad()
def generate_batch(batch_size: int, device: torch.device):
    dt = torch.float32
    num_ops = len(_ORDERED_OPS)

    n_uni = int(batch_size * 0.3)
    n_log = int(batch_size * 0.3)
    n_int = int(batch_size * 0.15)
    n_mic = int(batch_size * 0.05)
    n_edg = int(batch_size * 0.1)
    n_trap = batch_size - (n_uni + n_log + n_int + n_mic + n_edg)

    A_uni = torch.empty(n_uni, device=device, dtype=dt).uniform_(-1e5, 1e5)
    B_uni = torch.empty(n_uni, device=device, dtype=dt).uniform_(-1e5, 1e5)

    s_la = torch.randn(n_log, device=device, dtype=dt)
    e_la = torch.empty(n_log, device=device, dtype=dt).uniform_(-30, 30)
    A_log = torch.sign(s_la) * (10**e_la)
    s_lb = torch.randn(n_log, device=device, dtype=dt)
    e_lb = torch.empty(n_log, device=device, dtype=dt).uniform_(-30, 30)
    B_log = torch.sign(s_lb) * (10**e_lb)

    A_int = torch.randint(-1000, 1000, (n_int,), device=device).to(dt)
    B_int = torch.randint(-1000, 1000, (n_int,), device=device).to(dt)

    A_mic = torch.randn(n_mic, device=device, dtype=dt)
    B_mic = A_mic + 1e-5 * torch.randn(n_mic, device=device, dtype=dt)

    edges = torch.tensor([0.0, -0.0, 1.0, -1.0, math.pi, math.e], device=device, dtype=dt)
    ne = edges.shape[0]
    A_edg = edges[torch.randint(0, ne, (n_edg,), device=device)]
    B_edg = edges[torch.randint(0, ne, (n_edg,), device=device)]

    A_trap = torch.randn(n_trap, device=device, dtype=dt) * 1e30
    B_trap = torch.zeros(n_trap, device=device, dtype=dt)

    A = torch.cat([A_uni, A_log, A_int, A_mic, A_edg, A_trap], dim=0)
    B = torch.cat([B_uni, B_log, B_int, B_mic, B_edg, B_trap], dim=0)

    perm = torch.randperm(batch_size, device=device)
    A = A[perm]
    B = B[perm]

    op_idx = torch.randint(0, num_ops, (batch_size,), device=device)

    parts = []
    for name, _ in _ORDERED_OPS:
        if name == "add":
            parts.append(A + B)
        elif name == "sub":
            parts.append(A - B)
        elif name == "mul":
            parts.append(A * B)
        elif name == "div":
            parts.append(A / B)
        elif name == "sin":
            parts.append(torch.sin(A))
        elif name == "cos":
            parts.append(torch.cos(A))
        elif name == "exp":
            parts.append(torch.exp(A))
        elif name == "log":
            parts.append(torch.log(A))
        else:
            raise RuntimeError(f"unknown op {name}")
    stacked = torch.stack(parts, dim=1)
    Target = torch.gather(stacked, 1, op_idx.unsqueeze(1)).squeeze(1)

    bits_A = float32_to_bits(A)
    bits_B = float32_to_bits(B)
    bits_Target = float32_to_bits(Target)

    return bits_A, bits_B, op_idx, bits_Target
