import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class GeGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class ResMLPBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.geglu = GeGLU(dim, dim)

    def forward(self, x):
        return x + self.geglu(self.norm(x))


class GroupedBilinearFat(nn.Module):
    def __init__(self, dim: int, groups: int = 16):
        super().__init__()
        if dim % groups != 0:
            raise ValueError("dim must be divisible by groups")
        self.groups = groups
        self.dim_g = dim // groups
        self.weight = nn.Parameter(torch.empty(groups, self.dim_g, self.dim_g))
        nn.init.xavier_uniform_(self.weight)
        self.proj_a = nn.Linear(dim, dim)
        self.proj_b = nn.Linear(dim, dim)
        self.proj_out = nn.Linear(dim, dim)

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        ha = self.proj_a(A).view(A.shape[0], self.groups, self.dim_g)
        hb = self.proj_b(B).view(B.shape[0], self.groups, self.dim_g)
        x = torch.einsum("bgi,gij,bgj->bgj", ha, self.weight, hb)
        x = x.reshape(A.shape[0], self.groups * self.dim_g)
        return self.proj_out(x)


class BitEncoder(nn.Module):
    def __init__(self, latent_dim: int, num_layers: int, num_bits: int = 32):
        super().__init__()
        if latent_dim % num_bits != 0:
            raise ValueError("latent_dim must be divisible by num_bits")
        self.num_bits = num_bits
        self.embed_dim = latent_dim // num_bits
        self.bit_embed = nn.Embedding(2, self.embed_dim)
        self.layers = nn.Sequential(*[ResMLPBlock(latent_dim) for _ in range(num_layers)])

    def forward(self, bits: torch.Tensor):
        x = self.bit_embed(bits).reshape(bits.shape[0], -1)
        return self.layers(x)


class LatentSolverExperts(nn.Module):
    def __init__(self, num_ops: int, latent_dim: int, num_refine: int, op_map: dict):
        super().__init__()
        self.latent_dim = latent_dim
        self.op_table = nn.Embedding(num_ops, latent_dim * 3)

        is_unary = torch.zeros(num_ops)
        is_mul_div = torch.zeros(num_ops)
        for name, (op_id, is_un) in op_map.items():
            if is_un:
                is_unary[op_id] = 1.0
            if name in ("mul", "div"):
                is_mul_div[op_id] = 1.0
        self.register_buffer("is_unary_flag", is_unary)
        self.register_buffer("is_mul_div_flag", is_mul_div)

        self.pre_proj = nn.Linear(latent_dim * 3, latent_dim)
        self.expert_add = nn.Sequential(*[ResMLPBlock(latent_dim) for _ in range(num_refine)])
        self.expert_mul = nn.Sequential(*[ResMLPBlock(latent_dim) for _ in range(num_refine)])
        self.expert_unary = nn.Sequential(*[ResMLPBlock(latent_dim) for _ in range(num_refine)])

        nn.init.normal_(self.op_table.weight[:, : latent_dim * 2], mean=1.0, std=0.02)
        nn.init.normal_(self.op_table.weight[:, latent_dim * 2 :], mean=0.0, std=0.02)

    def forward(self, A: torch.Tensor, B: torch.Tensor, op_ids: torch.Tensor):
        combined = self.op_table(op_ids)
        gamma_a, gamma_b, unary_const = combined.split(self.latent_dim, dim=-1)

        m_un = self.is_unary_flag[op_ids].unsqueeze(-1)
        m_md = self.is_mul_div_flag[op_ids].unsqueeze(-1)
        m_add = 1.0 - m_un - m_md

        B_eff = m_un * unary_const + (1.0 - m_un) * B
        ag = A * gamma_a
        bg = B_eff * gamma_b
        x = self.pre_proj(torch.cat([ag, bg, ag * bg], dim=-1))

        z_add = self.expert_add(x)
        z_mul = self.expert_mul(x)
        z_un = self.expert_unary(x)
        return z_add * m_add + z_mul * m_md + z_un * m_un


class LatentSolverDeepALU(nn.Module):
    def __init__(self, num_ops: int, latent_dim: int, num_refine: int, op_map: dict, pipe_mult: int = 4, bilinear_groups: int = 16):
        super().__init__()
        self.latent_dim = latent_dim
        self.op_table = nn.Embedding(num_ops, latent_dim * 4)

        is_unary = torch.zeros(num_ops)
        for _, (op_id, is_un) in op_map.items():
            if is_un:
                is_unary[op_id] = 1.0
        self.register_buffer("is_unary_flag", is_unary)

        self.role_a = nn.Parameter(torch.randn(1, latent_dim))
        self.role_b = nn.Parameter(torch.randn(1, latent_dim))

        self.bilinear = GroupedBilinearFat(latent_dim, groups=bilinear_groups)
        self.fusion = nn.Linear(latent_dim * 2, latent_dim)

        n_pipe = max(1, int(num_refine) * int(pipe_mult))
        self.pipe = nn.Sequential(*[ResMLPBlock(latent_dim) for _ in range(n_pipe)])

        nn.init.normal_(self.op_table.weight[:, :latent_dim], mean=1.0, std=0.02)
        nn.init.normal_(self.op_table.weight[:, latent_dim : 2 * latent_dim], mean=1.0, std=0.02)
        nn.init.normal_(self.op_table.weight[:, 2 * latent_dim : 3 * latent_dim], mean=0.0, std=0.02)
        nn.init.normal_(self.op_table.weight[:, 3 * latent_dim :], mean=0.0, std=0.02)
        nn.init.normal_(self.role_a, mean=0.0, std=0.02)
        nn.init.normal_(self.role_b, mean=0.0, std=0.02)

    def forward(self, A: torch.Tensor, B: torch.Tensor, op_ids: torch.Tensor):
        combined = self.op_table(op_ids)
        ga, gb, ub, ctx = combined.split(self.latent_dim, dim=-1)

        m_un = self.is_unary_flag[op_ids].unsqueeze(-1)
        B_eff = m_un * ub + (1.0 - m_un) * B

        a_in = (A * ga) + self.role_a
        b_in = (B_eff * gb) + self.role_b

        bi_out = self.bilinear(a_in, b_in)
        x_combined = torch.cat([bi_out + ctx, a_in + b_in], dim=-1)
        x = self.fusion(x_combined)
        return self.pipe(x)


def build_latent_solver(config, op_map: dict) -> nn.Module:
    t = str(getattr(config, "SOLVER_TYPE", "experts")).lower().replace("-", "_")
    if t in ("deep_alu", "deepalu", "ultimate"):
        return LatentSolverDeepALU(
            config.NUM_OPS,
            config.LATENT_DIM,
            config.NUM_LAYERS_REFINE,
            op_map,
            pipe_mult=getattr(config, "SOLVER_DEEP_PIPE_MULT", 4),
            bilinear_groups=getattr(config, "SOLVER_GROUPED_BILINEAR_GROUPS", 16),
        )
    return LatentSolverExperts(config.NUM_OPS, config.LATENT_DIM, config.NUM_LAYERS_REFINE, op_map)


class BitDecoder(nn.Module):
    def __init__(self, latent_dim: int, num_blocks: int = 1):
        super().__init__()
        nb = max(0, int(num_blocks))
        self.blocks = nn.Sequential(*[ResMLPBlock(latent_dim) for _ in range(nb)])
        self.norm = RMSNorm(latent_dim)
        self.to_logits = nn.Linear(latent_dim, 32)

    def forward(self, z: torch.Tensor):
        z = self.blocks(z)
        return self.to_logits(self.norm(z))


class NLA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = BitEncoder(config.LATENT_DIM, config.NUM_LAYERS_ENC, getattr(config, "NUM_BITS", 32))
        self.solver = build_latent_solver(config, config.OPERATIONS_MAP)
        self.decoder = BitDecoder(config.LATENT_DIM, getattr(config, "NUM_DECODER_BLOCKS", 1))


LatentSolver = LatentSolverExperts
