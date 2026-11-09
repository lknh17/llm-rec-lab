"""小 RQ-VAE（TIGER 原文的量化器）：MLP 编码 → L 层残差码本（straight-through）→ MLP 解码。
码本随机初始化、不重置死码——故意保留，用来对照 RQ-KMeans 的"空簇用最远点重置"。"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class RQVAE(nn.Module):
    def __init__(self, dim: int, levels: int, k: int, hidden: int = 128, latent: int = 64):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, latent))
        self.dec = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(), nn.Linear(hidden, dim))
        self.books = nn.ParameterList(nn.Parameter(torch.randn(k, latent) * 0.1) for _ in range(levels))

    def quantize(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r, zq, codes, commit = z, torch.zeros_like(z), [], 0.0
        for book in self.books:
            a = torch.cdist(r, book).argmin(1)
            q = book[a]
            commit = commit + F.mse_loss(r.detach(), q) + 0.25 * F.mse_loss(r, q.detach())   # 码本损失 + β·commitment
            zq, r = zq + q, r - q.detach()
            codes.append(a)
        return z + (zq - z).detach(), torch.stack(codes, 1), commit         # straight-through

    def forward(self, x):
        z = self.enc(x)
        zq, codes, commit = self.quantize(z)
        return self.dec(zq), codes, commit


def train_rqvae(ctx, x: torch.Tensor, levels: int, k: int, steps: int, seed: int) -> tuple[torch.Tensor, dict]:
    torch.manual_seed(seed)
    m = RQVAE(x.shape[1], levels, k).to(x.device)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    g = torch.Generator(device="cpu").manual_seed(seed)
    loss = None
    for s in range(steps):
        b = x[torch.randint(len(x), (4096,), generator=g).to(x.device)]
        rec, codes, commit = m(b)
        loss = F.mse_loss(rec, b) + commit
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (s + 1) % 100 == 0 or s + 1 == steps:
            used = torch.bincount(codes[:, 0], minlength=k).double()
            p = used / used.sum()
            ppl = float(torch.exp(-(p[p > 0] * p[p > 0].log()).sum()))
            ctx.metric("rqvae_loss", float(loss), step=s + 1)
            ctx.metric("rqvae_ppl_L1", ppl / k, step=s + 1)
            ctx.check()
    with torch.no_grad():
        codes = torch.cat([m(x[i:i + 262144])[1] for i in range(0, len(x), 262144)])
    dead = float((torch.bincount(codes[:, 0], minlength=k) == 0).double().mean())
    return codes, {"final_loss": round(float(loss), 5), "dead_L1": round(dead, 4)}
