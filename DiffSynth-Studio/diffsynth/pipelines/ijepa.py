"""I-JEPA ViT-H/14 encoder.

Vendored from facebookresearch/ijepa (src/models/vision_transformer.py) with
the predictor branch, masking, and unused init helpers stripped out. We only
need the context encoder for feature extraction.

Key architectural details vs a standard ViT:
  - No CLS token.
  - Sin-cos 2D positional embedding stored as a non-trainable Parameter.
  - Output is (B, num_patches, embed_dim); no token to strip before returning.
"""

import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn


def _get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    emb = np.concatenate([np.sin(out), np.cos(out)], axis=1)
    return emb


def _get_2d_sincos_pos_embed(embed_dim, grid_size):
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])
    emb_h = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


class _MLP(nn.Module):
    def __init__(self, in_features, hidden_features, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class _Attention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class _Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = _Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.norm2 = norm_layer(dim)
        self.mlp = _MLP(in_features=dim, hidden_features=int(dim * mlp_ratio))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class _PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=14, in_chans=3, embed_dim=1280):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class _VisionTransformer(nn.Module):
    """Subset of facebookresearch/ijepa's VisionTransformer (encoder only)."""

    def __init__(self, img_size=224, patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = _PatchEmbed(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        # Non-trainable sin-cos positional embedding (matches I-JEPA checkpoint exactly).
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=False)
        pos = _get_2d_sincos_pos_embed(embed_dim, int(num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos).float().unsqueeze(0))
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.blocks = nn.ModuleList([
            _Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x) + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


class IJepaEncoder(nn.Module):
    """Wraps an I-JEPA ViT-H/14 encoder loaded from a raw .pth.tar checkpoint."""

    def __init__(self, ijepa_path: str):
        super().__init__()
        self.encoder = _VisionTransformer(
            img_size=224, patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4.0,
        )
        checkpoint = torch.load(ijepa_path, map_location="cpu", weights_only=False)
        # Prefer the EMA target encoder (more stable features); fall back to context encoder.
        state_dict = checkpoint.get("target_encoder", checkpoint.get("encoder", {}))
        # Strip DDP wrapper prefix if present.
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        missing, unexpected = self.encoder.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[IJepaEncoder] missing keys ({len(missing)}): {missing[:3]} ...")
        if unexpected:
            print(f"[IJepaEncoder] unexpected keys ({len(unexpected)}): {unexpected[:3]} ...")
        self.encoder.requires_grad_(False)
        self.hidden_size = self.encoder.embed_dim  # 1280

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, 3, 224, 224) -> (B, 256, 1280); no CLS to strip.
        # Cast input to encoder dtype (preprocessing produces float32; encoder weights are bfloat16).
        x = x.to(self.encoder.patch_embed.proj.weight.dtype)
        return self.encoder(x)


if __name__ == "__main__":
    import sys
    path = sys.argv[1]
    model = IJepaEncoder(path)
    x = torch.randn(1, 3, 224, 224)
    print("Output shape:", model(x).shape)  # expect (1, 256, 1280)
