# adapter.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchMerger(nn.Module):
    def __init__(self, num_patches: int = 196, num_tokens: int = 16):
        super().__init__()
        self.num_patches = num_patches
        self.num_tokens = num_tokens
        self.weights = nn.Parameter(torch.randn(num_patches, num_tokens))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x must be [B, P, H]
        B, P, H = x.shape
        assert P == self.num_patches, f"Patches mismatch: {P} vs {self.num_patches}"
        # align dtype/device
        w = self.weights.to(dtype=x.dtype, device=x.device)
        w = F.softmax(w, dim=0)             # [P, T]
        out = torch.einsum('bph,pt->bth', x, w)
        return out  # [B, T, H]

class CrossSliceTinyTransformer(nn.Module):
    def __init__(self,
                 seq_len: int,
                 hidden_dim: int = 768,
                 nhead: int = 8,
                 num_layers: int = 2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 2,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, seq_len, hidden]
        return self.transformer(x)

class CrossSliceAdapter(nn.Module):
    def __init__(self,
                 num_slices: int = 10,
                 num_patches: int = 196,
                 merge_tokens: int = 16,
                 hidden_dim: int = 768,
                 nhead: int = 8,
                 num_layers: int = 2):
        super().__init__()
        self.num_slices = num_slices
        self.merge_tokens = merge_tokens

        # 单切片合并
        self.slice_merger = PatchMerger(num_patches=num_patches,
                                        num_tokens=merge_tokens)
        # 跨切片 Transformer
        self.cross_transformer = CrossSliceTinyTransformer(
            seq_len=num_slices * merge_tokens,
            hidden_dim=hidden_dim,
            nhead=nhead,
            num_layers=num_layers
        )
        # 全局再合并
        self.global_merger = PatchMerger(num_patches=num_slices * merge_tokens,
                                         num_tokens=merge_tokens)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        # patch_tokens: [B, S, P, H]
        B, S, P, H = patch_tokens.shape
        assert S == self.num_slices

        # --- 1) 单张切片先合并到 T tokens ---
        # reshape to [B*S, P, H]
        x = patch_tokens.reshape(B * S, P, H)
        x = self.slice_merger(x)                # → [B*S, T, H]
        # 回到 [B, S*T, H]
        x = x.reshape(B, S * self.merge_tokens, H)

        # --- 2) 跨切片 Transformer ---
        x = self.cross_transformer(x)           # [B, S*T, H]

        # --- 3) 全局再合并到 T tokens ---
        x = self.global_merger(x)               # [B, T, H]
        return x

# 单元测试（可删）
if __name__ == "__main__":
    B, S, P, H = 1, 10, 196, 768
    dummy = torch.randn(B, S, P, H)
    adapter = CrossSliceAdapter(num_slices=S)
    out = adapter(dummy)
    print("Adapter output shape:", out.shape)  # [1,16,768]

