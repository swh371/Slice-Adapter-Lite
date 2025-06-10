# src/my_embedding_layer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from adapter import CrossSliceAdapter
from feature_extractor import extract_patch_tokens  # 你之前在 feature_extractor.py 定义的工具
from .helpers import PerceiverResampler    
from .utils import get_visual_encoder
from .vit_3d import ViT
from .transformer_decoder import TransformerDecoder, TransformerDecoderLayer
from transformers import AutoTokenizer, AutoModel
from einops import rearrange

class MyEmbedding(nn.Module):
    """
    Custom embedding layer for multimodal inputs that combines text and vision features.
    Now uses frozen CLIP-ViT + CrossSliceAdapter instead of original 3D ViT + Perceiver.
    """
    def __init__(self,
                 num_embeddings=32000,
                 embedding_dim=5120,
                 perceiver_num=32,
                 vis_dim=768,
                 patch_size=32,
                 frame_patch_size=4,
                 seg_channel=256,
                 num_slices=10,
                 merge_tokens=16,
                 adapter_nhead=8,
                 adapter_layers=2):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Text embedding weights
        self.weight = nn.Parameter(torch.randn((num_embeddings, embedding_dim)))
        self.figure_token_weight = nn.Parameter(torch.randn((2, embedding_dim)))

        # BERT for optional keyword matching (unchanged)
        self.bert_tokenizer = AutoTokenizer.from_pretrained("xmcmic/Med-KEBERT")
        self.bert_model     = AutoModel.from_pretrained("xmcmic/Med-KEBERT")
        self.bert_projection_fc = nn.Linear(768, vis_dim)

        # —— 以下为新加入的 CLIP + Adapter —— #

        # 1. 加载并冻结 CLIP-ViT-B/16
        self.clip_model, _ = clip.load("ViT-B/16", device="cuda")
        self.clip_model.requires_grad_(False)

        # 2. 跨切片 Adapter
        self.adapter = CrossSliceAdapter(
            num_slices      = num_slices,
            num_patches     = 14*14,       # 224/16 == 14
            merge_tokens    = merge_tokens,
            hidden_dim      = vis_dim,
            nhead           = adapter_nhead,
            num_layers      = adapter_layers
        )
        self.fc_proj = nn.Linear(vis_dim, embedding_dim)


        self.vision_encoder = ViT(
            image_size=512,
            frames=512,
            image_patch_size=patch_size,
            frame_patch_size=frame_patch_size,
            dim=vis_dim,
            depth=12,
            heads=8,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1
        )
        self.vision_encoder.requires_grad_(False)

        self.perceiver = PerceiverResampler(
            dim=vis_dim,
            num_latents=perceiver_num
        )
        self.perceiver.requires_grad_(False)

        # optional decoder & mlp heads (unchanged)
        decoder_layer = TransformerDecoderLayer(d_model=vis_dim, nhead=8, normalize_before=True)
        self.transformer_decoder = TransformerDecoder(decoder_layer=decoder_layer, num_layers=4, norm=nn.LayerNorm(vis_dim))
        self.transformer_decoder_mlp = nn.Sequential(
            nn.Linear(vis_dim, vis_dim//4),
            nn.GELU(),
            nn.Linear(vis_dim//4, vis_dim//8),
            nn.GELU(),
        )
        self.cls_head = nn.Linear(vis_dim//8, 1)

        # Now freeze everything except adapter and text heads
        for name, p in self.named_parameters():
            if "adapter" not in name and "fc_proj" not in name and "bert" not in name and "cls_head" not in name:
                p.requires_grad = False


    def forward(self, text_input, vision_x, key_words_query=None):
        B = text_input.size(0)
        if vision_x.dim() == 5:
            # 1) reshape → [B*S,3,224,224]
            B, S, C, H, W = vision_x.shape
            flat = vision_x.view(B*S, C, H, W)

            # 2) 提取所有切片的 patch tokens
            with torch.no_grad():
                patch = extract_patch_tokens(self.clip_model, flat)  # [B*S,196,vis_dim]
            # 3) restore shape → [B, S, 196, vis_dim]
            patch = patch.view(B, S, 14*14, -1)

            # 4) Adapter 融合 → [B, merge_tokens, vis_dim]
            fused = self.adapter(patch)

            # 5) 映射到 embedding_dim
            vis_emb = self.fc_proj(fused)                 # [B, merge_tokens, emb_dim]
            vis_emb = rearrange(vis_emb, "b T d -> (b T) d")  # [B*T, emb_dim]
            vis_emb = rearrange(vis_emb, "(b T) d -> b T d", b=B)

        else:
            raise ValueError("Expected vision_x with 5 dims: [B,S,3,224,224]")

        # Optionally compute keyword-matching loss as before...
        loss_matching = None
        # text_input: [B, L]  → one-hot → matmul
        embedding_weight = torch.cat([self.weight, self.figure_token_weight], dim=0)
        embedding_weight = embedding_weight.unsqueeze(0).repeat(B, 1, 1)
        embedding_weight = torch.cat([embedding_weight, vis_emb], dim=1)

        text_oh = F.one_hot(text_input, embedding_weight.shape[1]).to(vis_emb.dtype).to(vis_emb.device)
        out_put = torch.matmul(text_oh, embedding_weight)

        return out_put, loss_matching
