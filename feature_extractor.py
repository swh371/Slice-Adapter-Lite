# feature_extractor.py

import torch
from torch.utils.data import DataLoader
import clip
from Dataset.variance_slice_dataset import VarianceSliceVQADataset
from torchvision import transforms

def extract_patch_tokens(model, images):
    """
    images: Tensor[B,3,224,224], dtype should match model.visual.conv1
    return: Tensor[B,196,768]
    """
    # 1) to correct dtype
    target_dtype = next(model.visual.conv1.parameters()).dtype
    images = images.to(dtype=target_dtype)

    # 2) conv1 → [B,768,14,14]
    x = model.visual.conv1(images)
    B, C, H, W = x.shape
    x = x.reshape(B, C, H*W).permute(0, 2, 1)  # [B,196,768]

    # 3) prepend cls token
    cls_token = model.visual.class_embedding.to(images.device).to(target_dtype)
    cls = cls_token.unsqueeze(0).unsqueeze(0).expand(B, 1, C)  # [B,1,768]
    x = torch.cat([cls, x], dim=1)                            # [B,197,768]

    # 4) add positional embedding & norm
    pos = model.visual.positional_embedding.to(images.device).to(target_dtype)  # [197,768]
    x = x + pos.unsqueeze(0)  # broadcast to [B,197,768]
    x = model.visual.ln_pre(x)

    # 5) transformer
    x = x.permute(1, 0, 2)       # [197,B,768]
    x = model.visual.transformer(x)
    x = x.permute(1, 0, 2)       # [B,197,768]

    # 6) drop cls, return [B,196,768]
    return x[:, 1:, :]

if __name__ == "__main__":
    # 简单测试一下
    device = "cuda"
    model, _ = clip.load("ViT-B/16", device=device)
    model.requires_grad_(False)
    clip_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466,0.4578275,0.40821073),
                             (0.26862954,0.26130258,0.27577711))
    ])
    from Dataset.variance_slice_dataset import VarianceSliceVQADataset
    ds = VarianceSliceVQADataset("../processed_samples","../train.json",10,clip_transform)
    dl = DataLoader(ds, batch_size=1)
    for imgs,_,_ in dl:
        # imgs: [1, S, 3,224,224] → flatten to [S,3,224,224]
        imgs = imgs.squeeze(0).to(device)  
        with torch.no_grad():
            patches = extract_patch_tokens(model, imgs)
        print("patches shape:",patches.shape)  # 应该是 [S,196,768]
        break


