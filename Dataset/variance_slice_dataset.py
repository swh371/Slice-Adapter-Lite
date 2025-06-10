import os
import json
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class VarianceSliceVQADataset(Dataset):
    def __init__(self,
                 samples_root: str,
                 annotations_file: str,
                 num_slices: int = 10,
                 transforms=None):
        """
        Args:
            samples_root:     你的 processed_samples 根目录
            annotations_file: train.json 文件路径
            num_slices:       要挑选的切片数量（建议 8-12）
            transforms:       PIL→Tensor 及 Normalize 等预处理
        """
        super().__init__()
        assert 1 <= num_slices <= 16, "num_slices 最多不能超过 16"
        self.samples_root = samples_root
        self.num_slices = num_slices
        self.transforms = transforms

        # 1. 读取 JSON 标注文件
        with open(annotations_file, 'r', encoding='utf-8') as f:
            ann = json.load(f)

        # 2. 构建 (sample_dir, question, answer) 列表
        self.entries = []
        for item in ann:
            sid = str(item["sample_id"])
            q = item.get("question_en", "").strip()
            a = item.get("answer_en", "").strip()
            sample_dir = os.path.join(self.samples_root, sid)
            if not os.path.isdir(sample_dir):
                # 没有这个子文件夹就跳过
                continue
            self.entries.append((sample_dir, q, a))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        sample_dir, question, answer = self.entries[idx]

        # 3. 找到所有 jpg 切片，并按文件名排序
        slice_paths = sorted(glob.glob(os.path.join(sample_dir, "*.jpg")))

        # 4. 计算每张切片（灰度）的像素方差
        variances = []
        for p in slice_paths:
            im = Image.open(p).convert("L")          # 灰度
            arr = np.array(im, dtype=np.float32)
            variances.append(arr.var())

        # 5. 选出方差最高的 num_slices 张
        topk = np.argsort(variances)[-self.num_slices:]
        topk = sorted(topk)  # 保证切片顺序
        selected_paths = [slice_paths[i] for i in topk]

        # 6. 读取并预处理选中的切片
        slices = []
        for p in selected_paths:
            im = Image.open(p).convert("RGB")  # CLIP 需要 3 通道
            if self.transforms:
                im = self.transforms(im)
            else:
                # 最简单的 ToTensor
                from torchvision.transforms import ToTensor
                im = ToTensor()(im)
            slices.append(im)

        # (N, C, H, W)
        selected_slices = torch.stack(slices, dim=0)

        return selected_slices, question, answer

# === 使用示例 ===
if __name__ == "__main__":
    from torchvision import transforms

    clip_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        ),
    ])

    ds = VarianceSliceVQADataset(
        samples_root="../../../processed_samples",
        annotations_file="train.json",
        num_slices=10,
        transforms=clip_transforms
    )

    # 测试一个样本
    imgs, q, a = ds[0]
    print("切片张数及维度：", imgs.shape)  # torch.Size([10, 3, 224, 224])
    print("Question:", q)
    print("Answer:  ", a)
