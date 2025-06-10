# train.py

import tqdm.auto as tqdm
import torch
import torch.nn.functional as F
import numpy as np
import transformers
from dataclasses import dataclass, field
from typing import Dict, Sequence, Optional

from My_Trainer.trainer import Trainer
from Dataset.multi_dataset_test_for_close import multi_dataset_close  # 如果你仍想用 close-test
from Model.RadFM.multimodality_model import MultiLLaMAForCausalLM

from Dataset.variance_slice_dataset import VarianceSliceVQADataset
from torchvision import transforms

from datasampler import My_DistributedBatchSampler
# from datasets import load_metric

def compute_metrics(eval_preds):
    ACCs = eval_preds.predictions
    return {"accuracy": float(np.mean(ACCs, axis=-1))}

@dataclass
class ModelArguments:
    lang_encoder_path: str = field(
        default="/home/cs/leijiayu/wuchaoyi/book_pretrain/Results/Book_mix_2048_13B_full/checkpoint-45800"
    )
    tokenizer_path: str = field(
        default="/home/cs/leijiayu/wuchaoyi/Finetune_LLAMA/LLAMA_Model/tokenizer"
    )

@dataclass
class DataArguments:
    Mode: str = field(default="Train")

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    remove_unused_columns: bool = field(default=False)
    batch_size_2D: int = field(default=4)
    batch_size_3D: int = field(default=1)
    output_dir: str = field(
        default="/home/cs/leijiayu/wuchaoyi/multi_modal/src/Results/BLIP_overfit/"
    )
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")

class DataCollator:
    """
    把 VarianceSliceVQADataset 输出的样本打包成一个 batch。
    每个 instance 包含：
      vision_x: Tensor[S,3,224,224]
      lang_x: Tensor[L]
      attention_mask: Tensor[L]
      labels: Tensor[L]
      loss_reweight: Tensor[L]
      key_words_query: list[str]
    """
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # 1) 文本部分按 batch 合并
        lang_x       = torch.stack([inst["lang_x"] for inst in instances], dim=0)
        attention_mask = torch.stack([inst["attention_mask"] for inst in instances], dim=0)
        labels       = torch.stack([inst["labels"] for inst in instances], dim=0)
        loss_reweight= torch.stack([inst["loss_reweight"] for inst in instances], dim=0)
        key_words_query = [inst["key_words_query"] for inst in instances]

        # 2) 视觉部分：所有切片都已统一为 [S,3,224,224]
        vision_x = torch.stack([inst["vision_x"] for inst in instances], dim=0)  
        # shape: [B, S, 3, 224, 224]

        return dict(
            lang_x          = lang_x,
            vision_x        = vision_x,
            attention_mask  = attention_mask,
            labels          = labels,
            loss_reweight   = loss_reweight,
            key_words_query = key_words_query,
        )

def main():
    # 解析命令行参数
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 使用自定义的 sampler
    training_args.data_sampler = My_DistributedBatchSampler

    # -------------- 数据集与预处理 -------------- #

    # CLIP 预处理：Resize→CenterCrop→ToTensor→Normalize
    clip_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        ),
    ])

    # 训练 & 验证 dataset，注意路径根据你项目实际位置调整
    Train_dataset = VarianceSliceVQADataset(
        samples_root     = "processed_samples",
        annotations_file = "train.json",
        num_slices       = 10,
        transforms       = clip_transforms
    )
    Eval_dataset = VarianceSliceVQADataset(
        samples_root     = "../../processed_samples",
        annotations_file = "../../train.json",  # 或评测用的 JSON
        num_slices       = 10,
        transforms       = clip_transforms
    )

    # -------------- 模型初始化 -------------- #

    model = MultiLLaMAForCausalLM(
        lang_model_path = model_args.lang_encoder_path
    )

    # -------------- Trainer 设置 -------------- #

    trainer = Trainer(
        model           = model,
        train_dataset   = Train_dataset,
        eval_dataset    = Eval_dataset,
        args            = training_args,
        data_collator   = DataCollator(),
        compute_metrics = compute_metrics
    )

    # -------------- 启动训练 -------------- #
    trainer.train()
    trainer.save_state()


if __name__ == "__main__":
    main()
