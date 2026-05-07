from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionConfig, CLIPVisionModel


CLIP_IMAGE_SIZE = 224
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


class SVDResidualLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, residual_rank: int = 1, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.residual_rank = residual_rank
        self.weight_main = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)
        self.S_residual = nn.Parameter(torch.empty(residual_rank))
        self.U_residual = nn.Parameter(torch.empty(out_features, residual_rank))
        self.V_residual = nn.Parameter(torch.empty(residual_rank, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        residual = self.U_residual @ torch.diag(self.S_residual) @ self.V_residual
        return F.linear(tensor, self.weight_main + residual, self.bias)


def replace_self_attention_linears(module: nn.Module, residual_rank: int = 1) -> nn.Module:
    for child_name, child in module.named_children():
        if child_name == "self_attn":
            for linear_name, linear in list(child.named_children()):
                if isinstance(linear, nn.Linear):
                    setattr(
                        child,
                        linear_name,
                        SVDResidualLinear(
                            linear.in_features,
                            linear.out_features,
                            residual_rank=residual_rank,
                            bias=linear.bias is not None,
                        ),
                    )
        else:
            replace_self_attention_linears(child, residual_rank=residual_rank)
    return module


def load_vision_config(clip_model_dir: Path) -> CLIPVisionConfig:
    config_path = clip_model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing CLIP config: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config_data = json.load(handle)
    vision_config = config_data.get("vision_config", config_data)
    return CLIPVisionConfig(**vision_config)


class EffortDetector(nn.Module):
    def __init__(self, clip_model_dir: Path, residual_rank: int = 1) -> None:
        super().__init__()
        vision_model = CLIPVisionModel(load_vision_config(clip_model_dir))
        vision_model = getattr(vision_model, "vision_model", vision_model)
        self.backbone = replace_self_attention_linears(vision_model, residual_rank=residual_rank)
        self.head = nn.Linear(1024, 2)

    def features(self, image_tensor: torch.Tensor) -> torch.Tensor:
        return self.backbone(image_tensor).pooler_output

    def classifier(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(features)

    def forward(self, data_dict: dict[str, torch.Tensor] | torch.Tensor, inference: bool = False) -> dict[str, torch.Tensor]:
        image_tensor = data_dict["image"] if isinstance(data_dict, dict) else data_dict
        features = self.features(image_tensor)
        logits = self.classifier(features)
        prob = torch.softmax(logits, dim=1)[:, 1]
        return {"cls": logits, "prob": prob, "feat": features}
