import torch
import torch.nn as nn
import timm
from typing import Tuple, Optional

class MedicalVisionTransformer(nn.Module):
    def __init__(
        self,
        num_classes: int = 23,  # Number of classes in HyperKvasir
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        
        # Load pretrained ViT model
        self.vit = timm.create_model(
            'vit_base_patch16_224',
            pretrained=True,
            num_classes=0,  # We'll add our own classification head
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
        )
        
        # Custom classification heads for different tasks
        self.lesion_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)  # Binary classification for lesion detection (no sigmoid)
        )
        
        self.polyp_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 5)  # Assuming 5 polyp types
        )
        
        self.fibrosis_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 4)  # Assuming 4 fibrosis grades
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Get features from ViT backbone
        features = self.vit(x)
        
        # Apply task-specific heads
        lesion_out = self.lesion_head(features)
        polyp_out = self.polyp_head(features)
        fibrosis_out = self.fibrosis_head(features)
        
        return lesion_out, polyp_out, fibrosis_out

    def get_attention_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Get attention maps for visualization"""
        return self.vit.get_attention_maps(x) 