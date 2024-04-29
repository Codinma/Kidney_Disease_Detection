import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

class VIT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=4, dim=768, depth=6, heads=8, mlp_dim=3072, dropout=0.1):
        super(VIT, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.mlp_dim = mlp_dim

        # Patch Embedding
        self.patch_embedding = nn.Conv2d(in_channels=3, out_channels=dim, kernel_size=patch_size, stride=patch_size)

        # Positional Embedding
        self.positional_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Classifier
        self.classifier = nn.Linear(dim, num_classes)

        self.softmax=nn.Softmax(dim=1)

    def forward(self, x):
        # Patch Embedding
        x = self.patch_embedding(x)  # [batch_size, dim, num_patches_h, num_patches_w]
        x = x.flatten(2).transpose(1, 2)  # [batch_size, num_patches, dim]

        # Positional Embedding
        x = torch.cat([x, self.positional_embedding.repeat(x.shape[0], 1, 1)], dim=-2)

        # Transformer Encoder
        x = self.transformer_encoder(x)

        # Classifier
        x = x.mean(dim=1)
        x = self.classifier(x)
        x=self.softmax(x)

        return x