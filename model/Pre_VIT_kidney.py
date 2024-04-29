import torch
from torchvision.models import resnet50
from torch import nn


class Pre_VIT(nn.Module):
    def __init__(self, num_classes=4):
        super(Pre_VIT, self).__init__()

        # 使用 torchvision 中的预训练模型
        self.vit = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)

        # 替换原来的分类层，适应我们的四分类问题
        self.vit.head = nn.Linear(self.vit.head.in_features, num_classes)
        self.softmax=nn.Softmax(dim=1)

    def forward(self, x):
        x = self.vit(x)
        x=self.softmax(x)
        return x