import torch.nn as nn
import torchvision


class ViT(nn.Module):
    def __init__(self, name, pretrained=False):
        super(ViT, self).__init__()
        self.model = None
        if name == "vit_b_16":
            self.model = torchvision.models.vit_b_16(pretrained=pretrained)
        elif name == "vit_l_16":
            self.model = torchvision.models.vit_l_16(pretrained=pretrained)
        elif name == "vit_b_32":
            self.model = torchvision.models.vit_b_32(pretrained=pretrained)
        elif name == "vit_l_32":
            self.model = torchvision.models.vit_l_32(pretrained=pretrained)
        else:
            print("Unknown ViT model.")
        self.feat_dim = self.model.heads[0].in_features
        self.model.heads = nn.Identity()

    def forward(self, x):
        x = self.model(x)
        return x


def vit_b_16(pretrained=False):
    model = ViT(name="vit_b_16", pretrained=pretrained)
    return model


def vit_l_16(pretrained=False):
    model = ViT(name="vit_l_16", pretrained=pretrained)
    return model


def vit_b_32(pretrained=False):
    model = ViT(name="vit_b_32", pretrained=pretrained)
    return model


def vit_l_32(pretrained=False):
    model = ViT(name="vit_l_32", pretrained=pretrained)
    return model
