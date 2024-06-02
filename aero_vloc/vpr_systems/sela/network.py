#  Copyright (c) 2024, Feng Lu, Ivan Moskalenko, Anastasiia Kornilova
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import torch
import torch.nn.functional as F

from torch import nn
from torch.nn.parameter import Parameter

from aero_vloc.vpr_systems.sela.backbone.vision_transformer import vit_large


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, work_with_tokens=False):
        super().__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps
        self.work_with_tokens = work_with_tokens

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps, work_with_tokens=self.work_with_tokens)

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )


def gem(x, p=3, eps=1e-6, work_with_tokens=False):
    if work_with_tokens:
        x = x.permute(0, 2, 1)
        # unseqeeze to maintain compatibility with Flatten
        return (
            F.avg_pool1d(x.clamp(min=eps).pow(p), (x.size(-1)))
            .pow(1.0 / p)
            .unsqueeze(3)
        )
    else:
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(
            1.0 / p
        )


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        assert x.shape[2] == x.shape[3] == 1
        return x[:, :, 0, 0]


class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim)


class LocalAdapt(nn.Module):
    def __init__(self):
        super().__init__()
        self.upconv1 = torch.nn.ConvTranspose2d(
            in_channels=1024, out_channels=256, kernel_size=3, stride=2, padding=1
        )
        self.upconv2 = torch.nn.ConvTranspose2d(
            in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.upconv1(x)
        x = self.relu(x)
        x = self.upconv2(x)
        return x


class GeoLocalizationNet(nn.Module):
    """The used networks are composed of a backbone and an aggregation layer."""

    def __init__(self, dinov2_path):
        super().__init__()
        self.backbone = get_backbone(dinov2_path)
        self.aggregation = nn.Sequential(
            L2Norm(), GeM(work_with_tokens=None), Flatten()
        )
        self.LocalAdapt = LocalAdapt()

    def global_feat(self, x):
        x = self.backbone(x)
        patch_feature = x["x_norm_patchtokens"].view(-1, 16, 16, 1024)

        x1 = patch_feature.permute(0, 3, 1, 2)
        x1 = self.aggregation(x1)
        global_feature = torch.nn.functional.normalize(x1, p=2, dim=-1)
        return global_feature

    def local_feat(self, x):
        x = self.backbone(x)
        patch_feature = x["x_norm_patchtokens"].view(-1, 16, 16, 1024)

        x0 = patch_feature.permute(0, 3, 1, 2)
        x0 = self.LocalAdapt(x0)
        x0 = x0.permute(0, 2, 3, 1)
        local_feature = torch.nn.functional.normalize(x0, p=2, dim=-1)
        return local_feature


def get_backbone(dinov2_path):
    backbone = vit_large(patch_size=14, img_size=518, init_values=1, block_chunks=0)
    model_dict = backbone.state_dict()
    state_dict = torch.load(dinov2_path)
    model_dict.update(state_dict.items())
    backbone.load_state_dict(model_dict)
    return backbone
