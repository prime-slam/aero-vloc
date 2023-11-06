#  Copyright (c) 2023, Stephen Hausler, Sourav Garg, Ming Xu, Michael Milford, Tobias Fischer,
#  Ivan Moskalenko, Anastasiia Kornilova
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
#
#  Significant part of our code is based on Patch-NetVLAD repository
#  (https://github.com/QVPR/Patch-NetVLAD)
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from aero_vloc.vpr_systems.netvlad.model.layers import NetVLADModule


class Flatten(nn.Module):
    def forward(self, input_data):
        return input_data.view(input_data.size(0), -1)


class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, input_data):
        return F.normalize(input_data, p=2, dim=self.dim)


def get_backend():
    enc_dim = 512
    enc = models.vgg16(weights="IMAGENET1K_V1")
    layers = list(enc.features.children())[:-2]
    # only train conv5_1, conv5_2, and conv5_3 (leave rest same as Imagenet trained weights)
    for layer in layers[:-5]:
        for p in layer.parameters():
            p.requires_grad = False
    enc = nn.Sequential(*layers)
    return enc_dim, enc


def get_pca_encoding(model, vlad_encoding):
    pca_encoding = model.WPCA(vlad_encoding.unsqueeze(-1).unsqueeze(-1))
    return pca_encoding


def get_model(
    encoder,
    encoder_dim,
    num_clusters,
    use_vladv2=False,
    append_pca_layer=False,
    num_pcs=8192,
):
    nn_model = nn.Module()
    nn_model.add_module("encoder", encoder)

    net_vlad = NetVLADModule(
        num_clusters=num_clusters, dim=encoder_dim, vladv2=use_vladv2
    )
    nn_model.add_module("pool", net_vlad)
    if append_pca_layer:
        netvlad_output_dim = encoder_dim
        netvlad_output_dim *= num_clusters
        pca_conv = nn.Conv2d(
            netvlad_output_dim, num_pcs, kernel_size=(1, 1), stride=1, padding=0
        )
        nn_model.add_module(
            "WPCA", nn.Sequential(*[pca_conv, Flatten(), L2Norm(dim=-1)])
        )
    return nn_model
