#  Copyright (c) 2023, Ivan Moskalenko, Anastasiia Kornilova
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
import numpy as np
import torch

from aero_vloc.utils import transform_image_for_vpr
from aero_vloc.vpr_systems.netvlad.model.models_generic import (
    get_backend,
    get_model,
    get_pca_encoding,
)
from aero_vloc.vpr_systems.vpr_system import VPRSystem


class NetVLAD(VPRSystem):
    """
    Implementation of [NetVLAD](https://github.com/QVPR/Patch-NetVLAD) global localization method.
    """

    def __init__(self, path_to_weights: str, resize: int = 800, gpu_index: int = 0):
        """
        :param path_to_weights: Path to the weights
        :param resize: The size to which the larger side of the image will be reduced while maintaining the aspect ratio
        :param gpu_index: The index of the GPU to be used
        """
        super().__init__(gpu_index)
        self.resize = resize
        encoder_dim, encoder = get_backend()

        checkpoint = torch.load(
            path_to_weights, map_location=lambda storage, loc: storage
        )
        num_clusters = checkpoint["state_dict"]["pool.centroids"].shape[0]
        num_pcs = checkpoint["state_dict"]["WPCA.0.bias"].shape[0]
        self.model = get_model(
            encoder,
            encoder_dim,
            num_clusters,
            append_pca_layer=True,
            num_pcs=num_pcs,
        )
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()

    def get_image_descriptor(self, image: np.ndarray):
        image = transform_image_for_vpr(image, self.resize)[None, :].to(self.device)

        with torch.no_grad():
            image_encoding = self.model.encoder(image)
            vlad_global = self.model.pool(image_encoding)
            vlad_global_pca = get_pca_encoding(self.model, vlad_global)
            desc = vlad_global_pca.detach().cpu().numpy()[0]
        return desc
