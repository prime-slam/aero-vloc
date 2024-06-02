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
import numpy as np
import torch

from aero_vloc.utils import transform_image_for_vpr
from aero_vloc.vpr_systems.sela.network import GeoLocalizationNet
from aero_vloc.vpr_systems.vpr_system import VPRSystem


class Sela(VPRSystem):
    """
    Wrapper for [Sela](https://github.com/Lu-Feng/SelaVPR) VPR method
    """

    def __init__(
        self,
        path_to_state_dict,
        dinov2_path,
        gpu_index: int = 0,
    ):
        """
        :param path_to_state_dict: Path to the SelaVPR weights
        :param dinov2_path: Path to the DINOv2 (ViT-L/14) foundation model
        :param gpu_index: The index of the GPU to be used
        """
        super().__init__(gpu_index)
        self.resize = (224, 224)

        self.model = GeoLocalizationNet(dinov2_path)
        self.model = self.model.eval().to(self.device)

        state_dict = torch.load(path_to_state_dict)["model_state_dict"]
        state_dict = {k[7:]: v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict)

    def get_image_descriptor(self, image: np.ndarray):
        image = transform_image_for_vpr(image, self.resize)[None, :].to(self.device)
        with torch.no_grad():
            descriptor = self.model.global_feat(image)
        descriptor = descriptor.cpu().numpy()[0]
        return descriptor
