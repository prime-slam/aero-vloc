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
import torch

from pathlib import Path
from PIL import Image

from aero_vloc.utils import transform_image
from aero_vloc.vpr_systems.vpr_system import VPRSystem


class EigenPlaces(VPRSystem):
    """
    Implementation of [EigenPlaces](https://github.com/gmberton/EigenPlaces) global localization method.
    """

    def __init__(
        self, backbone: str = "ResNet101", fc_output_dim: int = 2048, resize: int = 800
    ):
        """
        :param backbone: Type of backbone
        :param fc_output_dim: Dimension of descriptors
        :param resize: The size to which the larger side of the image will be reduced while maintaining the aspect ratio
        """
        super().__init__()
        self.resize = resize
        self.backbone = backbone
        self.fc_output_dim = fc_output_dim

        self.model = torch.hub.load(
            "gmberton/eigenplaces",
            "get_trained_model",
            backbone=backbone,
            fc_output_dim=fc_output_dim,
        )
        self.model.eval().to(self.device)

    def get_image_descriptor(self, image_path: Path):
        image = Image.open(image_path).convert("RGB")
        image = transform_image(image, self.resize)[None, :].to(self.device)
        with torch.no_grad():
            descriptor = self.model(image)
        descriptor = descriptor.cpu().numpy()[0]
        return descriptor

    get_image_descriptor.__doc__ = VPRSystem.get_image_descriptor.__doc__
