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
import torchvision

from aero_vloc.utils import transform_image_for_vpr
from aero_vloc.vpr_systems.vpr_system import VPRSystem
from aero_vloc.vpr_systems.mixvpr.model.mixvpr_model import VPRModel


class MixVPR(VPRSystem):
    """
    Implementation of [MixVPR](https://github.com/amaralibey/MixVPR) global localization method.
    """

    def __init__(self, ckpt_path, gpu_index: int = 0):
        """
        :param ckpt_path: Path to the checkpoint file
        :param gpu_index: The index of the GPU to be used
        """
        super().__init__(gpu_index)
        self.model = VPRModel(
            backbone_arch="resnet50",
            layers_to_crop=[4],
            agg_arch="MixVPR",
            agg_config={
                "in_channels": 1024,
                "in_h": 20,
                "in_w": 20,
                "out_channels": 1024,
                "mix_depth": 4,
                "mlp_ratio": 1,
                "out_rows": 4,
            },
        )

        state_dict = torch.load(ckpt_path)
        self.model.load_state_dict(state_dict)
        self.model.eval().to(self.device)
        print(f"Loaded model from {ckpt_path} successfully!")

    def get_image_descriptor(self, image: np.ndarray):
        # Note that images must be resized to 320x320
        image = transform_image_for_vpr(
            image, (320, 320), torchvision.transforms.InterpolationMode.BICUBIC
        )[None, :].to(self.device)
        with torch.no_grad():
            descriptor = self.model(image)
        descriptor = descriptor.cpu().numpy()[0]
        return descriptor
