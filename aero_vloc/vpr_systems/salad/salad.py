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

from torchvision import transforms as tvf

from aero_vloc.utils import transform_image_for_vpr
from aero_vloc.vpr_systems.vpr_system import VPRSystem


class SALAD(VPRSystem):
    """
    Wrapper for [SALAD](https://github.com/serizba/salad) VPR method
    """

    def __init__(
        self,
        resize: int = 800,
        gpu_index: int = 0,
    ):
        """
        :param resize: The size to which the larger side of the image will be reduced while maintaining the aspect ratio
        :param gpu_index: The index of the GPU to be used
        """
        super().__init__(gpu_index)
        self.resize = resize
        self.model = torch.hub.load("serizba/salad", "dinov2_salad")
        self.model.eval().to(self.device)

    def get_image_descriptor(self, image: np.ndarray):
        image = transform_image_for_vpr(image, self.resize).to(self.device)
        _, h, w = image.shape
        h_new, w_new = (h // 14) * 14, (w // 14) * 14
        img_cropped = tvf.CenterCrop((h_new, w_new))(image)[None, ...]
        with torch.no_grad():
            descriptor = self.model(img_cropped)
        descriptor = descriptor.cpu().numpy()[0]
        return descriptor
