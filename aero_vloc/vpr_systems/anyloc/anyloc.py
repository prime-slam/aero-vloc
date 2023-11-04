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
import torchvision

from pathlib import Path
from PIL import Image
from torchvision import transforms as tvf

from uav_loc.utils import transform_image
from uav_loc.vpr_systems import VPRSystem
from uav_loc.vpr_systems.anyloc.models import DinoV2ExtractFeatures, VLAD


class AnyLoc(VPRSystem):
    """
    Implementation of [AnyLoc](https://github.com/AnyLoc/AnyLoc) global localization method.
    """

    def __init__(self, c_centers_file: Path, resize: int = 800):
        super().__init__()
        self.resize = resize
        self.extractor = DinoV2ExtractFeatures(
            dino_model="dinov2_vitg14", layer=31, facet="value", device=self.device
        )
        self.c_centers = torch.load(c_centers_file)
        self.vlad = VLAD(num_clusters=32, desc_dim=None, c_centers_path=c_centers_file)
        self.vlad.fit()

    def get_image_descriptor(self, image_path: Path):
        image = Image.open(image_path).convert("RGB")
        image = transform_image(
            image, self.resize, torchvision.transforms.InterpolationMode.BICUBIC
        ).to(self.device)
        _, h, w = image.shape
        h_new, w_new = (h // 14) * 14, (w // 14) * 14
        img_cropped = tvf.CenterCrop((h_new, w_new))(image)[None, ...]
        with torch.no_grad():
            ret = self.extractor(img_cropped)
        gd = self.vlad.generate(ret.cpu().squeeze())
        gd_np = gd.numpy()
        return gd_np

    get_image_descriptor.__doc__ = VPRSystem.get_image_descriptor.__doc__
