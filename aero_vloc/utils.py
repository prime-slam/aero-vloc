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
import cv2
import torch
import torchvision

from pathlib import Path
from PIL import Image
from torchvision.transforms import InterpolationMode
from typing import Tuple


def get_new_size(height: int, width: int, resize: int):
    scale = resize / max(height, width)
    if scale >= 1:
        return height, width
    else:
        height_new, width_new = int(round(height * scale)), int(round(width * scale))
        return height_new, width_new


def transform_image(
    image: Image,
    resize: int | Tuple[int, int],
    interpolation: InterpolationMode = InterpolationMode.BILINEAR,
):
    if isinstance(resize, int):
        h_new, w_new = get_new_size(image.height, image.width, resize)
    else:
        h_new, w_new = resize
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((h_new, w_new), interpolation=interpolation),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    transformed_image = transform(image)
    return transformed_image


def load_image_for_sp(image_path: Path, resize: int):
    grayim = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    h, w = grayim.shape[:2]
    h_new, w_new = get_new_size(h, w, resize)
    grayim = cv2.resize(grayim, (w_new, h_new), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(grayim / 255.0).float()[None, None]
