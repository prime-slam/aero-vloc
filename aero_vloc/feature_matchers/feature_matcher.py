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

from abc import ABC, abstractmethod
from typing import Tuple


class FeatureMatcher(ABC):
    def __init__(self, resize: int | Tuple[int, int], gpu_index: int = 0):
        self.resize = resize
        self.device = f"cuda:{gpu_index}" if torch.cuda.is_available() else "cpu"
        print('Running inference on device "{}"'.format(self.device))

    @abstractmethod
    def get_feature(self, image: np.ndarray):
        """
        Gets features of RGB image given
        :param image: The image for which features should be calculated in OpenCV format
        :return: Features for image
        """
        pass

    @abstractmethod
    def match_feature(self, query_features, db_features, k_best):
        """
        Matches query features with database features
        :param query_features: Features for matching
        :param db_features: Database features
        :param k_best: Determines how many top predictions will be returned
        :return: Indices of matched images from database, chosen query features, chosen DB features
        """
        pass
