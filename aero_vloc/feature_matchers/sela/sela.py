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

from pathlib import Path

from aero_vloc.feature_matchers.feature_matcher import FeatureMatcher
from aero_vloc.feature_matchers.sela.local_similarity import local_sim
from aero_vloc.utils import transform_image_for_vpr
from aero_vloc.vpr_systems.sela.network import GeoLocalizationNet


class SelaLocal(FeatureMatcher):
    """
    Implementation of [SelaVPR](https://github.com/Lu-Feng/SelaVPR)
    re-ranking method.
    """

    def __init__(self, path_to_state_dict: Path, dinov2_path: Path, gpu_index: int = 0):
        """
        :param path_to_state_dict: Path to the SelaVPR weights
        :param dinov2_path: Path to the DINOv2 (ViT-L/14) foundation model
        :param gpu_index: The index of the GPU to be used
        """
        super().__init__((61, 61), gpu_index)

        self.model = GeoLocalizationNet(dinov2_path)
        self.model = self.model.eval().to(self.device)

        state_dict = torch.load(path_to_state_dict)["model_state_dict"]
        state_dict = {k[7:]: v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict)

    def get_feature(self, image: np.ndarray):
        image = transform_image_for_vpr(image, resize=(224, 224))[None, :].to(
            self.device
        )
        with torch.no_grad():
            local_feat = self.model.local_feat(image)
        local_feat = local_feat.cpu().numpy()[0]
        return local_feat

    def match_feature(self, query_features, db_features, k_best):
        query_local_features = torch.Tensor(query_features).to(self.device)
        db_features = torch.Tensor(db_features).to(self.device)
        scores, all_kpts_query, all_kpts_reference = local_sim(
            query_local_features, db_features, self.device
        )
        rerank_index = scores.cpu().numpy().argsort()[::-1]
        res_indices = rerank_index[:k_best]

        matched_kpts_query = [all_kpts_query[i] for i in res_indices]
        matched_kpts_reference = [all_kpts_reference[i] for i in res_indices]
        return (
            res_indices,
            matched_kpts_query,
            matched_kpts_reference,
        )
