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

from tqdm import tqdm

from aero_vloc.feature_detectors import SuperPoint
from aero_vloc.feature_matchers import FeatureMatcher
from aero_vloc.feature_matchers.lightglue.model.lightglue_matcher import (
    LightGlueMatcher,
)
from aero_vloc.utils import transform_image_for_sp


class LightGlue(FeatureMatcher):
    """
    Implementation of [LightGlue](https://github.com/cvg/LightGlue)
    matcher with SuperPoint extractor.
    """

    def __init__(self, resize: int = 800, gpu_index: int = 0):
        """
        :param resize: The size to which the larger side of the image will be reduced while maintaining the aspect ratio
        :param gpu_index: The index of the GPU to be used
        """
        super().__init__(resize, gpu_index)
        self.super_point = SuperPoint().eval().to(self.device)
        self.light_glue_matcher = (
            LightGlueMatcher(features="superpoint").eval().to(self.device)
        )

    def get_feature(self, image: np.ndarray):
        img = transform_image_for_sp(image, self.resize).to(self.device)
        shape = img.shape[-2:][::-1]
        with torch.no_grad():
            feats = self.super_point({"image": img})
        feats["descriptors"] = feats["descriptors"].transpose(-1, -2).contiguous()
        feats = {k: v.to("cpu") for k, v in feats.items()}
        feats["image_size"] = torch.tensor(shape)[None].to(img).float()
        return feats

    def match_feature(self, query_features, db_features, k_best):
        num_matches = []
        matched_kpts_query = []
        matched_kpts_reference = []

        for db_index, db_feature in enumerate(db_features):
            keys = ["keypoints", "scores", "descriptors"]
            query_features = {
                k: (v.to(self.device) if k in keys else v)
                for k, v in query_features.items()
            }
            db_feature = {
                k: (v.to(self.device) if k in keys else v)
                for k, v in db_feature.items()
            }
            matches = self.light_glue_matcher(
                {"image0": query_features, "image1": db_feature}
            )
            matches = matches["matches"][0]
            points_query = query_features["keypoints"][0][matches[..., 0]].cpu().numpy()
            points_db = db_feature["keypoints"][0][matches[..., 1]].cpu().numpy()
            num_matches.append(len(points_query))
            matched_kpts_query.append(points_query)
            matched_kpts_reference.append(points_db)

        num_matches = np.array(num_matches)
        res_indices = (-num_matches).argsort()[:k_best]

        matched_kpts_query = [matched_kpts_query[i] for i in res_indices]
        matched_kpts_reference = [matched_kpts_reference[i] for i in res_indices]
        return (
            res_indices,
            matched_kpts_query,
            matched_kpts_reference,
        )
