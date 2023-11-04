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
from tqdm import tqdm

from uav_loc.feature_detectors import SuperPoint
from uav_loc.feature_matchers import FeatureMatcher
from uav_loc.feature_matchers.lightglue.model.lightglue_matcher import LightGlueMatcher
from uav_loc.utils import load_image_for_sp


class LightGlue(FeatureMatcher):
    """
    Implementation of [LightGlue](https://github.com/cvg/LightGlue)
    matcher with SuperPoint extractor.
    """

    def __init__(self, resize: int = 800):
        super().__init__(resize)
        self.super_point = SuperPoint().eval().to(self.device)
        self.light_glue_matcher = (
            LightGlueMatcher(features="superpoint").eval().to(self.device)
        )

    def get_feature(self, image_path: Path):
        img = load_image_for_sp(image_path, self.resize).to(self.device)
        shape = img.shape[-2:][::-1]
        with torch.no_grad():
            feats = self.super_point({"image": img})
        feats["descriptors"] = feats["descriptors"].transpose(-1, -2).contiguous()
        feats["image_size"] = torch.tensor(shape)[None].to(img).float()
        return feats

    def match_feature(self, query_features, db_features):
        matched_index = None
        matched_kpts_query = None
        matched_kpts_reference = None
        max_matches = 0

        for db_index, db_feature in enumerate(
            tqdm(db_features, desc="Matching of LG features")
        ):
            matches = self.light_glue_matcher(
                {"image0": query_features, "image1": db_feature}
            )
            matches = matches["matches"][0]
            points_query = query_features["keypoints"][0][matches[..., 0]].cpu().numpy()
            points_db = db_feature["keypoints"][0][matches[..., 1]].cpu().numpy()
            num_matches = len(points_query)

            if num_matches > max_matches:
                max_matches = num_matches
                matched_index = db_index
                matched_kpts_query = points_query
                matched_kpts_reference = points_db

        return matched_index, matched_kpts_query, matched_kpts_reference

    get_feature.__doc__ = FeatureMatcher.get_feature.__doc__
    match_feature.__doc__ = FeatureMatcher.match_feature.__doc__
