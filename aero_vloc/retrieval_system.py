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

from typing import Optional, Tuple
from tqdm import tqdm

from aero_vloc.feature_matchers import FeatureMatcher
from aero_vloc.index_searchers import IndexSearcher
from aero_vloc.primitives import UAVImage, Map
from aero_vloc.vpr_systems import VPRSystem


class RetrievalSystem:
    """
    Allows to create a predictor based on one of the VPR methods
    and feature matcher like LightGlue/SuperGlue.
    """

    def __init__(
        self,
        vpr_system: VPRSystem,
        sat_map: Map,
        feature_matcher: FeatureMatcher,
        index_searcher: IndexSearcher,
    ):
        self.vpr_system = vpr_system
        self.feature_matcher = feature_matcher
        self.sat_map = sat_map
        self.index = index_searcher

        self.global_descs = []
        for image in tqdm(
            sat_map, desc="Calculating of global descriptors for source DB"
        ):
            image.descriptor = self.vpr_system.get_image_descriptor(image.path)
            self.global_descs.append(self.vpr_system.get_image_descriptor(image.path))
        self.index.create(np.asarray(self.global_descs))

        local_features = []
        for image in tqdm(sat_map, desc="Calculating of local features for source DB"):
            local_features.append(self.feature_matcher.get_feature(image.path))
        self.source_local_features = np.asarray(local_features)

    def __call__(
        self,
        query_image: UAVImage,
        vpr_k_closest: int,
        feature_matcher_k_closest: int | None,
    ) -> Tuple[list, Optional[list], Optional[list]]:
        """
        Retrieves the best matching images using the VPR system and keypoint matcher.

        :param query_image: The image for which you need to find the most relevant images in the database
        :param vpr_k_closest: Determines how many best images are to be obtained with the VPR system
        :param feature_matcher_k_closest: Determines how many best images are to be obtained with the feature matcher
        If it is None, then the feature matcher turns off

        :return: List of predictions,
        list of matched query keypoints for every query -- reference pair (optional),
        list of matched reference keypoints for every query -- reference pair (optional)
        """
        query_global_desc = np.expand_dims(
            self.vpr_system.get_image_descriptor(query_image.path), axis=0
        )
        global_predictions = self.index.search(query_global_desc, vpr_k_closest)

        if feature_matcher_k_closest is None:
            return global_predictions, None, None

        query_local_features = self.feature_matcher.get_feature(query_image.path)
        filtered_db_features = self.source_local_features[global_predictions]
        (
            local_predictions,
            matched_kpts_query,
            matched_kpts_reference,
        ) = self.feature_matcher.match_feature(
            query_local_features, filtered_db_features, feature_matcher_k_closest
        )
        res_predictions = global_predictions[local_predictions]
        return res_predictions, matched_kpts_query, matched_kpts_reference

    def end_of_query_seq(self):
        """
        Notifies the retrieval system that the sequence from the UAV
        is over to prepare it for the following sequence
        """
        self.index.end_of_query_seq()
