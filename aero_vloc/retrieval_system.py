#  Copyright (c) 2023, Ivan Moskalenko, Anastasiia Kornilova, Mikhail Kiselyov
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

from pathlib import Path
from timeit import default_timer as timer
from typing import List, Optional, Tuple, Dict
from tqdm import tqdm

from aero_vloc.feature_matchers import FeatureMatcher
from aero_vloc.index_searchers import IndexSearcher
from aero_vloc.primitives import UAVImage
from aero_vloc.vpr_systems import VPRSystem
from aero_vloc.dataset import Data


class RetrievalSystem:
    """
    Allows to create a predictor based on one of the VPR methods
    and feature matcher like LightGlue/SuperGlue.
    """

    def __init__(
        self,
        vpr_system: VPRSystem,
        dataset: Data,
        feature_matcher: FeatureMatcher,
        index_searcher: IndexSearcher,
    ):
        self.vpr_system = vpr_system
        self.feature_matcher = feature_matcher
        self.dataset = dataset
        self.index = index_searcher
        self.time_measurements = {
            "global_descs": [],
            "index_search": [],
            "feature_extraction": [],
            "feature_matching": [],
        }
        self.global_descs = []

        for image in tqdm(
            dataset, desc="Calculating of global descriptors for source DB"
        ):
            self.global_descs.append(self.vpr_system.get_image_descriptor(image))
        self.index.create(np.asarray(self.global_descs))

        local_features = []
        for i, image in enumerate(
            tqdm(dataset, desc="Calculating of local features for source DB")
        ):
            local_features.append(self.feature_matcher.get_feature(image))
        self.source_local_features = np.asarray(local_features)


    def process_batch(
            self,
            images,
            vpr_k_closest: int,
            feature_matcher_k_closest: int | None,
    ) -> List[Tuple[list, Optional[list], Optional[list]]]:
        start = timer()
        query_global_desc = [np.expand_dims(
            self.vpr_system.get_image_descriptor(image), axis=0
        ) for image in images]
        self.time_measurements["global_descs"] = timer() - start
       
        start = timer()
        global_predictions = [self.index.search(query_global_desc, vpr_k_closest) for query_global_desc in query_global_desc]
        self.time_measurements["index_search"] = timer() - start

        if feature_matcher_k_closest is None:
            return global_predictions, None, None
        
        start = timer()
        query_local_features = [self.feature_matcher.get_feature(image) for image in images]
        self.time_measurements["feature_extraction"] = timer() - start

        start = timer()
        filtered_db_features = [self.source_local_features[global_prediction] for global_prediction in global_predictions]
        local_predictions = [self.feature_matcher.match_feature(
            query_local_feature, filtered_db_feature, feature_matcher_k_closest
        ) for query_local_feature, filtered_db_feature in zip(query_local_features, filtered_db_features)]
        self.time_measurements["feature_matching"] = timer() - start

        res_predictions = [((global_prediction[local_prediction], matched_kpts_query, matched_kpts_reference)) for global_prediction, (local_prediction, matched_kpts_query, matched_kpts_reference) in zip(global_predictions, local_predictions)]
        return res_predictions


    def __call__(
        self,
        image,
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
        start = timer()
        query_global_desc = np.expand_dims(
            self.vpr_system.get_image_descriptor(image), axis=0
        )
        self.time_measurements["global_descs"].append(timer() - start)

        start = timer()
        global_predictions = self.index.search(query_global_desc, vpr_k_closest)
        self.time_measurements["index_search"].append(timer() - start)

        if feature_matcher_k_closest is None:
            return global_predictions, None, None

        start = timer()
        query_local_features = self.feature_matcher.get_feature(image)
        self.time_measurements["feature_extraction"].append(timer() - start)

        start = timer()
        filtered_db_features = self.source_local_features[global_predictions]
        (
            local_predictions,
            matched_kpts_query,
            matched_kpts_reference,
        ) = self.feature_matcher.match_feature(
            query_local_features, filtered_db_features, feature_matcher_k_closest
        )
        res_predictions = global_predictions[local_predictions]
        self.time_measurements["feature_matching"].append(timer() - start)
        return res_predictions, matched_kpts_query, matched_kpts_reference

    def end_of_query_seq(self):
        """
        Notifies the retrieval system that the sequence from the UAV
        is over to prepare it for the following sequence
        """
        self.index.end_of_query_seq()

    def get_time_measurements(self) -> Dict[str, float]:
        return self.time_measurements

