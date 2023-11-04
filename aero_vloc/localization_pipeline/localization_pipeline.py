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
import faiss
import numpy as np

from tqdm import tqdm
from typing import Optional, Tuple

from uav_loc.feature_matchers import FeatureMatcher
from uav_loc.primitives import UAVImage, Map
from uav_loc.utils import get_new_size
from uav_loc.vpr_systems import VPRSystem


class LocalizationPipeline:
    """
    Allows to create a predictor based on one of the VPR methods
    and feature matcher like LightGlue/SuperGlue.
    The final location is computed using homography.
    """

    def __init__(
        self,
        vpr_system: VPRSystem,
        sat_map: Map,
        feature_matcher: FeatureMatcher,
    ):
        self.vpr_system = vpr_system
        self.feature_matcher = feature_matcher
        self.sat_map = sat_map

        global_descs = []
        for image in tqdm(
            sat_map, desc="Calculating of global descriptors for source DB"
        ):
            global_descs.append(self.vpr_system.get_image_descriptor(image.path))
        self.source_global_descs = np.asarray(global_descs)

        self.faiss_index = faiss.IndexFlatL2(self.source_global_descs.shape[1])
        self.faiss_index.add(self.source_global_descs)
        if self.feature_matcher is not None:
            local_features = []
            for image in tqdm(
                sat_map, desc="Calculating of local features for source DB"
            ):
                local_features.append(self.feature_matcher.get_feature(image.path))
            self.source_local_features = np.asarray(local_features)

    def __retrieve_image(self, query_image: UAVImage, k_closest: int):
        if k_closest < 1:
            raise ValueError("K closest value can't be below 1")

        query_global_desc = np.expand_dims(
            self.vpr_system.get_image_descriptor(query_image.path), axis=0
        )
        _, global_predictions = self.faiss_index.search(query_global_desc, k_closest)
        global_predictions = global_predictions[0]

        query_local_features = self.feature_matcher.get_feature(query_image.path)
        filtered_db_features = self.source_local_features[global_predictions]
        (
            local_prediction,
            matched_kpts_query,
            matched_kpts_reference,
        ) = self.feature_matcher.match_feature(
            query_local_features, filtered_db_features
        )
        if local_prediction is None:
            print("Failed to match images using local features")
            return None
        res_prediction = global_predictions[local_prediction]
        return res_prediction, matched_kpts_query, matched_kpts_reference

    def __visualize_localization(
        self,
        matched_kpts_query,
        matched_kpts_reference,
        sat_image,
        drone_img,
        new_shape,
    ):
        drone_img = cv2.resize(cv2.imread(str(drone_img.path)), new_shape)
        sat_img = cv2.resize(
            cv2.imread(str(sat_image.path)),
            (self.feature_matcher.resize, self.feature_matcher.resize),
        )
        matches = [cv2.DMatch(i, i, 1) for i in range(len(matched_kpts_query))]
        matched_kpts_query = [cv2.KeyPoint(x, y, 1) for x, y in matched_kpts_query]
        matched_kpts_reference = [
            cv2.KeyPoint(x, y, 1) for x, y in matched_kpts_reference
        ]
        img = cv2.drawMatches(
            drone_img,
            matched_kpts_query,
            sat_img,
            matched_kpts_reference,
            matches,
            None,
        )
        return img

    def localize(
        self, query_image: UAVImage, k_closest: int, visualize: bool = False
    ) -> Optional[Tuple[float, float, Optional[np.ndarray]]]:
        """
        Calculates UAV location using global localization and homography
        :param query_image: The image for which the location should be calculated
        :param k_closest: Specifies how many predictions for each query the global localization should make.
        If this value is greater than 1, the best match will be chosen with local matcher
        :param visualize: Determines if visualization of keypoint matching
        between the query keypoints and the selected satellite image should be returned
        :return: Latitude, longitude, visualization (optional). None, if localization failed
        """
        retrieve_result = self.__retrieve_image(query_image, k_closest)
        if retrieve_result is None:
            return None
        res_prediction, matched_kpts_query, matched_kpts_reference = retrieve_result
        if len(matched_kpts_reference) < 4:
            print("Not enough points for homography")
            return None
        chosen_sat_image = self.sat_map.tiles[res_prediction]

        h_new, w_new = get_new_size(
            *cv2.imread(str(query_image.path)).shape[:2], self.feature_matcher.resize
        )
        M, mask = cv2.findHomography(
            matched_kpts_query, matched_kpts_reference, cv2.RANSAC, 5.0
        )
        pts = np.float32(
            [[0, 0], [0, h_new - 1], [w_new - 1, h_new - 1], [w_new - 1, 0]]
        ).reshape(-1, 1, 2)
        try:
            dst = cv2.perspectiveTransform(pts, M)
        except cv2.error:
            print("Perspective transform error. Abort matching")
            return None

        moments = cv2.moments(dst)
        cX = int(moments["m10"] / moments["m00"])
        cY = int(moments["m01"] / moments["m00"])

        center = (cX / self.feature_matcher.resize, cY / self.feature_matcher.resize)

        latitude = chosen_sat_image.top_left_lat + abs(center[1]) * (
            chosen_sat_image.bottom_right_lat - chosen_sat_image.top_left_lat
        )
        longitude = chosen_sat_image.top_left_lon + abs(center[0]) * (
            chosen_sat_image.bottom_right_lon - chosen_sat_image.top_left_lon
        )

        if visualize:
            visualization = self.__visualize_localization(
                matched_kpts_query,
                matched_kpts_reference,
                chosen_sat_image,
                query_image,
                (w_new, h_new),
            )
            return latitude, longitude, visualization
        else:
            return latitude, longitude, None
