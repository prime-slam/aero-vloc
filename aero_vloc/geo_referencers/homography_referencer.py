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
import numpy as np

from typing import Optional, Tuple

from aero_vloc.geo_referencers.geo_referencer import GeoReferencer
from aero_vloc.primitives import MapTile, UAVImage
from aero_vloc.utils import get_new_size


class HomographyReferencer(GeoReferencer):
    def __call__(
        self,
        matched_kpts_query: list,
        matched_kpts_reference: list,
        query_image: UAVImage,
        sat_image: MapTile,
        resize_param: int,
    ) -> Optional[Tuple[float, float]]:
        if len(matched_kpts_reference) < 4:
            print("Not enough points for homography")
            return None
        h_new, w_new = get_new_size(
            *cv2.imread(str(query_image.path)).shape[:2], resize_param
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

        center = (cX / resize_param, cY / resize_param)

        latitude = sat_image.top_left_lat + abs(center[1]) * (
            sat_image.bottom_right_lat - sat_image.top_left_lat
        )
        longitude = sat_image.top_left_lon + abs(center[0]) * (
            sat_image.bottom_right_lon - sat_image.top_left_lon
        )
        return latitude, longitude
