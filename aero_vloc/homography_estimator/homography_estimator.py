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

from aero_vloc.primitives import UAVImage
from aero_vloc.utils import get_new_size


class HomographyEstimator:
    def __call__(
        self,
        matched_kpts_query: list,
        matched_kpts_reference: list,
        query_image: UAVImage,
        resize_param: int | Tuple[int, int],
    ) -> Optional[Tuple[int, int]]:
        """
        Determines UAV pixel coordinates using key point correspondences
        between an aerial photo and a satellite image.

        Lengths of lists of key points of aerial photo and satellite image should be the same.
        Correspondences of key points are determined by order in the list.

        :param matched_kpts_query: Keypoints of the query image
        :param matched_kpts_reference: Keypoints of the satellite image
        :param query_image: UAV image
        :param resize_param: The image resize parameter that was used in keypoint matching
        :return: Pixel coordinates of the center of query image. None if the location cannot be determined
        """
        if len(matched_kpts_reference) < 4:
            print("Not enough points for homography")
            return None

        if type(resize_param) is tuple:
            h_new, w_new = resize_param
        elif type(resize_param) is int:
            h_new, w_new = get_new_size(*query_image.shape[:2], resize_param)
        else:
            raise ValueError("Resize param should be int or Tuple[int, int]")

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
        if moments["m00"] == 0:
            return None
        cX = int(moments["m10"] / moments["m00"])
        cY = int(moments["m01"] / moments["m00"])
        return cX, cY
