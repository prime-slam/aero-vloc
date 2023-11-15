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
from abc import ABC, abstractmethod
from typing import Optional, Tuple

from aero_vloc.primitives import MapTile, UAVImage


class GeoReferencer(ABC):
    @abstractmethod
    def __call__(
        self,
        matched_kpts_query: list,
        matched_kpts_reference: list,
        query_image: UAVImage,
        sat_image: MapTile,
        resize_param: int,
    ) -> Optional[Tuple[float, float]]:
        """
        Determines UAV geocordinates using key point correspondences
        between an aerial photo and a satellite image.

        Lengths of lists of key points of aerial photo and satellite image should be the same.
        Correspondences of key points are determined by order in the list.

        :param matched_kpts_query: Keypoints of the query image
        :param matched_kpts_reference: Keypoints of the satellite image
        :param query_image: UAV image
        :param sat_image: Satellite map tile
        :param resize_param: The image resize parameter that was used in keypoint matching
        :return: Latitude and longitude of the image. None if the location cannot be determined.
        """
        pass
