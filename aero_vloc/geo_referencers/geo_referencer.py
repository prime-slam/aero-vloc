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
from typing import Tuple

from aero_vloc.primitives import MapTile


class GeoReferencer(ABC):
    @abstractmethod
    def get_lat_lon(
        self,
        map_tile: MapTile,
        pixel: Tuple[int, int],
        resize: int | Tuple[int, int] = None,
    ) -> Tuple[float, float]:
        """
        Finds geographic coordinates of a given pixel on a satellite image

        :param map_tile: Satellite map tile
        :param pixel: Pixel coordinates
        :param resize: The image resize parameter that was used in keypoint matching
        :return: Latitude and longitude of the pixel
        """
        pass
