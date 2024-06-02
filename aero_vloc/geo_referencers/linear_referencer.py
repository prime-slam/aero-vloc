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
from typing import Tuple

from aero_vloc.geo_referencers.geo_referencer import GeoReferencer
from aero_vloc.primitives import MapTile
from aero_vloc.utils import get_new_size


class LinearReferencer(GeoReferencer):
    def get_lat_lon(
        self,
        map_tile: MapTile,
        pixel: Tuple[int, int],
        resize: int | Tuple[int, int] = None,
    ) -> Tuple[float, float]:
        height, width = map_tile.shape
        if resize is not None:
            if type(resize) is tuple:
                height, width = resize
            elif type(resize) is int:
                height, width = get_new_size(height, width, resize)
            else:
                raise ValueError("Resize param should be int or Tuple[int, int]")

        lat = map_tile.top_left_lat + (abs(pixel[1]) / height) * (
            map_tile.bottom_right_lat - map_tile.top_left_lat
        )
        lon = map_tile.top_left_lon + (abs(pixel[0]) / width) * (
            map_tile.bottom_right_lon - map_tile.top_left_lon
        )
        return lat, lon
