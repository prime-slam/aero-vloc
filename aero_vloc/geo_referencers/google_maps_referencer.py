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
import math

from typing import Tuple

from aero_vloc.geo_referencers.geo_referencer import GeoReferencer
from aero_vloc.primitives import MapTile


class GoogleMapsReferencer(GeoReferencer):
    def __init__(self, zoom):
        self.zoom = zoom

        # Magic constants
        self.map_size = 256
        self.img_size = 640
        self.scale = math.pow(2, self.zoom) / (self.img_size / self.map_size)

    def __lat_lon_to_world(self, lat, lon):
        x = (lon + 180) * (self.map_size / 360)
        y = (
            (
                1
                - math.log(
                    math.tan(lat * math.pi / 180) + 1 / math.cos(lat * math.pi / 180)
                )
                / math.pi
            )
            / 2
        ) * self.map_size

        return x, y

    def __world_to_lat_lon(self, x, y):
        lon = x / self.map_size * 360 - 180

        n = math.pi - 2 * math.pi * y / self.map_size
        lat = 180 / math.pi * math.atan(0.5 * (math.exp(n) - math.exp(-n)))

        return lat, lon

    def get_lat_lon(
        self,
        map_tile: MapTile,
        pixel: Tuple[int, int],
        resize: int | Tuple[int, int] = None,
    ) -> Tuple[float, float]:
        top_left_x, top_left_y = self.__lat_lon_to_world(
            map_tile.top_left_lat, map_tile.top_left_lon
        )
        resize_x, resize_y = self.img_size * 2, self.img_size * 2
        if resize is not None:
            height, width = map_tile.shape
            if type(resize) is tuple:
                new_height, new_width = resize
                if width > new_width:
                    resize_x = resize_x * (new_width / width)
                if height > new_height:
                    resize_y = resize_y * (new_height / height)
            elif type(resize) is int:
                tile_size = max(height, width)
                if tile_size > resize:
                    resize_x = resize_x * (resize / tile_size)
                    resize_y = resize_x
            else:
                raise ValueError("Resize param should be int or Tuple[int, int]")

        desired_x = top_left_x + (self.map_size * abs(pixel[0]) / resize_x) / self.scale
        desired_y = top_left_y + (self.map_size * abs(pixel[1]) / resize_y) / self.scale

        lat, lon = self.__world_to_lat_lon(desired_x, desired_y)
        return lat, lon
