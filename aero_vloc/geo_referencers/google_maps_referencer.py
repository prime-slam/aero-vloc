import cv2
import math

from typing import Tuple

from aero_vloc.geo_referencers.geo_referencer import GeoReferencer
from aero_vloc.primitives import MapTile
from aero_vloc.utils import get_new_size


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
        self, map_tile: MapTile, pixel: Tuple[int, int], resize: int
    ) -> Tuple[float, float]:
        top_left_x, top_left_y = self.__lat_lon_to_world(
            map_tile.top_left_lat, map_tile.top_left_lon
        )

        desired_x = top_left_x + (self.map_size * abs(pixel[0]) / resize) / self.scale
        desired_y = top_left_y + (self.map_size * abs(pixel[1]) / resize) / self.scale

        lat, lon = self.__world_to_lat_lon(desired_x, desired_y)
        return lat, lon
