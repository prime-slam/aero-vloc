import math

from typing import Tuple

from aero_vloc.geo_referencers.geo_referencer import GeoReferencer
from aero_vloc.primitives import MapTile


class GoogleMapsReferencer(GeoReferencer):
    def __init__(self, zoom):
        self.zoom = zoom

        # Magic constants
        self.map_width = 256
        self.map_height = 256

    def __lat_lon_to_world(self, lat, lon):
        x = (lon + 180) * (self.map_width / 360)
        y = (
            (
                1
                - math.log(
                    math.tan(lat * math.pi / 180) + 1 / math.cos(lat * math.pi / 180)
                )
                / math.pi
            )
            / 2
        ) * self.map_height

        return x, y

    def __world_to_lat_lon(self, x, y):
        lon = x / self.map_width * 360 - 180

        n = math.pi - 2 * math.pi * y / self.map_height
        lat = 180 / math.pi * math.atan(0.5 * (math.exp(n) - math.exp(-n)))

        return lat, lon

    def get_lat_lon(
        self, map_tile: MapTile, pixel: Tuple[int, int], resize: int
    ) -> Tuple[float, float]:
        top_left_x, top_left_y = self.__lat_lon_to_world(
            map_tile.top_left_lat, map_tile.top_left_lon
        )

        x_scale = math.pow(2, self.zoom) / (resize / self.map_width)
        y_scale = math.pow(2, self.zoom) / (resize / self.map_width)

        desired_x = top_left_x + (self.map_width * abs(pixel[0]) / resize) / x_scale
        desired_y = top_left_y + (self.map_height * abs(pixel[1]) / resize) / y_scale

        lat, lon = self.__world_to_lat_lon(desired_x, desired_y)
        return lat, lon
