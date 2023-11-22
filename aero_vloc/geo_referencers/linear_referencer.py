import cv2

from typing import Tuple

from aero_vloc.geo_referencers.geo_referencer import GeoReferencer
from aero_vloc.primitives import MapTile
from aero_vloc.utils import get_new_size


class LinearReferencer(GeoReferencer):
    def get_lat_lon(
        self, map_tile: MapTile, pixel: Tuple[int, int], resize: int
    ) -> Tuple[float, float]:
        map_image = cv2.imread(str(map_tile.path))
        h_new, w_new = get_new_size(*map_image.shape[:2], resize)

        lat = map_tile.top_left_lat + (abs(pixel[1]) / h_new) * (
            map_tile.bottom_right_lat - map_tile.top_left_lat
        )
        lon = map_tile.top_left_lon + (abs(pixel[0]) / w_new) * (
            map_tile.bottom_right_lon - map_tile.top_left_lon
        )
        return lat, lon
