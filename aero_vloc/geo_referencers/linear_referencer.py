from typing import Tuple

from aero_vloc.geo_referencers.geo_referencer import GeoReferencer
from aero_vloc.primitives import MapTile


class LinearReferencer(GeoReferencer):
    def get_lat_lon(
        self, map_tile: MapTile, pixel: Tuple[int, int], resize: int
    ) -> Tuple[float, float]:
        lat = map_tile.top_left_lat + (abs(pixel[1]) / resize) * (
            map_tile.bottom_right_lat - map_tile.top_left_lat
        )
        lon = map_tile.top_left_lon + (abs(pixel[0]) / resize) * (
            map_tile.bottom_right_lon - map_tile.top_left_lon
        )
        return lat, lon
