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
from pathlib import Path

from aero_vloc.geo_referencers import GeoReferencer
from aero_vloc.primitives.map_tile import MapTile
from aero_vloc.maps.base_map import BaseMap


class Map(BaseMap):
    """
    The class represents the satellite map required for UAV localization.
    Based on the BaseMap class, it allows
    to specify an arbitrary level of overlap and zoom level.
    """

    def __init__(
        self,
        path_to_metadata: Path,
        zoom: float,
        overlap_level: float,
        geo_referencer: GeoReferencer,
    ):
        """
        Reads map from metadata file.
        File format -- sequence of lines, each line is a single tile.

        The format of a line is as follows:
        `filename top_left_lat top_left_lon bottom_right_lat bottom_right_lon`

        :param path_to_metadata: Path to the metadata file
        :param zoom: Zoom Level. For example, a level equal
                     to 0.5 means that the coverage area is doubled.
        :param geo_referencer: Georeference model of the map
        """
        assert zoom > 0
        assert 0 <= overlap_level < 1
        super().__init__(path_to_metadata)
        self.geo_referencer = geo_referencer

        old_tile_h, old_tile_w = self.tile_shape
        map_pixel_height, map_pixel_width = self.pixel_shape
        new_tile_h, new_tile_w = int(old_tile_h // zoom), int(old_tile_w // zoom)

        # Generating of the new tiles
        tiles = []
        for new_top_left_y in range(
            0, map_pixel_height - new_tile_h + 1, int(new_tile_h * (1 - overlap_level))
        ):
            for new_top_left_x in range(
                0,
                map_pixel_width - new_tile_w + 1,
                int(new_tile_w * (1 - overlap_level)),
            ):
                # Finding the tiles that should participate in the creation of new ones
                top_left_index_x, top_left_index_y = (
                    new_top_left_x // old_tile_w,
                    new_top_left_y // old_tile_h,
                )
                new_bottom_right_x, new_bottom_right_y = (
                    new_top_left_x + new_tile_w - 1,
                    new_top_left_y + new_tile_h - 1,
                )
                bottom_right_index_x, bottom_right_index_y = (
                    new_bottom_right_x // old_tile_w,
                    new_bottom_right_y // old_tile_h,
                )
                involved_tiles = self.tiles_2d[
                    top_left_index_y : bottom_right_index_y + 1,
                    top_left_index_x : bottom_right_index_x + 1,
                ]

                # Finding the global pixel coordinates of the top left involved tile
                old_top_left_x, old_top_left_y = (
                    top_left_index_x * old_tile_w,
                    top_left_index_y * old_tile_h,
                )
                # Finding the coordinates of a tile in the involved tiles coordinate system
                top_left_local_x, top_left_local_y = (
                    new_top_left_x - old_top_left_x,
                    new_top_left_y - old_top_left_y,
                )
                bottom_right_local_x, bottom_right_local_y = (
                    new_bottom_right_x - old_top_left_x,
                    new_bottom_right_y - old_top_left_y,
                )

                top_left_lat, top_left_lon = self.geo_referencer.get_lat_lon(
                    involved_tiles[0, 0], (top_left_local_x, top_left_local_y)
                )
                # We also need to find the coordinates of the bottom right corner
                # in the bottom right involved tile coordinate system for georeferencing
                bottom_right_lat, bottom_right_lon = self.geo_referencer.get_lat_lon(
                    involved_tiles[-1, -1],
                    (
                        new_bottom_right_x - bottom_right_index_x * old_tile_w,
                        new_bottom_right_y - bottom_right_index_y * old_tile_h,
                    ),
                )

                paths_to_tiles = [
                    [tile.paths[0][0] for tile in line] for line in involved_tiles
                ]
                tiles.append(
                    MapTile(
                        paths_to_tiles,
                        top_left_lat,
                        top_left_lon,
                        bottom_right_lat,
                        bottom_right_lon,
                        (
                            top_left_local_x,
                            top_left_local_y,
                            bottom_right_local_x,
                            bottom_right_local_y,
                        ),
                    )
                )
        self.tiles = tiles
