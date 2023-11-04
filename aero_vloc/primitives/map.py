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

from aero_vloc.primitives.map_tile import MapTile


class Map:
    """
    The class represents the satellite map required for UAV localization.
    It is assumed that the map is divided into tiles.
    """

    def __init__(self, path_to_metadata: Path):
        """
        Reads map from metadata file.
        File format -- sequence of lines, each line is a single tile.

        The format of a line is as follows:
        `filename top_left_lat top_left_lon bottom_right_lat bottom_right_lon`

        :param path_to_metadata: Path to the metadata file
        """
        tiles = []
        map_folder = path_to_metadata.parents[0]
        with open(path_to_metadata) as file:
            lines = file.readlines()[1:]
        for line in lines:
            (
                filename,
                top_left_lat,
                top_left_lon,
                bottom_right_lat,
                bottom_right_lon,
            ) = line.split()
            map_tile = MapTile(
                map_folder / filename,
                float(top_left_lat),
                float(top_left_lon),
                float(bottom_right_lat),
                float(bottom_right_lon),
            )
            tiles.append(map_tile)
        self.tiles = tiles

    def __iter__(self):
        for map_tile in self.tiles:
            yield map_tile
