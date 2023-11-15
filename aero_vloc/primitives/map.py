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
import numpy as np

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

        self.width = None
        for i, tile in enumerate(tiles[1:]):
            if tile.top_left_lat != tiles[i].top_left_lat:
                self.width = i + 1
                break
        if self.width is None:
            self.width = len(tiles)
        self.height = int(len(tiles) / self.width)
        self.tiles = tiles

    def __iter__(self):
        for map_tile in self.tiles:
            yield map_tile

    def __getitem__(self, key):
        return self.tiles[key]

    def get_neighboring_tiles(self, query_index: int) -> list[int]:
        """
        Returns the indexes of neighboring tiles

        :param query_index: Index of the tile for which you need to find neighbors
        :return: Neighboring tile indices
        """
        x, y = query_index % self.width, query_index // self.width
        potential_neighbors = [
            (x - 1, y - 1),
            (x, y - 1),
            (x + 1, y - 1),
            (x - 1, y),
            (x + 1, y),
            (x - 1, y + 1),
            (x, y + 1),
            (x + 1, y + 1),
        ]

        result_neighbors = []
        for x, y in potential_neighbors:
            if (0 <= x < self.width) and (0 <= y < self.height):
                result_neighbors.append(self.width * y + x)
        return result_neighbors

    def are_neighbors(self, index_1: int, index_2: int) -> bool:
        """Checks if given tiles are adjacent"""
        neighbors = self.get_neighboring_tiles(index_1)
        return index_2 in neighbors
