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


class BaseMap:
    """
    The class represents the base satellite map required for UAV localization.
    It is assumed that the map is divided into non-overlapping tiles.
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
                [[map_folder / filename]],
                float(top_left_lat),
                float(top_left_lon),
                float(bottom_right_lat),
                float(bottom_right_lon),
            )
            tiles.append(map_tile)
        self.tiles = tiles
        height, width = self.shape
        tile_height, tile_width = self.tiles[0].shape
        self.pixel_shape = height * tile_height, width * tile_width

    @property
    def shape(self) -> tuple[int, int]:
        """
        :return: Number of tiles by height and by width
        """
        width = None
        for i, tile in enumerate(self.tiles[1:]):
            if tile.top_left_lat != self.tiles[i].top_left_lat:
                width = i + 1
                break
        if width is None:
            width = len(self.tiles)
        height = int(len(self.tiles) / width)
        return height, width

    @property
    def tiles_2d(self) -> np.ndarray:
        """
        :return: Reshaped map based on the number of tiles in height and width
        """
        return np.array(self.tiles).reshape(self.shape)

    def __iter__(self):
        for map_tile in self.tiles:
            yield map_tile

    def __getitem__(self, key):
        return self.tiles[key]

    def __len__(self):
        return len(self.tiles)

    def get_neighboring_tiles(self, query_index: int) -> list[int]:
        """
        Returns the indexes of neighboring tiles

        :param query_index: Index of the tile for which you need to find neighbors
        :return: Neighboring tile indices
        """
        height, width = self.shape
        x, y = query_index % width, query_index // width
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
            if (0 <= x < width) and (0 <= y < height):
                result_neighbors.append(width * y + x)
        return result_neighbors

    def are_neighbors(self, index_1: int, index_2: int) -> bool:
        """Checks if given tiles are adjacent"""
        neighbors = self.get_neighboring_tiles(index_1)
        return index_2 in neighbors
