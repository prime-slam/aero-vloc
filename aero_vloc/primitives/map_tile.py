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
import cv2
import numpy as np

from pathlib import Path


class MapTile:
    """
    The class represents one satellite map tile with specified coordinates.
    Generally consists of several image files.
    """

    def __init__(
        self,
        paths: list[list[Path]],
        top_left_lat: float,
        top_left_lon: float,
        bottom_right_lat: float,
        bottom_right_lon: float,
        region_of_interest: tuple[int, int, int, int] = None,
    ):
        """
        :param paths: 2D list of paths to the image files
                      according to their actual location
        :param top_left_lat: Top left latitude of the tile
        :param top_left_lon: Top left longitude of the tile
        :param bottom_right_lat: Bottom right latitude of the tile
        :param bottom_right_lon: Bottom right longitude of the tile
        :param region_of_interest: Region of the interest of the united image
                                   in the (top left X, top left Y,
                                   bottom right X, bottom right Y) format
                                   If None, no crop is applied
        """
        self.paths = paths
        self.top_left_lat = top_left_lat
        self.top_left_lon = top_left_lon
        self.bottom_right_lat = bottom_right_lat
        self.bottom_right_lon = bottom_right_lon
        self.region_of_interest = region_of_interest

    @property
    def image(self) -> np.ndarray:
        horizontal_lines = []
        for horizontal_line in self.paths:
            images = [cv2.imread(str(img)) for img in horizontal_line]
            horizontal_lines.append(np.hstack(images))
        result = np.vstack(horizontal_lines)
        if self.region_of_interest is not None:
            (
                top_left_x,
                top_left_y,
                bottom_right_x,
                bottom_right_y,
            ) = self.region_of_interest
            result = result[
                top_left_y : bottom_right_y + 1, top_left_x : bottom_right_x + 1
            ]
        return result
