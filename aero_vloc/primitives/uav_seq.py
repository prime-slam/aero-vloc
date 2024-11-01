#  Copyright (c) 2023, Ivan Moskalenko, Anastasiia Kornilova, Mikhail Kiselyov
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
from typing import Optional

from aero_vloc.primitives.uav_image import UAVImage


class UAVSeq:
    """
    The class represents a sequence of frames taken from a UAV
    """

    def __init__(self, path_to_metadata: Path):
        """
        Reads sequence of images from metadata file.
        File format -- sequence of lines, each line is a single image.

        The format of a line is as follows:
        `filename groundtruth_longitude groundtruth_latitude`

        :param path_to_metadata: Path to the metadata file
        """
        uav_images = []
        queries_folder = path_to_metadata.parents[0]
        with open(path_to_metadata) as file:
            lines = file.readlines()[1:]
        for line in lines:
            filename, longitude, latitude = line.split()
            uav_image = UAVImage(
                queries_folder / filename, float(latitude), float(longitude)
            )
            uav_images.append(uav_image)
        self.uav_images = uav_images

    def __iter__(self):
        for uav_image in self.uav_images:
            yield uav_image


class RegularSeq:
    def __init__(self, path_to_dataset: Path, limit: Optional[int] = None) -> None:
        queries_path = path_to_dataset / "images/test/queries"
        self.queries_paths = sorted(queries_path.glob("*.png"))
        if limit is not None:
            self.queries_paths = self.queries_paths[:limit]

    def __iter__(self):
        for query_path in self.queries_paths:
            easting, northing = query_path.split("@")[0], query_path.split("@")[1]
            yield UAVImage(query_path, easting, northing)
