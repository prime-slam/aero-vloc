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


class UAVImage:
    """
    The class represents one UAV image with specified groundtruth coordinates
    """

    def __init__(self, path: Path, gt_latitude: float, gt_longitude: float):
        self.path = path
        self.gt_latitude = gt_latitude
        self.gt_longitude = gt_longitude
