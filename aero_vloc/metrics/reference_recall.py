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
from aero_vloc.localization_pipeline import LocalizationPipeline
from aero_vloc.metrics.utils import calculate_distance
from aero_vloc.primitives import UAVSeq


def reference_recall(
    uav_seq: UAVSeq,
    localization_pipeline: LocalizationPipeline,
    k_closest: int,
    threshold: int,
) -> float:
    """
    The metric finds the number of correctly matched frames based on georeference error

    :param uav_seq: Sequence of UAV images
    :param localization_pipeline: Instance of LocalizationPipeline class
    :param k_closest: Specifies how many predictions for each query the global localization should make.
    If this value is greater than 1, the best match will be chosen with local matcher
    :param threshold: The distance between query and reference geocoordinates,
    below which the frame will be considered correctly matched

    :return: Recall value
    """
    recall_value = 0
    localization_results = localization_pipeline(uav_seq, k_closest)
    for loc_res, uav_image in zip(localization_results, uav_seq):
        if loc_res is not None:
            lat, lon = loc_res
            error = calculate_distance(
                lat, lon, uav_image.gt_latitude, uav_image.gt_longitude
            )
            if error < threshold:
                recall_value += 1
    return recall_value / len(uav_seq.uav_images)
