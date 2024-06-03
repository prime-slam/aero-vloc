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

from aero_vloc.primitives import UAVSeq
from aero_vloc.retrieval_system import RetrievalSystem


def retrieval_recall(
    uav_seq: UAVSeq,
    retrieval_system: RetrievalSystem,
    vpr_k_closest: int,
    feature_matcher_k_closest: int | None,
) -> np.ndarray:
    """
    The metric finds the number of correctly matched frames based on retrieval results

    :param uav_seq: Sequence of UAV images
    :param retrieval_system: Instance of RetrievalSystem class
    :param vpr_k_closest: Determines how many best images are to be obtained with the VPR system
    :param feature_matcher_k_closest: Determines how many best images are to be obtained with the feature matcher
    If it is None, then the feature matcher turns off

    :return: Array of Recall values for all N < vpr_k_closest,
             or for all N < feature_matcher_k_closest if it is not None
    """
    if feature_matcher_k_closest is not None:
        recalls = np.zeros(feature_matcher_k_closest)
    else:
        recalls = np.zeros(vpr_k_closest)
    for uav_image in uav_seq:
        predictions, _, _ = retrieval_system(
            uav_image, vpr_k_closest, feature_matcher_k_closest
        )
        for i, prediction in enumerate(predictions):
            map_tile = retrieval_system.sat_map[prediction]
            if (
                map_tile.top_left_lat
                > uav_image.gt_latitude
                > map_tile.bottom_right_lat
            ) and (
                map_tile.top_left_lon
                < uav_image.gt_longitude
                < map_tile.bottom_right_lon
            ):
                recalls[i:] += 1
                break

    retrieval_system.end_of_query_seq()
    recalls = recalls / len(uav_seq.uav_images)
    return recalls
