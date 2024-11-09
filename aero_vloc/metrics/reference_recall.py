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
from typing import Dict, Tuple
from timeit import default_timer as timer

import numpy as np

from aero_vloc.localization_pipeline import LocalizationPipeline
from aero_vloc.metrics.utils import calculate_distance
from aero_vloc.primitives import UAVSeq
from aero_vloc.dataset import Queries


def reference_recall(
    eval_q: Queries,
    localization_pipeline: LocalizationPipeline,
    k_closest: int,
    threshold: int,
) -> Tuple[float, Dict[str, float]]:
    """
    The metric finds the number of correctly matched frames based on georeference error

    :param eval_q: Sequence of UAV images
    :param localization_pipeline: Instance of LocalizationPipeline class
    :param k_closest: Specifies how many predictions for each query the global localization should make.
    If this value is greater than 1, the best match will be chosen with local matcher
    :param threshold: The distance between query and reference geocoordinates,
    below which the frame will be considered correctly matched

    :return: Recall value
    """
    recall_value = 0

    predictions = localization_pipeline.process_all(eval_q, k_closest)
    # predictions = localization_pipeline(eval_q, k_closest)

    for pred, positives in zip(predictions, eval_q.get_positives()):
        if pred in positives:
            recall_value += 1

    recall = recall_value / eval_q.queries_num
    return recall


    # for loc_res, positives in zip(localization_results, uav_seq.get_positives()):
    #     if loc_res is not None:
    #         lat, lon = loc_res
    #         error

    # for loc_res, uav_image in zip(localization_results, uav_seq):
    #     if loc_res is not None:
    #         lat, lon = loc_res
    #         error = calculate_distance(
    #             lat, lon, uav_image.gt_latitude, uav_image.gt_longitude
    #         )
    #         if error < threshold:
    #             recall_value += 1
    # return recall_value / len(uav_seq)
