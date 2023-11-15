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
from typing import Optional, Tuple

from aero_vloc.geo_referencers import GeoReferencer
from aero_vloc.primitives import UAVSeq
from aero_vloc.retrieval_system import RetrievalSystem


class LocalizationPipeline:
    """
    Allows to create a localizator based on the retrieval system
    and one of the georeference method.
    """

    def __init__(
        self,
        retrieval_system: RetrievalSystem,
        geo_referencer: GeoReferencer,
    ):
        self.retrieval_system = retrieval_system
        self.geo_referencer = geo_referencer

    def __call__(
        self,
        query_seq: UAVSeq,
        k_closest: int,
    ) -> list[Optional[Tuple[float, float]]]:
        """
        Calculates UAV locations using the retrieval system and one of the georeference methods.
        :param query_seq: The sequence of images for which locations should be calculated
        :param k_closest: Specifies how many predictions for each query the global localization should make.
        :return: List of geocoordinates. Also, the values can be None if the location could not be determined
        """
        localization_results = []
        for query_image in query_seq:
            (
                res_prediction,
                matched_kpts_query,
                matched_kpts_reference,
            ) = self.retrieval_system(
                query_image, k_closest, feature_matcher_k_closest=1
            )

            res_prediction = res_prediction[0]
            matched_kpts_query = matched_kpts_query[0]
            matched_kpts_reference = matched_kpts_reference[0]

            chosen_sat_image = self.retrieval_system.sat_map[res_prediction]
            referencer_result = self.geo_referencer(
                matched_kpts_query,
                matched_kpts_reference,
                query_image,
                chosen_sat_image,
                self.retrieval_system.feature_matcher.resize,
            )
            if referencer_result is None:
                localization_results.append(None)
                continue
            latitude, longitude = referencer_result
            localization_results.append((latitude, longitude))
        self.retrieval_system.end_of_query_seq()
        return localization_results
