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
import faiss
import itertools
import numpy as np

from aero_vloc.index_searchers.index_searcher import IndexSearcher
from aero_vloc.primitives import Map


class SequentialSearcher(IndexSearcher):
    def __init__(self, last_n: int, sat_map: Map):
        """
        A matcher that uses recent predictions to filter hypotheses for the current frame

        :param last_n: Determines how many previous predictions (including the current one) should be used
        :param sat_map: Satellite map used for localization.
        Necessary to determine which tiles are neighboring to which ones.
        """
        super().__init__()
        self.last_n = last_n
        self.sat_map = sat_map

    def create(self, descriptors: np.ndarray):
        self.faiss_index = faiss.IndexFlatL2(descriptors.shape[1])
        self.faiss_index.add(descriptors)

    def search(self, descriptor: np.ndarray, k_closest: int) -> list[int]:
        _, global_predictions_indices = self.faiss_index.search(descriptor, k_closest)
        global_predictions_indices = global_predictions_indices[0]
        self.computed_query_predictions_indices.append(global_predictions_indices)

        possible_seqs = list(
            itertools.product(*self.computed_query_predictions_indices[-self.last_n :])
        )
        correct_seqs = []
        for possible_seq in possible_seqs:
            is_correct = True
            for prediction_index in possible_seq[1:]:
                if not (
                    self.sat_map.are_neighbors(prediction_index, possible_seq[0])
                    or prediction_index == possible_seq[0]
                ):
                    is_correct = False
                    break
            if is_correct:
                correct_seqs.append(possible_seq)
        predictions_indices = list(
            set([correct_seq[-1] for correct_seq in correct_seqs])
        )
        return predictions_indices
