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


from abc import ABC, abstractmethod


class IndexSearcher(ABC):
    def __init__(self):
        self.faiss_index = None
        self.computed_query_predictions_indices = []

    @abstractmethod
    def create(self, descriptors: np.ndarray):
        """
        Creates index for storing vectors
        :param descriptors: Descriptors for storage and retrieval
        """
        pass

    @abstractmethod
    def search(self, descriptor: np.ndarray, k_closest: int) -> list[int]:
        """
        Finds the index of the matched database descriptor
        :param descriptor: Query descriptor
        :param k_closest: Specifies how many predictions should be returned
        :return: Indices of the matched descriptors
        """
        pass

    def end_of_query_seq(self):
        """
        Notifies the indexing system that the sequence from the UAV
        is over to prepare it for the following sequence
        """
        self.computed_query_predictions_indices = []
