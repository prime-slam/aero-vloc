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
import numpy as np

from aero_vloc.index_searchers.index_searcher import IndexSearcher


class FaissSearcher(IndexSearcher):
    def __init__(self):
        """Simple one-shot bruteforce FAISS matcher"""
        super().__init__()

    def create(self, descriptors: np.ndarray):
        self.faiss_index = faiss.IndexFlatL2(descriptors.shape[1])
        self.faiss_index.add(descriptors)

    def search(self, descriptor: np.ndarray, k_closest: int) -> list[int]:
        _, global_predictions = self.faiss_index.search(descriptor, k_closest)
        global_predictions = global_predictions[0]

        return global_predictions
