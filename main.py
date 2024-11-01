#  Copyright (c) 2023, Mikhail Kiselyov
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

import aero_vloc as avl

from pathlib import Path
from aero_vloc.dataset import Data, Queries
from aero_vloc.retrieval_system import RetrievalSystem


test_ds = Data(Path("../VPR-datasets-downloader/datasets"), "nordland", limit=20)
queries = Queries(
    Path("../VPR-datasets-downloader/datasets"), "nordland", test_ds.knn, limit=10
)

eigen_places = avl.SALAD()
super_glue = avl.SuperGlue("weights/superglue_outdoor.pth", resize=800)
faiss_searcher = avl.FaissSearcher()
retrieval_system = RetrievalSystem(eigen_places, test_ds, super_glue, faiss_searcher)

homography_estimator = avl.HomographyEstimator()
localization_pipeline = avl.LocalizationPipeline(retrieval_system, homography_estimator)

recall_value, localization_time = avl.reference_recall(
    queries, localization_pipeline, k_closest=50, threshold=100
)

all_time_measurements = retrieval_system.get_time_measurements() | localization_time
print()
print(recall_value)
print(all_time_measurements)
