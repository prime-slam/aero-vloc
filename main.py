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
import pickle

from pathlib import Path
from aero_vloc.dataset import Data, Queries
from aero_vloc.retrieval_system import RetrievalSystem


test_ds = Data(Path("datasets"), "st_lucia", limit=None)
queries = Queries(
    Path("datasets"), "st_lucia", test_ds.knn, limit=None
)

extractors = {
    'cosplace': avl.CosPlace(),
    'eigenplaces': avl.EigenPlaces(),
    'mixvpr': avl.MixVPR('weights/resnet50_MixVPR_4096_channels(1024)_rows(4).ckpt'),
    'salad': avl.SALAD(),
}

matcher = avl.LightGlue(resize=800)
index_searcher = avl.FaissSearcher()

measurements = {}

for name, extractor in extractors.items():
    retrieval_system = RetrievalSystem(extractor, test_ds, matcher, index_searcher)
    homography_estimator = avl.HomographyEstimator()
    localization_pipeline = avl.LocalizationPipeline(retrieval_system, homography_estimator)

    recall_value = avl.reference_recall(
        queries, localization_pipeline, k_closest=10, threshold=100
    )

    measurements[name] = retrieval_system.get_time_measurements()
print()
print(measurements)

with open('measurements_k10.pkl', 'wb') as f:
    pickle.dump(measurements, f)
