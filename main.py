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

LIMIT = None

test_ds = Data(Path("datasets"), "satellite", limit=LIMIT, gt=False)
queries = Queries(
    Path("datasets"),
    "satellite",
    knn=None,
    # knn=test_ds.knn,
    limit=LIMIT
)

extractors = {
    # 'anyloc': [avl.AnyLoc, ['weights/anyloc_cluster_centers_aerial.pt']],
    'cosplace': [avl.CosPlace, []],
    'eigenplaces': [avl.EigenPlaces, []],
    'mixvpr': [avl.MixVPR, ['weights/resnet50_MixVPR_4096_channels(1024)_rows(4).ckpt']],
    'salad': [avl.SALAD, []],
    'selavpr': [avl.Sela, ['weights/SelaVPR_msls.pth', 'weights/dinov2_vitl14_pretrain.pth']],
    # 'netvlad': [avl.NetVLAD, ['weights/mapillary_WPCA4096.pth.tar']],
}

matcher = avl.LightGlue(resize=800)
index_searcher = avl.FaissSearcher()

measurements = {}

for name, (method, args) in extractors.items():
    extractor = method(*args)
    retrieval_system = RetrievalSystem(extractor, test_ds, matcher, index_searcher)
    homography_estimator = avl.HomographyEstimator()
    localization_pipeline = avl.LocalizationPipeline(retrieval_system, homography_estimator)

    # recall_value = avl.reference_recall(
    #     queries, localization_pipeline, k_closest=10, threshold=100
    # )
    predictions = localization_pipeline.process_all(queries, k_closest=10)


    measurements[name] = retrieval_system.get_time_measurements()

with open('sattelite_k10.pkl', 'wb') as f:
    pickle.dump(measurements, f)
