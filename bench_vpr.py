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

import numpy as np
import aero_vloc as avl
import pickle

from pathlib import Path
from aero_vloc.benchmarking import benchmark_feature_matcher, benchmark_vpr_system, create_index, create_local_features
from aero_vloc.dataset import Data, Queries
from aero_vloc.retrieval_system import RetrievalSystem

LIMIT = None

test_ds_name = "satellite"
test_ds = Data(Path("datasets"), test_ds_name, limit=LIMIT, gt=False)
queries = Queries(
    Path("datasets"),
    "satellite",
    knn=None,
    limit=LIMIT
)

vpr_systems = {
    # 'anyloc': [avl.AnyLoc, ['weights/anyloc_cluster_centers_aerial.pt']],
    'cosplace': [avl.CosPlace, []],
    'eigenplaces': [avl.EigenPlaces, []],
    'mixvpr': [avl.MixVPR, ['weights/resnet50_MixVPR_4096_channels(1024)_rows(4).ckpt']],
    'salad': [avl.SALAD, []],
    'selavpr': [avl.Sela, ['weights/SelaVPR_msls.pth', 'weights/dinov2_vitl14_pretrain.pth']],
    # 'netvlad': [avl.NetVLAD, ['weights/mapillary_WPCA4096.pth.tar']],
}


index_searchers = {
    'faiss': [avl.FaissSearcher],
}

matcher = avl.LightGlue(resize=800)
index_searcher = avl.FaissSearcher()

vpr_measurements = {}

for vpr_system_name, (method, args) in vpr_systems.items():
    print('Processing', vpr_system_name)
    index = avl.FaissSearcher()
    vpr_system = method(*args)
    create_index(test_ds, vpr_system, index)
    vpr_measurements[vpr_system_name] = benchmark_vpr_system(queries, vpr_system, index, 10)
    del vpr_system, index


for name, (method, args) in vpr_systems.items():
    extractor = method(*args)
    retrieval_system = RetrievalSystem(extractor, test_ds, matcher, index_searcher)
    homography_estimator = avl.HomographyEstimator()
    localization_pipeline = avl.LocalizationPipeline(retrieval_system, homography_estimator)

    # recall_value = avl.reference_recall(
    #     queries, localization_pipeline, k_closest=10, threshold=100
    # )
    predictions = localization_pipeline.process_all(queries, k_closest=10)


    vpr_measurements[name] = retrieval_system.get_time_measurements()

with open('satellite_vpr.pkl', 'wb') as f:
    pickle.dump(vpr_measurements, f)
