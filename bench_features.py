#  Copyright (c) 2024, Mikhail Kiselyov
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
    # knn=test_ds.knn,
    limit=LIMIT
)

feature_matchers = {
    'lightglue': [avl.LightGlue, [800]],
    'superglue': [avl.SuperGlue, ['weights/superglue_outdoor.pth']],
    # 'sela': [avl.SelaLocal, ['weights/SelaVPR_msls.pth', 'weights/dinov2_vitl14_pretrain.pth']],
}

feature_matcher_measurements = {}

for feature_matcher_name, (method, args) in feature_matchers.items():
    print('Processing', feature_matcher_name)
    feature_matcher = method(*args)
    file_path = f'local_features_{test_ds_name}_{feature_matcher_name}.pkl'
    if Path(file_path).exists():
        with open(file_path, 'rb') as f:
            local_features = pickle.load(f)
    else:
        local_features = create_local_features(test_ds, feature_matcher)
        local_features = np.asarray(local_features)
        with open(file_path, 'wb') as f:
            pickle.dump(local_features, f)
    feature_matcher_measurements[feature_matcher_name] = benchmark_feature_matcher(queries, feature_matcher, local_features, 10)
    del feature_matcher