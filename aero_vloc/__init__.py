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
from aero_vloc.dataset import Data, Queries
from aero_vloc.feature_matchers import LightGlue, SelaLocal, SuperGlue
from aero_vloc.homography_estimator import HomographyEstimator
from aero_vloc.index_searchers import FaissSearcher
from aero_vloc.localization_pipeline import LocalizationPipeline
from aero_vloc.metrics import reference_recall, retrieval_recall
from aero_vloc.primitives import UAVSeq
from aero_vloc.retrieval_system import RetrievalSystem
from aero_vloc.utils import visualize_matches
from aero_vloc.vpr_systems import (
    AnyLoc,
    CosPlace,
    EigenPlaces,
    MixVPR,
    NetVLAD,
    SALAD,
    Sela,
)