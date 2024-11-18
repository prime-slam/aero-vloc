from timeit import default_timer as timer
from typing import Dict, List
from random import sample

import numpy as np
from tqdm import tqdm
from aero_vloc.dataset import Data, Queries
from aero_vloc.feature_matchers.feature_matcher import FeatureMatcher
from aero_vloc.index_searchers.index_searcher import IndexSearcher
from aero_vloc.vpr_systems.vpr_system import VPRSystem

def create_index(
        dataset: Data,
        vpr_system: VPRSystem,
        index_searcher: IndexSearcher,
) -> None:
    """
    Create index for given dataset

    :param dataset: Data object
    :param vpr_system: VPR system
    :param index_searcher: Index searcher
    """
    global_descs = []
    for image in tqdm(
        dataset, desc="DB descriptors"
    ):
        global_descs.append(vpr_system.get_image_descriptor(image))
    index_searcher.create(np.asarray(global_descs))

def benchmark_vpr_system(
        queries: Queries,
        vpr_system: VPRSystem,
        index: IndexSearcher,
        k_closest: int,
) -> Dict[str, Dict[str, List[float]]]:
    """
    Benchmark VPR systems with given dataset and queries

    :param queries: Queries object
    :param vpr_systems: List of VPR systems
    :param index: Index searcher
    :param k_closest: Number of closest images to return

    :return: Dictionary with time measurements
    """
    time_measurements = {}
    start = timer()
    query_global_desc = [np.expand_dims(
        vpr_system.get_image_descriptor(image), axis=0
    ) for image in tqdm(queries, desc=" Q descriptors")]
    time_measurements["global_descs"] = timer() - start

    start = timer()
    for query_global_desc in query_global_desc:
        index.search(query_global_desc, k_closest)
    time_measurements["index_search"] = timer() - start
    
    return time_measurements


def create_local_features(
        dataset: Data,
        feature_matcher,
) -> np.ndarray:
    """
    Create local features for given dataset

    :param dataset: Data object
    :param feature_matcher: Feature matcher

    :return: Local features
    """
    local_features = []
    for i, image in enumerate(
        tqdm(dataset, desc="DB features")
    ):
        local_features.append(feature_matcher.get_feature(image))
    return np.asarray(local_features)


def benchmark_feature_matcher(
        queries: Queries,
        feature_matcher: FeatureMatcher,
        local_features: np.ndarray,
        k_closest: int | None,
) -> Dict[str, Dict[str, List[float]]]:
    """
    Benchmark feature matchers with given dataset and queries

    :param queries: Queries object
    :param feature_matchers: List of feature matchers
    :param k_closest: Number of closest images to return

    :return: Dictionary with time measurements
    """
    time_measurements = {}

    start = timer()
    query_local_features = [feature_matcher.get_feature(queries[i]) for i in tqdm(sample(range(queries.queries_num), k_closest), desc=" Q features")]
    time_measurements["feature_extraction"] = timer() - start

    start = timer()
    for query_local_feature in tqdm(query_local_features, desc="Matching"):
        feature_matcher.match_feature(query_local_feature, local_features, k_closest)
    time_measurements["feature_matching"] = timer() - start

    return time_measurements

