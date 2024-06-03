import aero_vloc as avl
import numpy as np

from tests.utils import create_localization_pipeline, queries


def test_retrieval_recall():
    """
    Validates the metric using a test dataset.
    Since one image was taken outside the test map,
    the recall value should be equal to 0.5
    """
    localization_pipeline = create_localization_pipeline()
    retrieval_system = localization_pipeline.retrieval_system
    recalls = avl.retrieval_recall(
        queries, retrieval_system, vpr_k_closest=2, feature_matcher_k_closest=1
    )

    assert np.isclose(recalls[0], 0.5)
