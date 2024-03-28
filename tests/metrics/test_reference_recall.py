import aero_vloc as avl
import numpy as np

from tests.utils import create_localization_pipeline, queries


def test_reference_recall():
    """
    Validates the metric using a reasonable level of threshold.
    Since one image was taken outside the test map,
    the recall value should be equal to 0.5
    """
    localization_pipeline = create_localization_pipeline()
    recall, mask = avl.reference_recall(
        queries, localization_pipeline, k_closest=2, threshold=10
    )

    assert np.isclose(recall, 0.5)
    assert mask == [True, False]


def test_reference_recall_low_threshold():
    """
    Validates the metric using a low level of threshold.
    System can't locate queries with such accuracy,
    so the recall value should be equal to 0
    """
    localization_pipeline = create_localization_pipeline()
    recall, mask = avl.reference_recall(
        queries, localization_pipeline, k_closest=2, threshold=1
    )

    assert np.isclose(recall, 0)
    assert mask == [False, False]
