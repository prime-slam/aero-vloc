from tests.utils import create_localization_pipeline, homography_estimator, queries


def test_homography_estimator():
    """
    Tests homography estimator with a good example from the test dataset
    """
    uav_image = queries.uav_images[0]
    retrieval_system = create_localization_pipeline().retrieval_system
    predictions, matched_kpts_query, matched_kpts_reference = retrieval_system(
        uav_image, vpr_k_closest=2, feature_matcher_k_closest=1
    )
    resize = retrieval_system.feature_matcher.resize
    homography_result = homography_estimator(
        matched_kpts_query[0], matched_kpts_reference[0], uav_image, resize
    )

    assert homography_result == (259, 363)
