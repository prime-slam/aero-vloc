import aero_vloc as avl

from aero_vloc.metrics.utils import calculate_distance
from tests.utils import create_localization_pipeline, queries


def test_linear_referencer_different_zooms():
    """
    Tests linear referencer with different zoom levels
    """
    for zoom in [1, 1.25, 1.5, 1.75, 2]:
        localization_pipeline = create_localization_pipeline(
            zoom=zoom,
            overlap_level=0.5,
            geo_referencer=avl.LinearReferencer(),
        )
        number_of_tiles = len(localization_pipeline.retrieval_system.sat_map)
        localization_results = localization_pipeline(queries, k_closest=number_of_tiles)

        test_result = localization_results[0]
        lat, lon = test_result
        uav_image = queries.uav_images[0]
        error = calculate_distance(
            lat, lon, uav_image.gt_latitude, uav_image.gt_longitude
        )

        assert error < 5


def test_linear_referencer_different_overlaps():
    """
    Tests linear referencer with different overlap levels
    """
    for overlap in [0, 0.25, 0.5, 0.75]:
        localization_pipeline = create_localization_pipeline(
            zoom=1.5,
            overlap_level=overlap,
            geo_referencer=avl.LinearReferencer(),
        )
        number_of_tiles = len(localization_pipeline.retrieval_system.sat_map)
        localization_results = localization_pipeline(queries, k_closest=number_of_tiles)

        test_result = localization_results[0]
        lat, lon = test_result
        uav_image = queries.uav_images[0]
        error = calculate_distance(
            lat, lon, uav_image.gt_latitude, uav_image.gt_longitude
        )

        assert error < 5
