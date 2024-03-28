import aero_vloc as avl

from pathlib import Path

salad = avl.SALAD()
light_glue = avl.LightGlue()
homography_estimator = avl.HomographyEstimator()
queries = avl.UAVSeq(Path("tests/test_data/queries/queries.txt"))


def create_localization_pipeline(
    zoom=1, overlap_level=0, geo_referencer=avl.LinearReferencer()
):
    """
    Creates localization pipeline based on SALAD place recognition system,
    LightGlue keypoint matcher and test satellite map
    """
    sat_map = avl.Map(
        Path("tests/test_data/map/map_metadata.txt"),
        zoom=zoom,
        overlap_level=overlap_level,
        geo_referencer=geo_referencer,
    )
    faiss_searcher = avl.FaissSearcher()
    retrieval_system = avl.RetrievalSystem(salad, sat_map, light_glue, faiss_searcher)
    localization_pipeline = avl.LocalizationPipeline(
        retrieval_system, homography_estimator
    )
    return localization_pipeline
