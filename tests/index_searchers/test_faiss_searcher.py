import aero_vloc as avl
import numpy as np


def test_faiss_searcher_k_closest():
    """
    Checks that the index searcher returns the required number of descriptors
    """
    descs_shape = 1024
    for number_of_descs in range(50, 1000, 50):
        for k_closest in range(1, number_of_descs + 1):
            faiss_searcher = avl.FaissSearcher()
            descs = np.random.rand(number_of_descs, descs_shape)
            faiss_searcher.create(descs)

            query_desc = np.random.rand(1, descs_shape)
            result = faiss_searcher.search(query_desc, k_closest=k_closest)

            assert len(result) == k_closest
