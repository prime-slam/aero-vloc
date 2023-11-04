#  Copyright (c) 2023, Nikhil Keetha, Avneesh Mishra, Jay Karhade,
#  Krishna Murthy Jatavallabhula, Sebastian Scherer, Madhava Krishna, Sourav Garg,
#  Ivan Moskalenko, Anastasiia Kornilova
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
import einops as ein
import fast_pytorch_kmeans as fpk
import numpy as np
import torch

from pathlib import Path
from torch.nn import functional as F
from typing import Union


class VLAD:
    """
    An implementation of VLAD algorithm given database and query
    descriptors.

    Constructor arguments:
    - num_clusters:     Number of cluster centers for VLAD
    - desc_dim:         Descriptor dimension. If None, then it is
                        inferred when running `fit` method.
    - intra_norm:       If True, intra normalization is applied
                        when constructing VLAD
    - norm_descs:       If True, the given descriptors are
                        normalized before training and predicting
                        VLAD descriptors. Different from the
                        `intra_norm` argument.
    - dist_mode:        Distance mode for KMeans clustering for
                        vocabulary (not residuals). Must be in
                        {'euclidean', 'cosine'}.
    - vlad_mode:        Mode for descriptor assignment (to cluster
                        centers) in VLAD generation. Must be in
                        {'soft', 'hard'}
    - soft_temp:        Temperature for softmax (if 'vald_mode' is
                        'soft') for assignment
    - cache_dir:        Directory to cache the VLAD vectors. If
                        None, then no caching is done. If a str,
                        then it is assumed as the folder path. Use
                        absolute paths.

    Notes:
    - Arandjelovic, Relja, and Andrew Zisserman. "All about VLAD."
        Proceedings of the IEEE conference on Computer Vision and
        Pattern Recognition. 2013.
    """

    def __init__(
        self,
        num_clusters: int,
        c_centers_path: Path,
        desc_dim: Union[int, None] = None,
        intra_norm: bool = True,
        norm_descs: bool = True,
        dist_mode: str = "cosine",
        vlad_mode: str = "hard",
        soft_temp: float = 1.0,
    ) -> None:
        self.num_clusters = num_clusters
        self.desc_dim = desc_dim
        self.intra_norm = intra_norm
        self.norm_descs = norm_descs
        self.mode = dist_mode
        self.vlad_mode = str(vlad_mode).lower()
        assert self.vlad_mode in ["soft", "hard"]
        self.soft_temp = soft_temp
        # Set in the training phase
        self.c_centers = None
        self.kmeans = None
        # Set the caching
        self.c_centers_path = c_centers_path

    # Generate cluster centers
    def fit(self):
        self.kmeans = fpk.KMeans(self.num_clusters, mode=self.mode)
        print("Using cached cluster centers")
        self.c_centers = torch.load(self.c_centers_path)
        self.kmeans.centroids = self.c_centers
        if self.desc_dim is None:
            self.desc_dim = self.c_centers.shape[1]
            print(f"Desc dim set to {self.desc_dim}")

    def generate(self, query_descs: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        residuals = self.generate_res_vec(query_descs)
        # Un-normalized VLAD vector: [c*d,]
        un_vlad = torch.zeros(self.num_clusters * self.desc_dim)
        if self.vlad_mode == "hard":
            # Get labels for assignment of descriptors
            labels = self.kmeans.predict(query_descs)  # [q]
            # Create VLAD from residuals and labels
            used_clusters = set(labels.numpy())
            for k in used_clusters:
                # Sum of residuals for the descriptors in the cluster
                #  Shape:[q, c, d]  ->  [q', d] -> [d]
                cd_sum = residuals[labels == k, k].sum(dim=0)
                if self.intra_norm:
                    cd_sum = F.normalize(cd_sum, dim=0)
                un_vlad[k * self.desc_dim : (k + 1) * self.desc_dim] = cd_sum
        else:  # Soft cluster assignment
            # Cosine similarity: 1 = close, -1 = away
            cos_sims = F.cosine_similarity(  # [q, c]
                ein.rearrange(query_descs, "q d -> q 1 d"),
                ein.rearrange(self.c_centers, "c d -> 1 c d"),
                dim=2,
            )
            soft_assign = F.softmax(self.soft_temp * cos_sims, dim=1)
            # Soft assignment scores (as probabilities): [q, c]
            for k in range(0, self.num_clusters):
                w = ein.rearrange(soft_assign[:, k], "q -> q 1 1")
                # Sum of residuals for all descriptors (for cluster k)
                cd_sum = ein.rearrange(w * residuals, "q c d -> (q c) d").sum(
                    dim=0
                )  # [d]
                if self.intra_norm:
                    cd_sum = F.normalize(cd_sum, dim=0)
                un_vlad[k * self.desc_dim : (k + 1) * self.desc_dim] = cd_sum
        # Normalize the VLAD vector
        n_vlad = F.normalize(un_vlad, dim=0)
        return n_vlad

    def generate_res_vec(
        self, query_descs: Union[np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        assert self.kmeans is not None
        assert self.c_centers is not None
        # Compute residuals (all query to cluster): [q, c, d]
        if type(query_descs) == np.ndarray:
            query_descs = torch.from_numpy(query_descs).to(torch.float32)
        if self.norm_descs:
            query_descs = F.normalize(query_descs)
        residuals = ein.rearrange(query_descs, "q d -> q 1 d") - ein.rearrange(
            self.c_centers, "c d -> 1 c d"
        )
        return residuals
