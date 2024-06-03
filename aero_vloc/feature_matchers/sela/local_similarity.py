#  Copyright (c) 2024, Feng Lu, Ivan Moskalenko, Anastasiia Kornilova
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
import torch
import numpy as np


def get_keypoints():
    H, W = 61, 61
    patch_size = 1  # 224/16
    N_h = H // patch_size
    N_w = W // patch_size
    keypoints = np.zeros((2, N_h * N_w), dtype=int)  # (x,y)
    keypoints[0] = np.tile(
        np.linspace(patch_size // 2, W - patch_size // 2, N_w, dtype=int), N_h
    )
    keypoints[1] = np.repeat(
        np.linspace(patch_size // 2, H - patch_size // 2, N_h, dtype=int), N_w
    )
    return np.transpose(keypoints)


def match_batch_tensor(fm1, fm2, device):
    """
    fm1: (l,D)
    fm2: (N,l,D)
    mask1: (l)
    mask2: (N,l)
    """
    M = torch.matmul(fm2, fm1.T)  # (N,l,l)

    max1 = torch.argmax(M, dim=1)  # (N,l)
    max2 = torch.argmax(M, dim=2)  # (N,l)
    m = max2[torch.arange(M.shape[0]).reshape((-1, 1)), max1]  # (N, l)
    valid = (
        torch.arange(M.shape[-1]).repeat((M.shape[0], 1)).to(device) == m
    )  # (N, l) bool

    scores = torch.zeros(fm2.shape[0]).to(device)
    all_kpts_query = []
    all_kpts_reference = []

    for i in range(fm2.shape[0]):
        idx1 = torch.nonzero(valid[i, :]).squeeze()
        idx2 = max1[i, :][idx1]
        assert idx1.shape == idx2.shape

        kps = get_keypoints()
        inlier_keypoints_one = kps[idx1.cpu().numpy()]
        inlier_keypoints_two = kps[idx2.cpu().numpy()]

        all_kpts_query.append(inlier_keypoints_one)
        all_kpts_reference.append(inlier_keypoints_two)

        if len(idx1.shape) < 1:
            scores[i] = 0
        else:
            scores[i] = len(idx1)
    return scores, all_kpts_query, all_kpts_reference


def local_sim(features_1, features_2, device):
    B, H, W, C = features_2.shape
    query = features_1
    preds = features_2
    query, preds = query.view(H * W, C), preds.view(B, H * W, C)
    scores, all_kpts_query, all_kpts_reference = match_batch_tensor(
        query, preds, device
    )
    return scores, all_kpts_query, all_kpts_reference
