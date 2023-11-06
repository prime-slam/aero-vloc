# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#                       Daniel DeTone
#                       Tomasz Malisiewicz
#                       Ivan Moskalenko
#                       Anastasiia Kornilova
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%
import numpy as np
import torch

from pathlib import Path
from tqdm import tqdm

from aero_vloc.feature_detectors import SuperPoint
from aero_vloc.feature_matchers.feature_matcher import FeatureMatcher
from aero_vloc.feature_matchers.superglue.model.superglue_matcher import (
    SuperGlueMatcher,
)
from aero_vloc.utils import load_image_for_sp


class SuperGlue(FeatureMatcher):
    """
    Implementation of [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork)
    matcher with SuperPoint extractor.
    """

    def __init__(self, path_to_sg_weights, resize=800):
        """
        :param path_to_sg_weights: Path to SuperGlue weights
        :param resize: The size to which the larger side of the image will be reduced while maintaining the aspect ratio
        """
        super().__init__(resize)
        self.super_point = SuperPoint().eval().to(self.device)
        self.super_glue_matcher = (
            SuperGlueMatcher(path_to_sg_weights).eval().to(self.device)
        )

    def get_feature(self, image_path: Path):
        inp = load_image_for_sp(image_path, self.resize).to(self.device)
        shape = inp.shape[2:]
        with torch.no_grad():
            features = self.super_point({"image": inp})
        features["shape"] = shape
        return features

    def match_feature(self, query_features, db_features):
        matched_index = None
        matched_kpts_query = None
        matched_kpts_reference = None
        max_matches = 0

        for db_index, db_feature in enumerate(
            tqdm(db_features, desc="Matching of SG features")
        ):
            pred = {k + "0": v for k, v in query_features.items()}
            pred = {**pred, **{k + "1": v for k, v in db_feature.items()}}
            kpts0 = pred["keypoints0"][0].cpu().numpy()
            kpts1 = pred["keypoints1"][0].cpu().numpy()

            with torch.no_grad():
                pred = self.super_glue_matcher(pred)

            matches = pred["matches0"][0].cpu().numpy()
            valid = matches > -1
            matched_kpts0 = kpts0[valid]
            matched_kpts1 = kpts1[matches[valid]]
            num_matches = np.sum(valid)

            if num_matches > max_matches:
                max_matches = num_matches
                matched_index = db_index
                matched_kpts_query = matched_kpts0
                matched_kpts_reference = matched_kpts1

        return matched_index, matched_kpts_query, matched_kpts_reference
