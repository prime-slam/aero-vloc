#  Copyright (c) 2023, Mikhail Kiselyov
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
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
import torchvision.transforms as transforms
from PIL import Image


def path_to_pil_img(path):
    return Image.open(path).convert("RGB")


base_transform = transforms.Compose([transforms.ToTensor()])


class Data(torch.utils.data.Dataset):
    def __init__(self, dataset_dir: Path, dataset_name, resize=[224, 224], limit=10, gt=True):
        super().__init__()
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset folder {dataset_dir} not found.")
        self.resize = resize

        database_dir = dataset_dir / dataset_name / "images/test" / "database"
        self.database_paths = sorted(database_dir.glob("*.png")) + sorted(database_dir.glob("*.jpg"))
        if limit is not None:
            self.database_paths = self.database_paths[:limit]

        if gt:
            self.database_utms = np.array(
                [
                    (path.stem.split("@")[1], path.stem.split("@")[2])
                    for path in self.database_paths
                ]
            ).astype(np.float64)

            self.knn = NearestNeighbors(n_jobs=-1)
            self.knn.fit(self.database_utms)

        self.database_num = len(self.database_paths)

    def __getitem__(self, index):
        img = path_to_pil_img(self.database_paths[index])
        # img = base_transform(img)
        # img = transforms.functional.resize(img, self.resize).numpy()
        return np.array(img)

    def __len__(self):
        return len(self.database_paths)

    def __repr__(self):
        return f"< {self.__class__.__name__}, - #database: {self.database_num};>"


class Queries(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dir: Path,
        dataset_name,
        knn: NearestNeighbors | None,
        resize=[224, 224],
        limit=10,
    ):
        super().__init__()
        queries_dir = dataset_dir / dataset_name / "images/test" / "queries"
        if not queries_dir.exists():
            raise FileNotFoundError(f"Queries folder {queries_dir} not found.")
        self.resize = resize

        self.queries_paths = sorted(queries_dir.glob("*.png")) + sorted(queries_dir.glob("*.jpg"))
        if limit is not None:
            self.queries_paths = self.queries_paths[:limit]
        self.queries_num = len(self.queries_paths)
        
        if knn is not None:
            self.queries_utms = np.array(
                [
                    (path.stem.split("@")[1], path.stem.split("@")[2])
                    for path in self.queries_paths
                ]
            ).astype(np.float64)

            self.knn = knn
            self.soft_positives_per_query = knn.radius_neighbors(
                self.queries_utms, 4, return_distance=False
            )

    def __getitem__(self, index):
        img = path_to_pil_img(self.queries_paths[index])
        # img = base_transform(img)
        # img = transforms.functional.resize(img, self.resize).numpy()
        return np.array(img)

    def __len__(self):
        return self.queries_num

    def __repr__(self):
        return f"< {self.__class__.__name__}, - #queries: {self.queries_num};>"

    def get_positives(self):
        return self.soft_positives_per_query
