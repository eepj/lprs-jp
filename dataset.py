import torch
from torch.utils.data import Dataset
import torchvision
from torchvision.io import read_image
import torchvision.transforms.v2 as transforms

import os
from typing import Sequence


class CharacterDataset(Dataset):

    def __init__(self,
                 image_dir: str,
                 image_paths: Sequence[str],
                 labels: Sequence[str],
                 label_map: dict | None = None):

        self.image_dir = image_dir
        self.image_paths = image_paths
        self.labels = labels
        self.label_map = label_map
        self.preprocessor = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.uint8, scale=True)
        ])

    def __len__(self):

        return len(self.image_paths)

    def __getitem__(self, index: int):

        image_path = os.path.join(self.image_dir, self.image_paths[index])
        image = read_image(image_path)
        label = self.labels[index]

        if self.label_map:
            label = self.label_map[label]

        return image, label
