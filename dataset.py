import os
import nibabel
import numpy as np
import pandas as pd
from torchvision.datasets import VisionDataset
import logging

logger = logging.getLogger()


class Lumbar(VisionDataset):
    @staticmethod
    def get_path(root, path):
        if path == "/" or root is None:
            return path
        return os.path.join(root, path)

    def __init__(self, root, metadatafile, transform=None, verify=False):
        super().__init__(root, transform=transform)
        self.df = pd.read_csv(metadatafile)

        if verify:
            # remove all those entries for which we dont have file
            indices = []
            for i, row in self.df.iterrows():
                if not os.path.exists(self.get_path(root, row[FILEPATHKEY])):
                    indices.append(i)
            if indices:
                logger.info(f"Dropping {len(indices)}")
                logger.debug(f"Dropped rows {indices}")
            self.df = self.df.drop(index=indices)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        name = row["name"]

        path = self.get_path(self.root, name)
        img = nibabel.load(path).get_fdata()
        # img = img.transpose(1,0,2)
        img = (img - img.mean()) / img.std()
        scan = img[np.newaxis, :, :, :]

        age = row["age"]

        if self.transform:
            scan = self.transform(scan)

        return name, np.float32(scan), np.float32(age)

    def __len__(self):
        return self.df.shape[0]
