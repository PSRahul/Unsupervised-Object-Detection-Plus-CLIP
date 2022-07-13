from logging import root
import os
import torch
import numpy as np
import random
from tqdm import tqdm
import sys

torch.manual_seed(26)
np.random.seed(26)
random.seed(26)


class CreateSampleSet:
    def __init__(self, root_path, num_files):
        self.delete_files(root_path, num_files)

    def delete_files(self, root_path, num_files):
        folder_names = sorted(os.listdir(root_path))

        for folder_name in tqdm(folder_names):
            class_path = os.path.join(root_path, folder_name)
            image_files = sorted(os.listdir(class_path))

            for image in image_files[num_files:]:
                os.remove(os.path.join(class_path, image))


def main():
    CreateSampleSet(
        root_path="/home/psrahul/MasterThesis/datasets/CUB_200_2011/sample/CUB_200_2011/train/images/",
        num_files=1,
    )


if __name__ == "__main__":
    sys.exit(main())
