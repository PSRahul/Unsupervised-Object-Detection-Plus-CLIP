from cgi import test
import os
import torch
from pathlib import Path
torch.manual_seed(26)

import shutil

import sys
sys.exit(0)


def main():
    image_root_path = "/home/psrahul/MasterThesis/datasets/CUB_200_2011/CUB_200_2011/images/"
    class_names = sorted(os.listdir(image_root_path))

    for i in range(len(class_names)):
        class_name = class_names[i]
        print(class_name)

        train_path = "/home/psrahul/MasterThesis/datasets/CUB_200_2011/CUB_200_2011/train/" + \
            str(class_name)
        Path(train_path).mkdir(parents=True, exist_ok=True)

        test_path = "/home/psrahul/MasterThesis/datasets/CUB_200_2011/CUB_200_2011/test/" + \
            str(class_name)
        Path(test_path).mkdir(
            parents=True, exist_ok=True)

        val_path = "/home/psrahul/MasterThesis/datasets/CUB_200_2011/CUB_200_2011/val/" + \
            str(class_name)
        Path(val_path).mkdir(
            parents=True, exist_ok=True)

        trainval_path = "/home/psrahul/MasterThesis/datasets/CUB_200_2011/CUB_200_2011/trainval/" + \
            str(class_name)
        Path(trainval_path).mkdir(
            parents=True, exist_ok=True)

        image_list = os.listdir(image_root_path + str(class_name))
        class_length = len(image_list)
        random_idx = torch.randperm(class_length)

        test_size = int(0.2 * class_length)
        trainval_size = class_length - test_size
        val_size = int(0.2 * class_length)
        train_size = trainval_size - val_size

        print(class_length, train_size, val_size, test_size)

        train_idx = random_idx[:train_size]
        val_idx = random_idx[train_size:train_size + val_size]
        test_idx = random_idx[train_size +
                              val_size:train_size + val_size + test_size]
        print(train_idx)
        print(val_idx)
        print(test_idx)
        for idx in train_idx:
            image_path = os.path.join(
                image_root_path + str(class_name), image_list[idx])
            shutil.copy2(image_path, train_path)
            shutil.copy2(image_path, trainval_path)

        for idx in val_idx:
            image_path = os.path.join(
                image_root_path + str(class_name), image_list[idx])
            shutil.copy2(image_path, val_path)
            shutil.copy2(image_path, trainval_path)

        for idx in test_idx:
            image_path = os.path.join(
                image_root_path + str(class_name), image_list[idx])
            shutil.copy2(image_path, test_path)


if __name__ == '__main__':
    main()
