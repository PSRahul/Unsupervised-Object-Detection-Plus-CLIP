import torch
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm

np.random.seed(26)
torch.manual_seed(26)


class SelectSupports:
    def __init__(self, feature_pt_path, image_class_label_txt):

        self.load_full_feature_vector(feature_pt_path)
        self.load_image_class_labels(image_class_label_txt)

    def load_full_feature_vector(self, feature_pt_path):
        self.full_image_features = torch.load(feature_pt_path)

    def load_image_class_labels(self, image_class_label_txt):
        self.image_class_labels = pd.read_csv(
            image_class_label_txt, sep=" ", header=None
        )

    def get_support_features(self, class_number, num_supports):
        image_class_specifics = self.image_class_labels.loc[
            self.image_class_labels[1] == class_number + 1, 1
        ]
        image_index = list(image_class_specifics.keys())
        class_support_list = torch.zeros((num_supports, 2048))
        image_index = np.random.permutation(image_index)
        image_index = image_index[0:num_supports]

        class_support_list = self.full_image_features[image_index, :]
        class_support_list = torch.mean(class_support_list, dim=0).reshape(1, 2048)
        return class_support_list


def main():

    save_folder = "/home/psrahul/MasterThesis/repo/Unsupervised-Object-Detection-Plus-CLIP/utils/data_utils/CUB_200_2011/image_features/support_image_features/v2_split/dino/support_features_k_"

    select_supports = SelectSupports(
        feature_pt_path="/home/psrahul/MasterThesis/repo/Unsupervised-Object-Detection-Plus-CLIP/utils/data_utils/CUB_200_2011/image_features/support_image_features/v2_split/dino/image_features_support_images.pt",
        image_class_label_txt="/home/psrahul/MasterThesis/datasets/CUB_200_2011/v2/CUB_200_2011/train/image_class_labels.txt",
    )

    with tqdm(total=10 * 200) as pbar:

        for j in tqdm(range(1, 11)):

            support_features = torch.zeros((200, 2048))
            for i in tqdm(range(200)):
                support_features[i] = select_supports.get_support_features(
                    class_number=i, num_supports=j
                )
                pbar.update(1)

            torch.save(support_features, save_folder + str(j))

    return 0


if __name__ == "__main__":
    sys.exit(main())
