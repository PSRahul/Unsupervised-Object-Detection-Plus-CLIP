import torch
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import random
import pathlib


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

    def get_support_features(self, class_number, num_supports, seed, embed_size):

        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

        image_class_specifics = self.image_class_labels.loc[
            self.image_class_labels[1] == class_number + 1, 1
        ]
        image_index = list(image_class_specifics.keys())
        class_support_list = torch.zeros((num_supports, embed_size))
        image_index = np.random.permutation(image_index)
        image_index = image_index[0:num_supports]

        class_support_list = self.full_image_features[image_index, :]
        class_support_list = torch.mean(class_support_list, dim=0).reshape(
            1, embed_size
        )
        return class_support_list


def main():

    save_folder = "/home/psrahul/MasterThesis/repo/Unsupervised-Object-Detection-Plus-CLIP/features/images/support/v0.5"

    model_name = [
        ("CLIPModel", 512),
        ("DINOExtractorResNet", 2048),
        ("DINOExtractorViT", 768),
        ("DINOExtractorViTMultiScale", 768),
    ]

    for model, embed_size in model_name:

        save_folder_model = os.path.join(save_folder, model)

        select_supports = SelectSupports(
            feature_pt_path=os.path.join(
                save_folder_model, "image_features_support.pt"
            ),
            image_class_label_txt="/home/psrahul/MasterThesis/datasets/CUB_200_2011/v2/CUB_200_2011/train/image_class_labels.txt",
        )

        with tqdm(total=10 * 200 * 100) as pbar:

            # Number of Experiments
            for seed in tqdm(range(100)):
                np.random.seed(seed)
                torch.manual_seed(seed)
                random.seed(seed)

                # Number of Samples
                for j in tqdm(range(1, 11)):

                    support_features = torch.zeros((200, embed_size))

                    # Number of Classes
                    for i in tqdm(range(200)):

                        support_features[i] = select_supports.get_support_features(
                            class_number=i,
                            num_supports=j,
                            seed=seed,
                            embed_size=embed_size,
                        )

                        pbar.update(1)

                    save_folder_seed = os.path.join(
                        save_folder_model, "seed_" + str(seed)
                    )

                    pathlib.Path(save_folder_seed).mkdir(parents=True, exist_ok=True)

                    save_file_name = str(f"support_features_seed_{seed}_shot_{j}")
                    torch.save(
                        support_features, os.path.join(save_folder_seed, save_file_name)
                    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
