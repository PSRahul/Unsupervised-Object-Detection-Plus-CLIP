from statistics import mode
import sys

sys.path.append(
    "/home/psrahul/MasterThesis/repo/Unsupervised-Object-Detection-Plus-CLIP/CLIP/clip/"
)
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score
import os
from tqdm import tqdm
import pandas as pd


class GetSampleSimilarity:
    def __init__(
        self, query_feature_path, support_feature_path, target_image_labels_txt
    ):
        self.support_features = torch.load(support_feature_path)
        self.query_features = torch.load(query_feature_path)
        self.target_image_labels_txt = target_image_labels_txt

    def get_predictions(self):
        self.support_features /= self.support_features.norm(dim=-1, keepdim=True)
        self.query_features /= self.query_features.norm(dim=-1, keepdim=True)

        support_probs = (100.0 * self.query_features @ self.support_features.T).softmax(
            dim=-1
        )
        top_probs, top_labels = support_probs.cpu().topk(1, dim=-1)
        self.predictions = top_labels.numpy().ravel()
        self.predictions += 1

    def get_targets(self):

        file = open(self.target_image_labels_txt, "r")
        file_Lines = file.readlines()

        target_labels = []
        for file_line in file_Lines:
            file_line = file_line.strip()
            search_idx, search_label = file_line.split(" ")
            target_labels.append(int(search_label))

        self.targets = np.array(target_labels)

    def get_metrics(self):
        self.get_predictions()
        self.get_targets()

        accuracy = accuracy_score(self.targets, self.predictions) * 100
        f1 = f1_score(self.targets, self.predictions, average="macro") * 100

        return (accuracy, f1)


class ImageImageSim:
    def __init__(self, model_name, seed_number, shot_number):

        target_image_labels_txt = "/home/psrahul/MasterThesis/datasets/CUB_200_2011/v2/CUB_200_2011/test/image_class_labels.txt"

        query_root = "/home/psrahul/MasterThesis/repo/Unsupervised-Object-Detection-Plus-CLIP/features/images/query/v0.8"
        query_feature_path = os.path.join(
            query_root, str(model_name), "image_features_query.pt"
        )

        support_root = "/home/psrahul/MasterThesis/repo/Unsupervised-Object-Detection-Plus-CLIP/features/images/support/v0.8"
        support_feature_path = os.path.join(
            support_root, str(model_name), str(f"seed_{seed_number}")
        )
        support_file_name = str(
            f"support_features_seed_{seed_number}_shot_{shot_number}"
        )
        support_feature_path = os.path.join(support_feature_path, support_file_name)

        self.get_sample_similarity = GetSampleSimilarity(
            query_feature_path, support_feature_path, target_image_labels_txt
        )

    def get_metrics(self):
        accuracy_score, f1_score = self.get_sample_similarity.get_metrics()
        return (accuracy_score, f1_score)


def main():

    df_save_root = "/home/psrahul/MasterThesis/repo/Unsupervised-Object-Detection-Plus-CLIP/exps/v0.8/"
    model_name = [
        ("CLIPModel", 512),
        ("DINOExtractorResNet", 2048),
        ("DINOExtractorViT", 768),
        ("DINOExtractorViTMultiScale", 768),
    ]

    with tqdm(total=4 * 10 * 100) as pbar:

        # Number of Models
        for model, embed_dim in model_name:

            model_df_list = []
            # Number of Experiments
            for seed_number in range(100):

                # Number of Shots
                for shot_number in range(1, 11):

                    image_image_sim = ImageImageSim(model, seed_number, shot_number)
                    accuracy_score, f1_score = image_image_sim.get_metrics()

                    model_df_row = [seed_number, shot_number, accuracy_score, f1_score]
                    model_df_list.append(model_df_row)
                    pbar.update(1)

            model_df = pd.DataFrame(model_df_list)
            columns = ["SeedNumber", "ShotNumber", "Accuracy", "F1Score"]
            model_df.to_csv(
                os.path.join(df_save_root, f"{model}.txt"),
                sep=" ",
                index=False,
            )


if __name__ == "__main__":
    sys.exit(main())
