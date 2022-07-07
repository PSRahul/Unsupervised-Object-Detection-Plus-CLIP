import sys

sys.path.append(
    "/home/psrahul/MasterThesis/repo/Unsupervised-Object-Detection-Plus-CLIP/CLIP/clip/"
)
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
)


class GetSimilarity:
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

        print(
            "Accuracy {0:.2f}".format(
                accuracy_score(self.targets, self.predictions) * 100
            )
        )r
        print(
            "F1 Score {0:.2f}".format(
                f1_score(self.targets, self.predictions, average="macro") * 100
            )
        )


def text_image_similarity(
    support_feature_path, query_feature_path, target_image_labels_txt
):

    get_similarity = GetSimilarity(
        support_feature_path=support_feature_path,
        query_feature_path=query_feature_path,
        target_image_labels_txt=target_image_labels_txt,
    )

    get_similarity.get_metrics()


def image_image_similarity():
    support_feature_path = (
        "/home/psrahul/MasterThesis/repo/Unsupervised-Object-Detection-Plus-CLIP/utils/data_utils/CUB_200_2011/image_features/support_image_features/v2_split/dino/"
        + "support_features_k_1"
    )

    query_feature_path = (
        "/home/psrahul/MasterThesis/repo/Unsupervised-Object-Detection-Plus-CLIP/utils/data_utils/CUB_200_2011/image_features/test_image_features/v2_split/dino/"
        + "test_features_full.pt"
    )

    target_image_labels_txt = "/home/psrahul/MasterThesis/datasets/CUB_200_2011/v2/CUB_200_2011/test/image_class_labels.txt"

    get_similarity = GetSimilarity(
        support_feature_path=support_feature_path,
        query_feature_path=query_feature_path,
        target_image_labels_txt=target_image_labels_txt,
    )

    get_similarity.get_metrics()


def main():

    image_image_similarity()
    """
    support_feature_path = (
        "//home/psrahul/MasterThesis/repo/Unsupervised-Object-Detection-Plus-CLIP/utils/data_utils/CUB_200_2011/text_features_v2/"
        + "text_feature_just_class_names.pt"
    )
    #   "text_feature_just_class_names.pt"
    # "text_feature_t5_text.pt"
    # "text_feature_t3_text.pt"
    # "text_feature_t1_text.pt"

    query_feature_path = (
        "/home/psrahul/MasterThesis/repo/Unsupervised-Object-Detection-Plus-CLIP/utils/data_utils/CUB_200_2011/image_features/test_image_features/v2_split/"
        + "test_features_full.pt"
    )
    # "query_feature_full.pt"
    # "query_feature_cropped.pt"

    target_image_labels_txt = "/home/psrahul/MasterThesis/datasets/CUB_200_2011/v2/CUB_200_2011/test/image_class_labels.txt"

    text_image_similarity(
        query_feature_path=query_feature_path,
        support_feature_path=support_feature_path,
        target_image_labels_txt=target_image_labels_txt,
    )

    
    text_image_similarity(
        query_feature_path="/home/psrahul/MasterThesis/repo/Unsupervised-Object-Detection-Plus-CLIP/utils/data_utils/PASCAL VOC 2007/image_features/test_features.pt",
        support_feature_path="/home/psrahul/MasterThesis/repo/Unsupervised-Object-Detection-Plus-CLIP/utils/data_utils/PASCAL VOC 2007/text_embeddings/just_class_names_features.pt",
        target_image_labels_txt="/home/psrahul/MasterThesis/datasets/PASCAL_VOC2007/Reformatted_Single_Object/test/image_class_labels.txt",
    )
    """


if __name__ == "__main__":
    sys.exit(main())
