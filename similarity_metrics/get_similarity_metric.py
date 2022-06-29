import sys
sys.path.append(
    "/home/psrahul/MasterThesis/repo/Unsupervised-Object-Detection-Plus-CLIP/CLIP/clip/")
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, multilabel_confusion_matrix, accuracy_score, balanced_accuracy_score, f1_score


class GetSimilarity():
    def __init__(self, query_feature_path, support_feature_path):
        self.support_features = torch.load(support_feature_path)
        self.query_features = torch.load(query_feature_path)

    def get_predictions(self):
        self.support_features /= self.support_features.norm(
            dim=-1, keepdim=True)
        self.query_features /= self.query_features.norm(dim=-1, keepdim=True)

        support_probs = (100.0 * self.query_features @
                         self.support_features.T).softmax(dim=-1)
        top_probs, top_labels = support_probs.cpu().topk(1, dim=-1)
        self.predictions = top_labels.numpy().ravel()

    def get_targets(self):

        query_label_txt = "/home/psrahul/MasterThesis/repo/Unsupervised-Object-Detection-Plus-CLIP/utils/data_utils/CUB_200_2011/data_splits/test/image_class_labels.txt"

        file = open(query_label_txt, 'r')
        file_Lines = file.readlines()

        target_labels = []
        for file_line in file_Lines:
            file_line = file_line.strip()
            search_idx, search_label = file_line.split(" ")
            target_labels.append(int(search_label))

        self.targets = np.array(target_labels)

    def get_accuracy_score(self):
        self.get_predictions()
        self.get_targets()
        print("Accuracy {0:.2f}".format(accuracy_score(
            self.targets, self.predictions) * 100))

    def get_f1_score(self):
        self.get_predictions()
        self.get_targets()
        print("F1 Score {0:.2f}".format(
            f1_score(self.targets, self.predictions, average="macro") * 100))


support_feature_path = "//home/psrahul/MasterThesis/repo/Unsupervised-Object-Detection-Plus-CLIP/utils/data_utils/CUB_200_2011/text_features_v2/" + \
    "text_feature_just_class_names.pt"
#   "text_feature_just_class_names.pt"
# "text_feature_t5_text.pt"
# "text_feature_t3_text.pt"
# "text_feature_t1_text.pt"

query_feature_path = "/home/psrahul/MasterThesis/repo/Unsupervised-Object-Detection-Plus-CLIP/utils/data_utils/CUB_200_2011/image_features/" + \
    "image_feature_cropped.pt"
# "query_feature_full.pt"
# "query_feature_cropped.pt"

get_similarity = GetSimilarity(support_feature_path=support_feature_path,
                               query_feature_path=query_feature_path)

get_similarity.get_accuracy_score()
get_similarity.get_f1_score()
