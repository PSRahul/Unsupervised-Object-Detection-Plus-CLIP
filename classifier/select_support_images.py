import torch


class SelectSupports():
    def __init__(self, feature_pt_path):

        self.load_full_feature_vector(feature_pt_path)

    def load_full_feature_vector(self, feature_pt_path):

        self.full_image_features = torch.load(feature_pt_path)


select_supports = SelectSupports(
    feature_pt_path="/home/psrahul/MasterThesis/repo/Unsupervised-Object-Detection-Plus-CLIP/utils/data_utils/CUB_200_2011/image_features/support_image_features/image_features_support_images.pt")
