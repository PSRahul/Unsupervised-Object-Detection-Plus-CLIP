import torch
import os


class CreateTokens:
    def __init__(self):
        self.class_names = [
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "diningtable",
            "dog",
            "horse",
            "motorbike",
            "person",
            "pottedplant",
            "sheep",
            "sofa",
            "train",
            "tvmonitor",
        ]
        self.save_root = "/home/psrahul/MasterThesis/repo/Unsupervised-Object-Detection-Plus-CLIP/utils/data_utils/PASCAL VOC 2007/text_tokens/"

    def save_token(self):
        class_texts = []
        for class_name in self.class_names:
            class_str = "This is a Photo of a " + str(class_name)
            class_texts.append(class_str)

        torch.save(class_texts, os.path.join(self.save_root, "just_class_names.pt"))


create_tokens = CreateTokens()
create_tokens.save_token()
