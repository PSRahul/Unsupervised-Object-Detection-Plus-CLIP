# %%

from asyncio import FastChildWatcher
import os
import sys
import time
from json import load
from subprocess import call
import torch
import clip
import numpy as np
import torch
from clip import available_models, tokenize
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
from transformers import ViTFeatureExtractor, ViTModel




class DINOExtractor_ViT:
    def __init__(self):
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(256, interpolation=3),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        self.model = torch.hub.load("facebookresearch/dino:main", "dino_vitb16")
        self.model = self.model.cuda()

    def get_features(self, image):

        feature_extractor = ViTFeatureExtractor.from_pretrained("facebook/dino-vitb16")
        model = ViTModel.from_pretrained("facebook/dino-vitb16")
        inputs = feature_extractor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state

        image_feat = last_hidden_states.float()
        print(image_feat.shape)
        return image_feat


class CreateTokenFeatures:
    def __init__(self):
        pass

    def create_image_features_from_classes_folders(
        self,
        images_path,
        images_list,
        save_folder,
        save_name,
        crop_bounding_box=False,
        model="dino",
    ):

        file = open(images_list, "r")
        file_Lines = file.readlines()

        image_list = []
        image_features = torch.zeros((len(file_Lines), 2048))
        with torch.no_grad():
            if model == "clip":
                model, preprocess = clip.load("ViT-B/16")

            if model == "dino":
                dino_extractor = DINOExtractor_ViT()

        for i, file_line in enumerate(tqdm(file_Lines)):
            time.sleep(0.01)
            file_line = file_line.strip()
            idx, file_name = file_line.split(" ")
            image = Image.open(os.path.join(images_path, file_name)).convert("RGB")

            if crop_bounding_box:
                pred = np.load(
                    os.path.join(
                        "/home/psrahul/MasterThesis/repo/Unsupervised-Object-Detection-Plus-CLIP/bounding_box_regression/outputs/TokenCut/CUB200/bounding_boxes/TokenCut-vit_small16_k",
                        str(file_name.split("/")[1].split(".")[0] + ".npy"),
                    )
                )
                # print(pred)
                image = image.crop(pred)
            with torch.no_grad():
                if model == "clip":
                    image = torch.tensor(preprocess(image)).detach().unsqueeze(0).cuda()
                    image_feat = model.encode_image(image).float()

                if model == "dino":
                    image_feat = dino_extractor.get_features(image)

            image_features[i] = image_feat

        torch.save(image_features, os.path.join(save_folder, save_name))


def call_create_image_features_from_classes_folders():

    class_names_txt = (
        "/home/psrahul/MasterThesis/datasets/CUB_200_2011/CUB_200_2011/classes.txt"
    )
    images_path = "/home/psrahul/MasterThesis/datasets/CUB_200_2011/CUB_200_2011/train/CUB_200_2011/images/"
    images_list = "/home/psrahul/MasterThesis/datasets/CUB_200_2011/CUB_200_2011/train/CUB_200_2011/images.txt"
    save_folder = "/home/psrahul/MasterThesis/repo/Unsupervised-Object-Detection-Plus-CLIP/utils/data_utils/CUB_200_2011/image_features/support_image_features/"
    save_name = "image_features_support_images_cropped.pt"

    create_image_features_from_classes_folders(
        images_path, images_list, save_folder, save_name, crop_bounding_box=True
    )


def main():
    create_token_features = CreateTokenFeatures()

    create_token_features.create_image_features_from_classes_folders(
        images_path="/home/psrahul/MasterThesis/datasets/CUB_200_2011/v2/CUB_200_2011/train/images/",
        images_list="/home/psrahul/MasterThesis/datasets/CUB_200_2011/v2/CUB_200_2011/train/images.txt",
        save_folder="/home/psrahul/MasterThesis/repo/Unsupervised-Object-Detection-Plus-CLIP/utils/data_utils/CUB_200_2011/image_features/support_image_features/v2_split/dino/",
        save_name="image_features_support_images.pt",
        crop_bounding_box=False,
    )


if __name__ == "__main__":
    main()
