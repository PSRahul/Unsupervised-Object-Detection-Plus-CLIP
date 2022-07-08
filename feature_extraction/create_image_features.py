import os
from re import M
import sys
import time
from json import load
from subprocess import call
from typing import MutableMapping

import clip
import numpy as np
import torch
from clip import available_models, tokenize
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
import torch.nn as nn
import pathlib


class CLIPModel:
    def __init__(self):
        self.model, self.preprocess = clip.load("ViT-B/16")
        self.model = self.model.cuda().eval()

    def get_features(self, image):
        image = torch.tensor(self.preprocess(image)).detach().unsqueeze(0).cuda()
        image_feat = self.model.encode_image(image).float()
        return image_feat


class DINOExtractorResNet:
    def __init__(self):
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(256, interpolation=3),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        self.model = torch.hub.load("facebookresearch/dino:main", "dino_resnet50")
        self.model = self.model.cuda().eval()

    def get_features(self, image):
        image = torch.tensor(self.preprocess(image)).detach().unsqueeze(0).cuda()
        image_feat = self.model(image).float()
        return image_feat


class DINOExtractorViT:
    def __init__(self, multiscale):
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(256, interpolation=3),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        self.multiscale = multiscale
        self.model = torch.hub.load("facebookresearch/dino:main", "dino_vitb16")
        self.model = self.model.cuda().eval()

    def multi_scale(self, image):
        v = None
        for s in [1, 1 / 2 ** (1 / 2), 1 / 2]:  # we use 3 different scales
            if s == 1:
                inp = image.clone()
            else:
                inp = nn.functional.interpolate(
                    image, scale_factor=s, mode="bilinear", align_corners=False
                )
            feats = self.model(inp).clone()
            if v is None:
                v = feats
            else:
                v += feats
        v /= 3
        v /= v.norm()
        return v

    def get_features(self, image):

        image = torch.tensor(self.preprocess(image)).detach().unsqueeze(0).cuda()
        if self.multiscale:
            image_feat = self.multi_scale(image)
        else:
            image_feat = self.model(image).clone()

        return image_feat


class CreateImageFeatures:
    def __init__(
        self,
        images_path,
        images_list,
        crop_tokencut_bbox,
        crop_gt_bbox,
        model_name,
        save_folder,
        save_name,
    ):
        self.create_image_features_from_classes_folders(
            images_path,
            images_list,
            crop_tokencut_bbox,
            crop_gt_bbox,
            model_name,
        )
        self.save_feature(save_folder, save_name)

    @torch.no_grad()
    def create_image_features_from_classes_folders(
        self, images_path, images_list, crop_tokencut_bbox, crop_gt_bbox, model_name
    ):

        file = open(images_list, "r")
        file_Lines = file.readlines()

        if crop_gt_bbox:
            bbox_file = open(
                "/home/psrahul/MasterThesis/datasets/CUB_200_2011/v2/CUB_200_2011/train/bounding_boxes.txt",
                "r",
            )
            bbox_file = bbox_file.readlines()

        if model_name == "CLIPModel":
            model = CLIPModel()
            embed_size = 512

        if model_name == "DINOExtractorResNet":
            model = DINOExtractorResNet()
            embed_size = 2048

        if model_name == "DINOExtractorViT":
            model = DINOExtractorViT(multiscale=False)
            embed_size = 768

        if model_name == "DINOExtractorViTMultiScale":
            model = DINOExtractorViT(multiscale=True)
            embed_size = 768

        image_list = []
        image_features = torch.zeros((len(file_Lines), embed_size))

        for i, file_line in enumerate(tqdm(file_Lines)):
            time.sleep(0.01)
            file_line = file_line.strip()
            idx, file_name = file_line.split(" ")
            image = Image.open(os.path.join(images_path, file_name)).convert("RGB")

            if crop_tokencut_bbox:
                pred = np.load(
                    os.path.join(
                        "/home/psrahul/MasterThesis/repo/Unsupervised-Object-Detection-Plus-CLIP/bounding_box_regression/TokenCutOutputs/bounding_boxes/TokenCut-vit_small16_k/",
                        str(file_name.split("/")[1].split(".")[0] + ".npy"),
                    )
                )
                # print(pred)
                image = image.crop(pred)
                # print(pred)
            if crop_gt_bbox:
                pred = bbox_file[i].split()
                pred = [
                    int(float(pred[1])),
                    int(float(pred[2])),
                    int(float(pred[1])) + int(float(pred[3])),
                    int(float(pred[2])) + int(float(pred[4])),
                ]
                # print(pred)
                # sys.exit(0)
                image = image.crop(pred)

            with torch.no_grad():
                image_feat = model.get_features(image)

            image_features[i] = image_feat
        self.image_features = image_features

    def save_feature(self, save_folder, save_name):
        torch.save(self.image_features, os.path.join(save_folder, save_name))


def main():
    images_path = (
        "/home/psrahul/MasterThesis/datasets/CUB_200_2011/v2/CUB_200_2011/test/images/"
    )
    images_list = "/home/psrahul/MasterThesis/datasets/CUB_200_2011/v2/CUB_200_2011/test/images.txt"
    save_root = "/home/psrahul/MasterThesis/repo/Unsupervised-Object-Detection-Plus-CLIP/features/images/query/v0.6"
    save_name = "image_features_query.pt"
    crop_bbox_tokencut = True
    crop_gt_bbox = False
    model_name = [
        "CLIPModel",
        "DINOExtractorResNet",
        "DINOExtractorViT",
        "DINOExtractorViTMultiScale",
    ]
    # model_name = "DINOExtractorResNet"
    # model_name = "DINOExtractorViT"
    # model_name = "DINOExtractorViTMultiScale"

    for model in model_name:
        save_folder = os.path.join(save_root, model)
        pathlib.Path(save_folder).mkdir(parents=True, exist_ok=True)

        _ = CreateImageFeatures(
            images_path,
            images_list,
            crop_bbox_tokencut,
            crop_gt_bbox,
            model,
            save_folder,
            save_name,
        )


if __name__ == "__main__":
    sys.exit(main())
