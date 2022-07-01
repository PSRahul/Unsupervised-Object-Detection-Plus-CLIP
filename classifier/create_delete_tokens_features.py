# %%

from asyncio import FastChildWatcher
import os
import sys
import time
from json import load
from subprocess import call

import clip
import numpy as np
import torch
from clip import available_models, tokenize
from PIL import Image
from tqdm import tqdm


class CreateTokenFeatures:
    def __init__(self):
        pass

    def get_text_token_from_desc_list(
        self, load_name, save_name, load_file_root, save_file_root
    ):

        class_token_list = []

        desc_list = torch.load(os.path.join(load_file_root, load_name))
        for desc in desc_list:
            class_token_list.append(tokenize(str(desc)).numpy().ravel())

        class_token_list = np.array(class_token_list)
        text_tokens = torch.from_numpy(class_token_list)

        torch.save(text_tokens, os.path.join(save_file_root, save_name))

    def create_image_features_from_classes_folders(
        self, images_path, images_list, save_folder, save_name, crop_bounding_box=False
    ):

        file = open(images_list, "r")
        file_Lines = file.readlines()
        model, preprocess = clip.load("ViT-B/16")
        image_list = []
        image_features = torch.zeros((len(file_Lines), 512))

        for i, file_line in enumerate(tqdm(file_Lines)):
            time.sleep(0.1)
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
                image = torch.tensor(preprocess(image)).detach().unsqueeze(0).cuda()
                image_feat = model.encode_image(image).float()
            image_features[i] = image_feat

        torch.save(image_features, os.path.join(save_folder, save_name))

    #
    def get_image_features(self):
        image_input = torch.load(
            "/home/psrahul/MasterThesis/repo/Unsupervised-Object-Detection-Plus-CLIP/CLIP/encodings/processed_images_cropped.pt"
        )

        model, preprocess = clip.load("ViT-B/16")
        image_features = torch.zeros((image_input.shape[0], 512))
        with torch.no_grad():
            for i in tqdm(range(image_input.shape[0])):
                image = image_input[i].unsqueeze(0).cuda()
                image_feat = model.encode_image(image).float()
                image_features[i] = image_feat

        torch.save(
            image_features,
            "/home/psrahul/MasterThesis/repo/Unsupervised-Object-Detection-Plus-CLIP/CLIP/encodings/image_features_cropped.pt",
        )

    def get_text_features(self, load_name, save_name, file_root):

        text_tokens = torch.load(os.path.join(file_root, load_name))

        model, preprocess = clip.load("ViT-B/16")
        text_features = torch.zeros((text_tokens.shape[0], 512))

        with torch.no_grad():
            for i in range(text_tokens.shape[0]):
                text_token = text_tokens[i].cuda().reshape((1, 77))
                text_feat = model.encode_text(text_token).float()
                text_features[i] = text_feat

        torch.save(text_features, os.path.join(file_root, save_name))


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


def call_get_text_features():
    file_root = "/home/psrahul/MasterThesis/repo/Unsupervised-Object-Detection-Plus-CLIP/utils/data_utils/CUB_200_2011/text_gen/text_tokens_v2/"
    get_text_features(
        load_name="text_token_just_class_names.pt",
        save_name="text_feature_just_class_names.pt",
    )
    get_text_features(
        load_name="text_token_t5_text.pt", save_name="text_feature_t5_text.pt"
    )
    get_text_features(
        load_name="text_token_t3_text.pt", save_name="text_feature_t3_text.pt"
    )
    get_text_features(
        load_name="text_token_t1_text.pt", save_name="text_feature_t1_text.pt"
    )


def call_get_text_token_from_desc_list():
    load_file_root = "/home/psrahul/MasterThesis/repo/Unsupervised-Object-Detection-Plus-CLIP/utils/data_utils/CUB_200_2011/text_gen/text_tokens_v2/"
    save_file_root = "/home/psrahul/MasterThesis/repo/Unsupervised-Object-Detection-Plus-CLIP/utils/data_utils/CUB_200_2011/text_gen/text_tokens_v2/"

    get_text_token_from_desc_list(
        load_name="just_class_names.pt",
        save_name="text_token_just_class_names.pt",
        load_file_root=load_file_root,
        save_file_root=save_file_root,
    )
    get_text_token_from_desc_list(
        load_name="t5_text.pt",
        save_name="text_token_t5_text.pt",
        load_file_root=load_file_root,
        save_file_root=save_file_root,
    )
    get_text_token_from_desc_list(
        load_name="t3_text.pt",
        save_name="text_token_t3_text.pt",
        load_file_root=load_file_root,
        save_file_root=save_file_root,
    )
    get_text_token_from_desc_list(
        load_name="t1_text.pt",
        save_name="text_token_t1_text.pt",
        load_file_root=load_file_root,
        save_file_root=save_file_root,
    )


def main():
    create_token_features = CreateTokenFeatures()
    """
    create_token_features.create_image_features_from_classes_folders(
        images_path="/home/psrahul/MasterThesis/datasets/PASCAL_VOC2007/Reformatted_Single_Object/test/images/",
        images_list="/home/psrahul/MasterThesis/datasets/PASCAL_VOC2007/Reformatted_Single_Object/test/images.txt",
        save_folder="/home/psrahul/MasterThesis/repo/Unsupervised-Object-Detection-Plus-CLIP/utils/data_utils/PASCAL VOC 2007/image_features/",
        save_name="test_features.pt",
        crop_bounding_box=False,
    )
    
    create_token_features.get_text_token_from_desc_list(
        load_name="just_class_names.pt",
        save_name="just_class_names.pt",
        load_file_root="/home/psrahul/MasterThesis/repo/Unsupervised-Object-Detection-Plus-CLIP/utils/data_utils/PASCAL VOC 2007/text_tokens/",
        save_file_root="/home/psrahul/MasterThesis/repo/Unsupervised-Object-Detection-Plus-CLIP/utils/data_utils/PASCAL VOC 2007/text_embeddings/",
    )
    pass
    
    create_token_features.get_text_features(
        load_name="just_class_names.pt",
        save_name="just_class_names_features.pt",
        file_root="/home/psrahul/MasterThesis/repo/Unsupervised-Object-Detection-Plus-CLIP/utils/data_utils/PASCAL VOC 2007/text_embeddings/",
    )
    """


if __name__ == "__main__":
    main()
