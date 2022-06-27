# %%
from clip import tokenize
import numpy as np
import torch
import os
from PIL import Image
from clip import available_models
import sys
import clip
# %%


def get_unique_encodings(class_name_list):
    for i in range(len(class_name_list)):
        print(class_name_list[i], class_token_list[i],
              np.sum(np.array(class_token_list[i])))

    print(np.array(class_token_list).shape)

    print(len(class_token_list))
    print(np.unique(np.sum(np.array(class_token_list), axis=1)).shape)


# %%


def create_token_images():
    class_names_txt = "/home/psrahul/MasterThesis/datasets/CUB_200_2011/CUB_200_2011/classes.txt"
    class_name_list = []
    class_token_list = []
    file = open(class_names_txt, 'r')
    file_Lines = file.readlines()
    for file_line in file_Lines:
        file_line = file_line.strip()
        class_name = file_line.split(".")[1]
        class_name_list.append(class_name)
        class_token_list.append(
            tokenize("This is " + str(class_name)).numpy().ravel())

    class_token_list = np.array(class_token_list)
    text_tokens = torch.from_numpy(class_token_list)

    torch.save(text_tokens, "/home/psrahul/MasterThesis/repo/Unsupervised-Object-Detection-Plus-CLIP/CLIP/encodings/text_tokens.pt")
    # %%
    image_root_path = "/home/psrahul/MasterThesis/datasets/CUB_200_2011/CUB_200_2011/test/"
    train_image_path = "/home/psrahul/MasterThesis/repo/Unsupervised-Object-Detection-Plus-CLIP/data_utils/CUB_200_2011/test/images.txt"
    class_names = sorted(os.listdir(image_root_path))
    file = open(train_image_path, 'r')
    file_Lines = file.readlines()
    model, preprocess = clip.load("ViT-B/16")
    image_list = []
    for file_line in file_Lines:
        file_line = file_line.strip()
        idx, file_name = file_line.split(" ")
        image = Image.open(os.path.join(
            image_root_path, file_name)).convert("RGB")
        image_list.append(preprocess(image))

    image_input = torch.tensor(np.stack(image_list)).cuda()

    torch.save(image_input, "/home/psrahul/MasterThesis/repo/Unsupervised-Object-Detection-Plus-CLIP/CLIP/encodings/processed_images.pt")


# %%
# create_token_images()


def get_text_features():
    text_tokens = torch.load(
        "/home/psrahul/MasterThesis/repo/Unsupervised-Object-Detection-Plus-CLIP/CLIP/encodings/text_tokens_cropped.pt")

    model, preprocess = clip.load("ViT-B/16")
    text_features = torch.zeros((text_tokens.shape[0], 512))

    with torch.no_grad():
        for i in range(text_tokens.shape[0]):
            text_token = text_tokens[i].cuda().reshape((1, 77))
            text_feat = model.encode_text(text_token).float()
            text_features[i] = text_feat

    torch.save(text_features, "/home/psrahul/MasterThesis/repo/Unsupervised-Object-Detection-Plus-CLIP/CLIP/encodings/text_feature_cropped.pt")


def get_image_features():
    image_input = torch.load(
        "/home/psrahul/MasterThesis/repo/Unsupervised-Object-Detection-Plus-CLIP/CLIP/encodings/processed_images_cropped.pt")

    model, preprocess = clip.load("ViT-B/16")
    image_features = torch.zeros((image_input.shape[0], 512))
    with torch.no_grad():
        for i in range(image_input.shape[0]):
            image = image_input[i].unsqueeze(0).cuda()
            image_feat = model.encode_image(image).float()
            image_features[i] = image_feat

    torch.save(image_features, "/home/psrahul/MasterThesis/repo/Unsupervised-Object-Detection-Plus-CLIP/CLIP/encodings/image_features_cropped.pt")


# create_token_images()
# get_text_features()
# get_image_features()


def create_token_images_with_bounding_boxes():
    class_names_txt = "/home/psrahul/MasterThesis/datasets/CUB_200_2011/CUB_200_2011/classes.txt"
    class_name_list = []
    class_token_list = []
    file = open(class_names_txt, 'r')
    file_Lines = file.readlines()
    for file_line in file_Lines:
        file_line = file_line.strip()
        class_name = file_line.split(".")[1]
        class_name_list.append(class_name)
        class_token_list.append(
            tokenize("This is a photo of a " + str(class_name)).numpy().ravel())

    class_token_list = np.array(class_token_list)
    text_tokens = torch.from_numpy(class_token_list)

    torch.save(text_tokens, "/home/psrahul/MasterThesis/repo/Unsupervised-Object-Detection-Plus-CLIP/CLIP/encodings/text_tokens_cropped.pt")
    # %%
    image_root_path = "/home/psrahul/MasterThesis/datasets/CUB_200_2011/CUB_200_2011/test/CUB_200_2011/images/"
    train_image_path = "/home/psrahul/MasterThesis/repo/Unsupervised-Object-Detection-Plus-CLIP/data_utils/CUB_200_2011/test/images.txt"
    class_names = sorted(os.listdir(image_root_path))
    file = open(train_image_path, 'r')
    file_Lines = file.readlines()
    model, preprocess = clip.load("ViT-B/16")
    image_list = []
    for file_line in file_Lines:
        file_line = file_line.strip()
        idx, file_name = file_line.split(" ")
        print(os.path.join(
            image_root_path, file_name))
        image = Image.open(os.path.join(
            image_root_path, file_name)).convert("RGB")
        # image.show()
        # print(file_name.split("/")[1].split(".")[0])
        pred = np.load(os.path.join(
            "/home/psrahul/MasterThesis/repo/Unsupervised-Object-Detection-Plus-CLIP/datasets/CUB_200_2011/bounding_boxes/TokenCut-vit_small16_k/", str(file_name.split("/")[1].split(".")[0] + ".npy")))
        # print(pred)
        image_cropped = image.crop(pred)
        image_list.append(preprocess(image_cropped))

    image_input = torch.tensor(np.stack(image_list)).cuda()

    torch.save(image_input, "/home/psrahul/MasterThesis/repo/Unsupervised-Object-Detection-Plus-CLIP/CLIP/encodings/processed_images_cropped.pt")


# create_token_images_with_bounding_boxes()

create_token_images_with_bounding_boxes()
get_text_features()
get_image_features()
