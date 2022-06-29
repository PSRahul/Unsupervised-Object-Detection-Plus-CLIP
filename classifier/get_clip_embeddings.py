# %%
from json import load
from clip import tokenize
import numpy as np
import torch
import os
from PIL import Image
from clip import available_models
import sys
import clip


def get_text_features(load_name, save_name, file_root="/home/psrahul/MasterThesis/repo/Unsupervised-Object-Detection-Plus-CLIP/utils/data_utils/CUB_200_2011/text_gen/text_tokens_v2/"):

    text_tokens = torch.load(os.path.join(file_root, load_name))

    model, preprocess = clip.load("ViT-B/16")
    text_features = torch.zeros((text_tokens.shape[0], 512))

    with torch.no_grad():
        for i in range(text_tokens.shape[0]):
            text_token = text_tokens[i].cuda().reshape((1, 77))
            text_feat = model.encode_text(text_token).float()
            text_features[i] = text_feat

    torch.save(text_features, os.path.join(file_root, save_name))


get_text_features(load_name="text_token_just_class_names.pt",
                  save_name="text_feature_just_class_names.pt")
get_text_features(load_name="text_token_t5_text.pt",
                  save_name="text_feature_t5_text.pt")
get_text_features(load_name="text_token_t3_text.pt",
                  save_name="text_feature_t3_text.pt")
get_text_features(load_name="text_token_t1_text.pt",
                  save_name="text_feature_t1_text.pt")
