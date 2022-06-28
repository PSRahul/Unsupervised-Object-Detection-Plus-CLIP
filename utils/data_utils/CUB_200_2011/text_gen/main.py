import re
import pandas as pd
import numpy as np
import torch
import os

import sys


class DescGenerator():
    def __init__(self, filter=True) -> None:

        attribute_definitions = "/home/psrahul/MasterThesis/datasets/CUB_200_2011/attributes.txt"
        attribute_certainities = "/home/psrahul/MasterThesis/datasets/CUB_200_2011/CUB_200_2011/attributes/certainties.txt"
        image_attributes = "/home/psrahul/MasterThesis/datasets/CUB_200_2011/CUB_200_2011/attributes/image_attribute_labels.txt"

        class_names = "/home/psrahul/MasterThesis/datasets/CUB_200_2011/CUB_200_2011/classes.txt"

        class_attribute_labels = "/home/psrahul/MasterThesis/datasets/CUB_200_2011/CUB_200_2011/attributes/class_attribute_labels_continuous.txt"

        self.attributes = pd.read_csv(attribute_definitions,
                                      sep=" ", header=None)

        self.class_names = pd.read_csv(class_names,
                                       sep=" ", header=None)
        # self.certainities = pd.read_csv(
        #    attribute_certainities, sep=" ", header=None)

        # self.image_attributes = pd.read_csv(
        #    image_attributes, sep=" ", header=None)

        self.class_attribute_labels = pd.read_csv(
            class_attribute_labels, sep=" ", header=None)

        self.filter = filter

    def filter_attributes(self):
        keep_filters = []
        # print(self.attributes)
        # We choose Wing Color, Eye Color, Primary color, Leg Color, Bill Color
        # Wing Color - 9 - 24
        # Eye Color - 135 - 150
        # Primary Color - 248 - 263
        # Leg Color - 263 - 278
        # Bill Color - 278 - 293

        keep_filters.append([int(x)
                             for x in np.arange(9, 24)])
        keep_filters.append([int(x)
                             for x in np.arange(135, 149)])
        keep_filters.append([int(x)
                             for x in np.arange(248, 293)])
        keep_filters = [x for xs in keep_filters for x in xs]

        self.class_attribute_labels = self.class_attribute_labels.iloc[:, keep_filters]
        self.attributes = self.attributes.iloc[keep_filters, :]

    def process(self):
        if(self.filter):
            self.filter_attributes()
        self.attribute_series_t15_list = []
        self.attribute_series_t5_list = []
        self.attribute_series_t3_list = []
        self.attribute_series_t1_list = []
        for idx in range(200):
            class_series = self.class_attribute_labels.iloc[idx].sort_values(
                ascending=False).keys()
            attribute_series_t15 = class_series[0:15]
            attribute_series_t5 = class_series[0:5]
            attribute_series_t3 = class_series[0:3]
            attribute_series_t1 = class_series[0]
            self.attribute_series_t15_list.append(
                np.array(attribute_series_t15))
            self.attribute_series_t5_list.append(
                np.array(attribute_series_t5))
            self.attribute_series_t3_list.append(
                np.array(attribute_series_t3))
            self.attribute_series_t1_list.append(
                np.array(attribute_series_t1))

    def __repr__(self):
        print(self.attribute_series_t5_list)
        print("\n")
        print(self.attribute_series_t3_list)
        print("\n")
        print(self.attribute_series_t1_list)

        return " "

    def get_index_class_list(self, attrib_list):
        self.attribute_series_text = []

        for idx in range(200):
            index_text = []
            # print(attrib_list)
            print(self.attributes.index)
            for attribute in list(attrib_list[idx].ravel()):

                print(attribute)
                # print(self.attributes.iloc[int(attribute)])
                index_text.append(self.attributes[int(attribute), 1])
            self.attribute_series_text.append(index_text)

        return self.attribute_series_text

    def get_class_text(self):
        self.attribute_series_t15_text = self.get_index_class_list(
            self.attribute_series_t15_list)
        self.attribute_series_t5_text = self.get_index_class_list(
            self.attribute_series_t5_list)
        self.attribute_series_t3_text = self.get_index_class_list(
            self.attribute_series_t3_list)
        self.attribute_series_t1_text = self.get_index_class_list(
            self.attribute_series_t1_list)

    def convert_list_to_str(self, desc_list):
        desc_str = ""
        for each_str in desc_list:
            desc_str += str(each_str) + ". "
        return desc_str

    def make_sentences(self, attribute_series_tk_text, save_str, save_root="/home/psrahul/MasterThesis/repo/Unsupervised-Object-Detection-Plus-CLIP/utils/data_utils/CUB_200_2011/text_gen/text_tokens_v2/", filter=True):
        class_texts = []
        class_texts_with_desc = []
        for idx in range(200):
            attribute_series_index = attribute_series_tk_text[idx]
            class_str = self.class_names.iloc[idx][1].split(".")[
                1].replace("_", " ")
            class_text = "This is a Photo of a " + str(class_str)
            class_texts.append(class_text)
            desc_text = []
            desc_text.append(class_text)
            for attribute in attribute_series_index:
                # ['has_eye_color', 'black']
                part, color = (attribute.split("::", 2))
                color = color.replace("_", " ")
                part = part.replace("_", " ")  # has underparts color
                part = part.split(" ", 1)
                attribute_text = "It " + part[0] + " " + color + " " + part[1]
                desc_text.append(attribute_text)
            desc_text = self.convert_list_to_str(desc_text)
            print(desc_text)
            class_texts_with_desc.append(desc_text)
        torch.save(class_texts, os.path.join(save_root, "just_class_names.pt"))
        torch.save(class_texts_with_desc, os.path.join(save_root, save_str))


def main():
    descgenerator = DescGenerator()
    descgenerator.process()
    descgenerator.get_class_text()
    descgenerator.make_sentences(
        descgenerator.attribute_series_t15_text, save_str="t15_text.pt")
    descgenerator.make_sentences(
        descgenerator.attribute_series_t5_text, save_str="t5_text.pt")
    descgenerator.make_sentences(
        descgenerator.attribute_series_t3_text, save_str="t3_text.pt")
    descgenerator.make_sentences(
        descgenerator.attribute_series_t1_text, save_str="t1_text.pt")


if __name__ == "__main__":
    main()
