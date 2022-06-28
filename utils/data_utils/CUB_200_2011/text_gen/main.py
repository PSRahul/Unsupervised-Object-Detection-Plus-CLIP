import pandas as pd
import numpy as np


class DescGenerator():
    def __init__(self) -> None:

        attribute_definitions = "/home/psrahul/MasterThesis/datasets/CUB_200_2011/attributes.txt"
        attribute_certainities = "/home/psrahul/MasterThesis/datasets/CUB_200_2011/CUB_200_2011/attributes/certainties.txt"
        image_attributes = "/home/psrahul/MasterThesis/datasets/CUB_200_2011/CUB_200_2011/attributes/image_attribute_labels.txt"

        class_attribute_labels = "/home/psrahul/MasterThesis/datasets/CUB_200_2011/CUB_200_2011/attributes/class_attribute_labels_continuous.txt"

        self.attributes = pd.read_csv(attribute_definitions,
                                      sep=" ", header=None)

        # self.certainities = pd.read_csv(
        #    attribute_certainities, sep=" ", header=None)

        # self.image_attributes = pd.read_csv(
        #    image_attributes, sep=" ", header=None)

        self.class_attribute_labels = pd.read_csv(
            class_attribute_labels, sep=" ", header=None)

    def process(self):
        self.attribute_series_t5_list = []
        self.attribute_series_t3_list = []
        self.attribute_series_t1_list = []
        for idx in range(200):
            class_series = self.class_attribute_labels.iloc[idx].sort_values(
                ascending=False).keys()
            attribute_series_t5 = class_series[0:5]
            attribute_series_t3 = class_series[0:3]
            attribute_series_t1 = class_series[0]
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
            for attribute in list(attrib_list[idx].ravel()):
                # print(attribute)
                # print(self.attributes.iloc[int(attribute)])
                index_text.append(self.attributes.iloc[int(attribute)][1])
            self.attribute_series_text.append(index_text)

        return self.attribute_series_text

    def get_class_text(self):
        self.attribute_series_t5_text = self.get_index_class_list(
            self.attribute_series_t5_list)
        self.attribute_series_t3_text = self.get_index_class_list(
            self.attribute_series_t3_list)
        self.attribute_series_t1_text = self.get_index_class_list(
            self.attribute_series_t1_list)


def main():
    descgenerator = DescGenerator()
    descgenerator.process()
    descgenerator.get_class_text()
    print(descgenerator.attribute_series_t5_text)
    print(descgenerator.attribute_series_t3_text)
    print(descgenerator.attribute_series_t1_text)


if __name__ == "__main__":
    main()
