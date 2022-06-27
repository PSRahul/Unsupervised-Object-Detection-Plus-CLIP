from email.mime import image
import pandas as pd


class DescGenerator():
    def __init__(self) -> None:

        attribute_definitions = "/home/psrahul/MasterThesis/datasets/CUB_200_2011/attributes.txt"
        attribute_certainities = "/home/psrahul/MasterThesis/datasets/CUB_200_2011/CUB_200_2011/attributes/certainties.txt"
        image_attributes = "/home/psrahul/MasterThesis/datasets/CUB_200_2011/CUB_200_2011/attributes/image_attribute_labels.txt"

        class_attribute_labels = "/home/psrahul/MasterThesis/datasets/CUB_200_2011/CUB_200_2011/attributes/class_attribute_labels_continuous.txt"

        # self.attributes = pd.read_csv(attribute_definitions,
        #                              sep=" ", header=None)

        # self.certainities = pd.read_csv(
        #    attribute_certainities, sep=" ", header=None)

        # self.image_attributes = pd.read_csv(
        #    image_attributes, sep=" ", header=None)

        self.class_attribute_labels = pd.read_csv(
            class_attribute_labels, sep=" ", header=None)

    def process(self):
        # print(self.attributes.head())
        # print(self.image_attributes.head())
        # print(self.certainities.head())
        print(self.class_attribute_labels.head())


def main():
    descgenerator = DescGenerator()
    descgenerator.process()


if __name__ == "__main__":
    main()
