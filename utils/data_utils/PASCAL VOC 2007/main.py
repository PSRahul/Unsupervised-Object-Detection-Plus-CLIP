import sys
from pascal_utils import PascalUtils


class CreateSubsetFeature:
    def __init__(self):
        self.pascal_utils = PascalUtils(
            root_dir="/home/psrahul/MasterThesis/datasets/PASCAL_VOC2007/VOCdevkit/VOC2007/"
        )

    def process(self):
        print(self.pascal_utils.class_names)

    def get_class_df(self):
        pass


class CreateSupportImagesCropped:
    def __init__(self):
        pass

    def create_support_images_cropped(self):
        pass


def main():
    create_subset_feature = CreateSubsetFeature()
    create_subset_feature.process()


if __name__ == "__main__":
    sys.exit(main())
