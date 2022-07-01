from email.mime import image
from operator import index
import sys
from pascal_utils import _load_data, list_image_sets
from tqdm import tqdm
import pandas as pd
import os
import shutil
import pathlib


class CreateSubsetCSV:
    def __init__(self):
        self.class_names = list_image_sets()
        self.splits = ["train", "val", "test"]

    def process(self):
        for class_name in tqdm(self.class_names):
            print(class_name)
            for split in self.splits:
                _load_data(class_name, split)

    def get_class_names(self):
        return self.class_names


class GenerateTxtFiles:
    def __init__(self, class_names, split):

        self.split = split
        self.root_csv_dir = (
            "/home/psrahul/MasterThesis/datasets/PASCAL_VOC2007/VOCdevkit/VOC2007/csvs/"
        )
        self.save_csv_root = os.path.join(
            "/home/psrahul/MasterThesis/datasets/PASCAL_VOC2007/Reformatted/", split
        )
        self.load_image_path = "/home/psrahul/MasterThesis/datasets/PASCAL_VOC2007/VOCdevkit/VOC2007/JPEGImages/"

        self.save_image_path = os.path.join(
            "/home/psrahul/MasterThesis/datasets/PASCAL_VOC2007/Reformatted/",
            split,
            "images",
        )

        self.class_names = class_names

        self.file_index = 1

    def copy_image(self, source, target):
        shutil.copy2(source, target)

    def generate_images_txt(self, copy_images=True):

        self.images_txt = []
        self.bounding_box_txt = []
        self.image_class_labels_txt = []

        for class_idx, class_name in tqdm(enumerate(self.class_names, 1)):
            print(class_name)
            pascal_csv = pd.read_csv(
                os.path.join(self.root_csv_dir, self.split + "_" + class_name + ".csv")
            )
            for i in tqdm(range(len(pascal_csv))):
                bounding_box_row_list = [
                    self.file_index,
                    # pascal_csv.loc[i, "fname"],
                    pascal_csv.loc[i, "xmin"],
                    pascal_csv.loc[i, "ymin"],
                    pascal_csv.loc[i, "xmax"],
                    pascal_csv.loc[i, "ymax"],
                ]
                self.bounding_box_txt.append(bounding_box_row_list)

                images_row_list = [
                    self.file_index,
                    str(class_name + "/" + pascal_csv.loc[i, "fname"]),
                ]

                self.images_txt.append(images_row_list)

                image_class_label_row_list = [self.file_index, class_idx]
                self.image_class_labels_txt.append(image_class_label_row_list)

                self.file_index += 1
                if copy_images:
                    source = os.path.join(
                        self.load_image_path, pascal_csv.loc[i, "fname"]
                    )
                    pathlib.Path(
                        os.path.join(
                            self.save_image_path,
                            str(class_name + "/"),
                        )
                    ).mkdir(parents=True, exist_ok=True)

                    target = os.path.join(
                        self.save_image_path,
                        str(class_name + "/" + pascal_csv.loc[i, "fname"]),
                    )

                    self.copy_image(source, target)

            images_df = pd.DataFrame(self.images_txt)
            bounding_box_df = pd.DataFrame(self.bounding_box_txt)
            image_class_label_box_df = pd.DataFrame(self.image_class_labels_txt)

            images_df.to_csv(
                os.path.join(
                    self.save_csv_root,
                    "images.txt",
                ),
                sep=" ",
                index=False,
            )
            bounding_box_df.to_csv(
                os.path.join(self.save_csv_root, "bounding_boxes.txt"),
                sep=" ",
                index=False,
            )
            image_class_label_box_df.to_csv(
                os.path.join(self.save_csv_root, "image_class_labels.txt"),
                sep=" ",
                index=False,
            )

            # pint(images_df)


def main():
    # Create Class Level Splits CSV
    create_subset_csv = CreateSubsetCSV()
    generate_txt_files = GenerateTxtFiles(create_subset_csv.get_class_names(), "test")
    generate_txt_files.generate_images_txt()


if __name__ == "__main__":
    sys.exit(main())
