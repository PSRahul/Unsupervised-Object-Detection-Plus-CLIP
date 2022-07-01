import sys
from pascal_utils import _load_data, list_image_sets
from tqdm import tqdm


class CreateSubsetCSV:
    def __init__(self):
        self.class_names = list_image_sets()
        self.splits = ["train", "val", "test"]
        self.process()

    def process(self):
        for class_name in tqdm(self.class_names):
            print(class_name)
            for split in self.splits:
                _load_data(class_name, split)


class CreateSupportImagesCropped:
    def __init__(self):
        pass

    def create_support_images_cropped(self):
        pass


def main():
    # Create Class Level Splits CSV
    #create_subset_csv = CreateSubsetCSV()


if __name__ == "__main__":
    sys.exit(main())
