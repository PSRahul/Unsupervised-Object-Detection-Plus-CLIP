from posixpath import split
import sys
from tqdm import tqdm


class RemoveOverlapImages:
    def __init__(self):
        self.source_images_path = "/home/psrahul/MasterThesis/repo/Unsupervised-Object-Detection-Plus-CLIP/utils/data_utils/CUB_200_2011/data_splits/v2/images.txt"
        self.remove_images_path = "//home/psrahul/MasterThesis/repo/Unsupervised-Object-Detection-Plus-CLIP/utils/data_utils/CUB_200_2011/data_splits/v2/cub_imagenet_overlap.txt"

        self.target_images_path = "/home/psrahul/MasterThesis/repo/Unsupervised-Object-Detection-Plus-CLIP/utils/data_utils/CUB_200_2011/data_splits/v2/filtered_images.txt"

        self.source_images = open(self.source_images_path, "r")
        self.source_images = self.source_images.readlines()

        self.remove_images = open(self.remove_images_path, "r")
        self.remove_images = self.remove_images.readlines()

    def remove_entries_and_save(self):
        with open(self.target_images_path, "w") as self.target_images:
            for source_image in tqdm(self.source_images):
                flag = 0
                split_source_image = source_image.split("/")[1].strip()
                print(split_source_image)
                for remove_image in self.remove_images:
                    split_remove_image = remove_image.split(",")[0].strip()
                    if str(split_remove_image) == str(split_source_image):
                        flag = 1
                if flag == 0:
                    self.target_images.write(source_image)


def main():
    remove_overlap_images = RemoveOverlapImages()
    remove_overlap_images.remove_entries_and_save()


if __name__ == "__main__":
    sys.exit(main())
