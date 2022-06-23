import os


def main():
    image_root_path = "/home/psrahul/MasterThesis/datasets/CUB_200_2011/CUB_200_2011/images/"
    class_names = sorted(os.listdir(image_root_path))

    for i in range(len(class_names)):
        class_name = class_names[i]
        print(class_name)


if __name__ == '__main__':
    main()
