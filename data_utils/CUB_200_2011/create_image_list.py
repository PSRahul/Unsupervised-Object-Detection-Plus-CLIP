import os

image_root_path = "/home/psrahul/MasterThesis/datasets/CUB_200_2011/CUB_200_2011/train/"
class_names = sorted(os.listdir(image_root_path))
with open('images.txt', 'w') as f:
    for i in range(len(class_names)):
        class_name = class_names[i]
        image_list = os.listdir(image_root_path + str(class_name))
        for idx in range(len(image_list)):
            image_path = os.path.join(str(class_name), image_list[idx])
            f.write(str(image_path) +'\n')
