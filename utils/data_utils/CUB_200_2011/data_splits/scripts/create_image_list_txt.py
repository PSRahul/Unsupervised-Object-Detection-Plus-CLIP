import os

root_path = "/home/psrahul/MasterThesis/datasets/CUB_200_2011/v2/CUB_200_2011/test/"

image_root_path = os.path.join(root_path, "images/")

class_names = sorted(os.listdir(image_root_path))
counter = 1
with open(os.path.join(root_path, "images.txt"), "w") as f:
    with open(os.path.join(root_path, "image_class_labels.txt"), "w") as g:
        with open(os.path.join(root_path, "train_test_split.txt"), "w") as h:
            for i in range(len(class_names)):
                class_name = class_names[i]
                image_list = sorted(os.listdir(image_root_path + str(class_name)))
                for idx in range(len(image_list)):
                    image_path = os.path.join(str(class_name), image_list[idx])
                    write_line_1 = str(counter) + " " + str(image_path) + "\n"
                    write_line_2 = str(counter) + " " + str(int(image_path[:3])) + "\n"
                    write_line_3 = str(counter) + " " + str(0) + "\n"
                    counter += 1
                    f.write(write_line_1)
                    g.write(write_line_2)
                    h.write(write_line_3)
