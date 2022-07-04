import os

root_path = "/home/psrahul/MasterThesis/datasets/CUB_200_2011/v2/CUB_200_2011/test/"

train_image_path = os.path.join(root_path, "images.txt")

save_bounding_box_path = os.path.join(root_path, "bounding_boxes.txt")

full_image_path = (
    "/home/psrahul/MasterThesis/datasets/CUB_200_2011/v2/CUB_200_2011/images.txt"
)

full_bounding_box_path = "/home/psrahul/MasterThesis/datasets/CUB_200_2011/v2/CUB_200_2011/bounding_boxes.txt"


file1 = open(train_image_path, "r")
file1_Lines = file1.readlines()

file2 = open(full_image_path, "r")
file2_Lines = file2.readlines()

file3 = open(full_bounding_box_path, "r")
file3_Lines = file3.readlines()

with open(save_bounding_box_path, "w") as h:
    for file1_line in file1_Lines:
        file1_line = file1_line.strip()
        search_idx, search_file_name = file1_line.split(" ")
        for file2_line in file2_Lines:
            file2_line = file2_line.strip()
            target_idx, target_file_name = file2_line.split(" ")
            if search_file_name == target_file_name:
                file3_line = file3_Lines[int(target_idx) - 1].strip()
                bbox = str(file3_line.split(" ", 1)[1])
                write_line_3 = str(search_idx) + " " + bbox + "\n"
                h.write(write_line_3)
            # print(search_idx, target_idx)
