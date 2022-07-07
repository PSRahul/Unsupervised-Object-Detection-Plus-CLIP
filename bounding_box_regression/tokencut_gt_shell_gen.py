import subprocess
import os

image_root_path = (
    "/home/psrahul/MasterThesis/datasets/CUB_200_2011/v2/CUB_200_2011/train/images/"
)

image_list_path = (
    "/home/psrahul/MasterThesis/datasets/CUB_200_2011/v2/CUB_200_2011/train/images.txt"
)

bounding_box_output = "/home/psrahul/MasterThesis/repo/Unsupervised-Object-Detection-Plus-CLIP/bounding_box_regression/TokenCutOutputs/v05_support_script.txt"

file = open(image_list_path, "r")
file_Lines = file.readlines()

with open(bounding_box_output, "w") as h:
    for file_line in file_Lines:
        file_line = file_line.strip()
        file_line = file_line.split(" ", 1)[1]
        # print(os.path.join(image_root_path, file_line))
        line_str = (
            "python /home/psrahul/MasterThesis/repo/Unsupervised-Object-Detection-Plus-CLIP/TokenCut/main_tokencut.py --image_path "
            + str(os.path.join(image_root_path, file_line))
            + " --visualize pred"
            + " --output_dir /home/psrahul/MasterThesis/repo/Unsupervised-Object-Detection-Plus-CLIP/bounding_box_regression/TokenCutOutputs/bounding_boxes/ --save_predictions True  \n"
        )
        print(line_str)
        h.write(line_str)
