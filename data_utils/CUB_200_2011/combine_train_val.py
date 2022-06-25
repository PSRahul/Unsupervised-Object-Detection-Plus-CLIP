text_file_path = "/home/psrahul/MasterThesis/repo/Unsupervised-Object-Detection-Plus-CLIP/data_utils/CUB_200_2011/train_test_split.txt"

file = open(text_file_path, 'r')
file_Lines = file.readlines()

with open('trainval/train_test_split.txt', 'w') as h:
    for file_line in file_Lines:
        file_line = file_line.strip()
        target = str(file_line.split(" ", 1)[1])
        index = str(file_line.split(" ", 1)[0])
        print(index)
        print(target)
        write_line = str(int(index) + 7142) + " " + str(target) + '\n'
        h.write(write_line)
