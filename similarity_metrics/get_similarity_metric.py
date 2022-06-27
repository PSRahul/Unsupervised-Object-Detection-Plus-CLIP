import sys
sys.path.append(
    "/home/psrahul/MasterThesis/repo/Unsupervised-Object-Detection-Plus-CLIP/CLIP/clip/")
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, multilabel_confusion_matrix, accuracy_score, balanced_accuracy_score
text_features = torch.load(
    "/home/psrahul/MasterThesis/repo/Unsupervised-Object-Detection-Plus-CLIP/CLIP/encodings/full_images_ThisisaPhoto/text_feature_full.pt")

image_features = torch.load(
    "/home/psrahul/MasterThesis/repo/Unsupervised-Object-Detection-Plus-CLIP/CLIP/encodings/full_images_ThisisaPhoto/image_features_full.pt")

text_features /= text_features.norm(dim=-1, keepdim=True)
image_features /= image_features.norm(dim=-1, keepdim=True)

text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
top_probs, top_labels = text_probs.cpu().topk(1, dim=-1)

image_label_txt = "/home/psrahul/MasterThesis/repo/Unsupervised-Object-Detection-Plus-CLIP/data_utils/CUB_200_2011/test/image_class_labels.txt"

file = open(image_label_txt, 'r')
file_Lines = file.readlines()

target_labels = []
for file_line in file_Lines:
    file_line = file_line.strip()
    search_idx, search_label = file_line.split(" ")
    target_labels.append(int(search_label))
    #print(search_idx, search_file_name)

# print(top_labels.shape)
# print(np.array(target_labels).shape)
predictions = top_labels.numpy().ravel()

target = np.array(target_labels)

# print(predictions.shape)
# print(target.shape)
#cm = confusion_matrix(target, predictions)

#cm_display = ConfusionMatrixDisplay(cm).plot()
# plt.show()

for i in range(2323):
    print(target[i], predictions[i])
print(accuracy_score(target, predictions))
