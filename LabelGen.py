import os
import csv

def list_files(folder_path):
    result = []
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            result_list = []
            file_path = os.path.join(root, file_name)
            # get the second last folder name
            label = os.path.basename(os.path.dirname(file_path))
            # add the file path to the list of files for this folder
            result_list.append(file_name)
            result_list.append(label)
            result.append(result_list)
    return result
images_dict = list_files("Dataset/Test/deepfake")
with open('Dataset/Test/deepfake/label.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(images_dict)
