import yaml
import clip
import torch
import os
from tqdm import tqdm
from PIL import Image
import pandas as pd
from torchvision.io import read_image

#! DEVICE DEFINE
device = "cuda" if torch.cuda.is_available() else "cpu"

#! File list retreive function
def list_files(folder_path, file_extension):
    result_dict = {}
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            if file_name.endswith(file_extension):
                file_path = os.path.join(root, file_name)
                # get the second last folder name
                parent_folder = os.path.basename(os.path.dirname(file_path))
                # add the file path to the list of files for this folder
                result_dict[file_path] = parent_folder
    return result_dict

#! FUNCTION DEFINITION
# Zeroshot
def zeroshot_classifier(classnames, templates):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates] #format with class
            texts = clip.tokenize(texts).cuda() #tokenize
            
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights

#! ACCURACY CALCULATE
def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]        

#! DATASET LOAD
class CustomImageDataset():
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

#! DATA READING
# prompts YAML
with open("Data/prompts.yml", 'r') as stream:
    model_prompt = yaml.safe_load(stream)
# models YAML
with open("Data/models.yml", 'r') as stream:
    models = yaml.safe_load(stream)

# load model and image preprocessing
model, transform = clip.load("ViT-B/32", device=device, jit=False)
images = CustomImageDataset(annotations_file="label.csv",img_dir="Dataset/Test/deepfake", transform=transform)

loader = torch.utils.data.DataLoader(images, batch_size=32, num_workers=2)
zeroshot_weights = zeroshot_classifier(model_prompt["FaceForensic++"]["classes"], model_prompt["FaceForensic++"]["templates"])

# with torch.no_grad():
#     top1, top5, n = 0., 0., 0.
#     for i, (images, target) in enumerate(loader):
#         images = images.cuda()
#         target = target.cuda()
        
#         # predict
#         image_features = model.encode_image(images)
#         image_features /= image_features.norm(dim=-1, keepdim=True)
#         logits = 100. * image_features @ zeroshot_weights

#         # measure accuracy
#         acc1, acc5 = accuracy(logits, target, topk=(1, 5))
#         top1 += acc1
#         top5 += acc5
#         n += images.size(0)

# top1 = (top1 / n) * 100
# result = str(round(top1,2))
