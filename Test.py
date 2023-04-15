import os
import clip
import torch
import pandas as pd
from torchvision.io import read_image
from PIL import Image
#! DATASET LOADER
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
        # image = read_image(img_path)
        image = Image.open(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

#! TEXT PROMPTS
text_classes = ["real", "fake"]

#! LOADING STUFFS
# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

# Load image and text
image_inputs = CustomImageDataset(annotations_file="Dataset/deepfake/0/labels_real.csv", img_dir="Dataset/deepfake/0",transform=preprocess)
text_inputs = torch.cat([clip.tokenize(f"this is a {c} photo") for c in text_classes]).to(device)


#! MAIN WORKING  STUFFS
# Extract feature
with torch.no_grad():
    image_features = model.encode_image(image_inputs[2])
    text_features = model.encode_text(text_inputs)

# Classify
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
values, indices = similarity[0].topk(1)
print("\nTop predictions:\n")
for value, index in zip(values, indices):
    print(f"{text_classes[index]:>16s}: {100 * value.item():.2f}%")