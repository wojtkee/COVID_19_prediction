import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import torchvision.models as models
import os
# from os import listdir
from torch.utils.data import  Dataset, DataLoader
from PIL import Image
import time 
import matplotlib.pyplot as plt
from collections import Counter
import gc
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import csv

label_to_class = {
    "Covid": 0,
    "Normal": 1,
    "Viral Pneumonia": 2
}

class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None, convert2gray=True, target_resolution=(432, 432)):
        self.data = []
        self.labels = []
        self.convert_to_gray = convert2gray
        self.target_resolution = target_resolution
        self.class_counts = {class_label: 0 for class_label in label_to_class.values()}
        class_folders = os.listdir(data_dir)

        for folder_name in class_folders:
            folder_path = os.path.join(data_dir, folder_name)
            class_label = label_to_class[folder_name]
            for img_name in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_name)
                img = Image.open(img_path)

                if self.convert_to_gray:
                    img = img.convert('L')  # konwersja na gray

                img = img.resize(self.target_resolution, Image.LANCZOS)

                self.data.append(img)
                self.labels.append(class_label)
                self.class_counts[class_label] += 1

        print(f"Loaded: {sum(self.class_counts.values())} images ")
        for class_label, count in self.class_counts.items():
            print(f"class {class_label}: {count} images ")

        # Define the transformation
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1) if self.convert_to_gray else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]) if self.convert_to_gray else transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

    def get_class_counters(self):
        return self.class_counts, self.target_resolution