import enum
import torch
from torchvision import transforms
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
import torchvision.models as models
from utils import *
import torch.nn as nn
import argparse
# from efficientnet_pytorch import EfficientNet

classes = ['astilbe', 'bellflower', 'black-eyed susan', 'calendula', 'california poppy', 'tulip']
classes_index = {
    'astilbe': 0,
    'bellflower': 1,
    'black-eyed susan': 2,
    'calendula': 3,
    'california poppy': 4,
    'tulip': 5
}
images = []
labels = [] 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"USING DEVICE: {device}")




parser = argparse.ArgumentParser(description="Parameters...")
parser.add_argument("--dataset_path", default= r"C:\Users\admin\Documents\Python\FlowerClassification\Dataset\Train", type=str)
parser.add_argument("--valid_split", default=0.15, type=float)
parser.add_argument("--batch_size", default=8, type=int)
parser.add_argument("--lr", default=1e-4, type=float)
parser.add_argument("--num_epochs", default =20, type=int)
args = parser.parse_args()


for file in tqdm(os.listdir(args.dataset_path)):
    for img in tqdm(os.listdir(os.path.join(args.dataset_path, file))):
        images.append(img)
        labels.append(file)
        
print(f"NUMBER OF IMAGES: {len(images)}")

data = {'Images': images, 'labels': labels}
data = pd.DataFrame(data)


# lb = LabelEncoder()
data['encoded_labels'] = fit_transform(data['labels'])
print(data.head())

shuffle_dataset = True
random_seed = 42

dataset_size = len(data)
indices = list(range(dataset_size))
split = int(np.floor(args.valid_split * dataset_size))
if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)

train_indices, valid_indices = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(valid_indices)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])     

# transform = transforms.Compose([
#     transforms.Resize(224),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
# ])     


dataset = FlowerDataset(data, args.dataset_path, transform)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler)

valid_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=valid_sampler)

model = models.resnext101_32x8d(pretrained=True)
# model = models.resnext50_32x4d(pretrained=True)
model.fc = nn.Linear(2048, len(classes), bias=True)


# model = EfficientNet.from_name('efficientnet-b0')
# model._fc = nn.Linear(1280, len(classes))
print(model)
model = model.to(device)


param_optimizer = model.parameters()
optimizer = torch.optim.Adam(param_optimizer, lr=args.lr)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
loss_fn = nn.CrossEntropyLoss().to(device)
train(device=device, model=model, optimizer=optimizer, train_dataloader=train_loader, loss_fn=loss_fn, 
        valid_dataloader=valid_loader, num_epochs=args.num_epochs, 
        file_path=r"C:\Users\admin\Documents\Python\FlowerClassification")

