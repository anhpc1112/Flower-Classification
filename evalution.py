from sklearn.metrics import accuracy_score, f1_score, classification_report
import torch
import numpy as np
import argparse
import pandas as pd
from utils import *
from tqdm import tqdm

classes = ['astilbe', 'bellflower', 'black-eyed susan', 'calendula', 'california poppy', 'tulip']
classes_index = {
    'astilbe': 0,
    'bellflower': 1,
    'black-eyed susan': 2,
    'calendula': 3,
    'california poppy': 4,
    'tulip': 5
}

parser = argparse.ArgumentParser(description="Parameters...")
parser.add_argument("--test_path", default =r"C:\Users\admin\Documents\Python\FlowerClassification\Test\flowers", type=str)
args = parser.parse_args()

images = []
labels = [] 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"USING DEVICE: {device}")
for file in tqdm(os.listdir(args.test_path)):
    for img in tqdm(os.listdir(os.path.join(args.test_path, file))):
        images.append(img)
        labels.append(file)

print(f"NUMBER OF IMAGES: {len(images)}")

data = {'Images': images, 'labels': labels}
data = pd.DataFrame(data)

data['encoded_labels'] = fit_transform(data['labels'])
# print(data.head())

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])     

dataset = FlowerDataset(data, args.test_path, transform)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=32)


def predict(model, test_dataloader, device):
    y_pred = []
    y_true = []
    model.eval()
    for image, label in test_dataloader:
        label = label.to(device)
        image = image.to(device)
        with torch.no_grad():
          outputs = model(image)
        logits = outputs.detach().cpu().numpy()
        pred_flat = np.argmax(logits, axis=1).flatten()
        y_pred.extend(pred_flat.tolist())

        labels = label.cpu().numpy().flatten()
        y_true.extend(labels.tolist())
        
    print('Classification Report:')
    print(classification_report(y_true, y_pred, digits=4))
    return accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average='macro')


model = torch.load('model.pth')
print(predict(model, test_loader, device))