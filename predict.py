from utils import *
import numpy as np
import torchvision.models as models
import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

classes = ['astilbe', 'bellflower', 'black-eyed susan', 'calendula', 'california poppy', 'tulip']
classes_index = {
    'astilbe': 0,
    'bellflower': 1,
    'black-eyed susan': 2,
    'calendula': 3,
    'california poppy': 4,
    'tulip': 5
}
index_classes = {
    0: 'astilbe',
    1: 'bellflower',
    2: 'black-eyed susan',
    3: 'calendula',
    4: 'california poppy',
    5: 'tulip'
}
parser = argparse.ArgumentParser(description="Evaluation...")
parser.add_argument("--model_path", default=r"C:\Users\admin\Documents\Python\FlowerClassification\modelImage.pt", type=str)
parser.add_argument("--image_path", type=str)
args = parser.parse_args()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])     

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"USING DEVICE: {device}")



# print(model)
# load_checkpoint(args.model_path, model, device)
model = torch.load('model.pth')



def predictImage(image_path, transform, model, device):
    image = Image.open(image_path)
    image = image.convert('RGB')
    image = image.resize((256,256))
    image = transform(image)
    image = torch.unsqueeze(image, 0)
    image = image.to(device)
    model = model.to(device)    
    with torch.no_grad():
        output = model(image)
        print(output)
    
    return index_classes[np.argmax(output.detach().cpu().numpy())]
print(predictImage(args.image_path, transform, model, device))






