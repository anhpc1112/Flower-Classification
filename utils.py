import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import os
from PIL import Image

classes = ['astilbe', 'bellflower', 'black-eyed susan', 'calendula', 'california poppy', 'tulip']
classes_index = {
    'astilbe': 0,
    'bellflower': 1,
    'black-eyed susan': 2,
    'calendula': 3,
    'california poppy': 4,
    'tulip': 5
}

def fit_transform(labels):
    index = []
    for label in labels:
        index.append(classes_index[label])
    return index

class FlowerDataset(Dataset):
    def __init__(self, img_data, img_path, transform=None):
        self.img_path = img_path
        self.transform = transform
        self.img_data = img_data
         
    def __len__(self):
        return len(self.img_data)
    
    def __getitem__(self, index):
        img_name = os.path.join(self.img_path,self.img_data.loc[index, 'labels'],
                                self.img_data.loc[index, 'Images'])
        image = Image.open(img_name)
        image = image.convert('RGB')
        image = image.resize((256,256))
        label = torch.tensor(self.img_data.loc[index, 'encoded_labels']).long()
        if self.transform is not None:
            image = self.transform(image)
        return image, label

def save_checkpoint(save_path, model, valid_loss):
    if save_path == None:
        return
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'valid_loss': valid_loss
    }
    torch.save(checkpoint, save_path)
    print(f"Checkpoint is saved ==> {save_path}")

def save_model(save_path, model):
    if save_path == None:
        return
    torch.save(model, save_path)
    print(f"Model is saved ==> {save_path}")


def load_checkpoint(load_path, model, device):
    if load_path == None:
        return
    
    checkpoint = torch.load(load_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return checkpoint['valid_loss']

def save_metrics(save_path, train_loss_list, valid_loss_list):
    if save_path == None:
        return
    metrics = {
        'train_loss_list': train_loss_list,
        'valid_loss_list': valid_loss_list
    }
    torch.save(metrics, save_path)
    print(f"Metrics is saved ==> {save_path}")

def load_metrics(load_path):
    if load_path == None:
        return
    metrics = torch.load(load_path)
    return metrics['train_loss_list'], metrics['valid_loss_list']

def evalution(y_preds, y_true):
    y_preds = y_preds.detach().cpu().numpy()
    y_preds = np.argmax(y_preds, axis=1).flatten()

    y_true = y_true.cpu().numpy().flatten()

    return accuracy_score(y_true, y_preds), f1_score(y_true, y_preds, average='macro')

def predict(model, test_dataloader, device):
    y_pred = []
    y_true = []
    model.eval()
    for img, lb in test_dataloader:
        lb = lb.to(device)
        img = img.to(device)
        with torch.no_grad():
            outputs = model(img)
        logits = outputs.detach().cpu().numpy()
        pred_flat = np.argmax(logits, axis=1).flatten()
        y_pred.extend(pred_flat.tolist())

        labels = lb.cpu().numpy().flatten()
        y_true.extend(labels.tolist())

    print('Classification Report:')
    print(classification_report(y_true, y_pred, digits=4))
    return accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average='macro')

def train(model, optimizer, train_dataloader,loss_fn, valid_dataloader, num_epochs, file_path, device):
    train_loss_list = []
    valid_loss_list = []

    best_valid_loss = 99999
    best_f1 = 0

    for epoch in range(num_epochs):
        print("====== Epoch: {}/{} =======".format(epoch+1, num_epochs))
    
        total_loss = 0
        train_accuracy = 0
        train_f1 = 0
        nb_train_steps = 0
    
        model.train()
        for step, (image, label) in enumerate(train_dataloader):
            label = label.to(device)
            image = image.to(device)
            model.zero_grad()
            outputs = model(image)

            loss = loss_fn(outputs, label)
            total_loss += loss.item()

            tmp_train_accuracy, tmp_train_f1 = evalution(outputs, label)

            train_accuracy += tmp_train_accuracy
            train_f1 += tmp_train_f1
            nb_train_steps +=1

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step % 100 == 0 or step == len(train_dataloader):
                print("[TRAIN] EPOCH: {} | BATCH: {}/{} | TRAIN_LOSS: {} | TRAIN_ACC: {}".format(epoch+1, step, len(train_dataloader), loss.item(), tmp_train_accuracy))
        
        avg_train_loss = total_loss/len(train_dataloader)
        train_loss_list.append(avg_train_loss)

        print("Train Accuracy: {}".format(train_accuracy/nb_train_steps))
        print("Train F1 Score: {}".format(train_f1/ nb_train_steps))
        print("Train Loss: {}".format(avg_train_loss))

        print("Running Validation....")
        model.eval()

        eval_loss = 0
        eval_accuracy = 0
        eval_f1 = 0
        nb_eval_steps = 0

        for (image, label) in valid_dataloader:
            label = label.to(device)
            image = image.to(device)
            with torch.no_grad():
                outputs = model(image)

            loss = loss_fn(outputs, label)
            eval_loss += loss.item()

            tmp_eval_accuracy, tmp_f1_score = evalution(outputs, label)

            eval_accuracy += tmp_eval_accuracy
            eval_f1 += tmp_f1_score
            nb_eval_steps += 1
    
        avg_eval_loss = eval_loss/nb_eval_steps
        print("Valid loss: {}".format(avg_eval_loss))
        print("Valid Accuracy: {}".format(eval_accuracy/nb_eval_steps))
        print("Valid F1 Score: {}".format(eval_f1 /nb_eval_steps))

        valid_loss_list.append(avg_eval_loss)

        if eval_f1 /nb_eval_steps > best_f1:
            best_f1 = eval_f1 /nb_eval_steps
            save_checkpoint(file_path+'/checkpoint.pt', model, best_valid_loss)
            save_model(file_path+'/model.pth', model)

    save_metrics(file_path + '/metrics.pt',train_loss_list, valid_loss_list)
    print("Fnished Training!")