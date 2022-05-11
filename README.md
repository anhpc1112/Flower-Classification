# Flower-Classification
Flower Classification Using Resnext_101_32x8d pretrained model

# Overview
The Dataset includes 5198 images and they belong to 6 classes which are:
- astilbe
- bellflower
- black-eyed susan
- calendula
- california poppy 
- tulip
<br />
The Dataset is divided by the ratio of 85% for the training set and 15% for the test set. This model was trained on a MSI GF65 THIN RTX 3060 6GB machine. 
<br />

# Model architecture
- We use Resnext_101_32x8d pretrained model for Flower Classification
- Reference: https://paperswithcode.com/model/resnext?variant=resnext-101-32x8d

# Requirements
- Python version: 3.8.0
- Framework: Pytorch
- All libraries are located in the requirement.txt file

# How to train model
```
cd ./FLOWERCLASSIFICATION
pip install -r requirements.txt
python train.py --dataset_path [Path of dataset folder] --valid_split [validation dataset split ratio] --batch_size [batch_size] --lr [learning rate] --num_epochs [number of epochs]
```
### Description of parameters
| Parameter  | Default | Description |
| ------------- | ------------- | ------------- |
| dataset_path  | ./Train |Path of the dataset folder  |
| valid_split  | 0.15 | validation dataset split ratio |
| batch_size   | 8 | batch size |
| lr | 1e-4 | learning rate |
| num_epochs | 20 | number of epochs | 

### Structure of dataset folder:
./Train
<br />
├───astilbe
<br />
├───bellflower
<br />
├───black-eyed susan
<br />
├───calendula
<br />
├───california poppy
<br />
└───tulip

# How to estimate the accuracy of the model
```
cd ./FLOWERCLASSIFICATION
pip install -r requirements.txt
python evalution.py --test_path [Path of data test folder]
```
### Structure of data test folder:
./Train
<br />
├───astilbe
<br />
├───bellflower
<br />
├───black-eyed susan
<br />
├───calendula
<br />
├───california poppy
<br />
└───tulip


# Demo predict image
```
cd ./FLOWERCLASSIFICATION
pip install -r requirements.txt
python predict.py --model_path [Path of the model after training] --image_path [Path of image you want to predict]
```
