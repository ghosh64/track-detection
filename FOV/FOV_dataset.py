from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as func
from PIL import Image
import os
import numpy as np
import pandas as pd

# Defining path for the data
path = "PATH TO DATA FOLDER" # Fill in the path
train_path = path + "train/"
test_path = path + "test/"
valid_path = path + "valid/"

train_data = os.listdir(train_path + "images")
train_label = os.listdir(train_path + "labels")
test_data = os.listdir(test_path + "images")
test_label = os.listdir(test_path + "labels")
valid_data = os.listdir(valid_path + "images")
valid_label = os.listdir(valid_path + "labels")

class dataset(Dataset):
    """
    Dataset for train/test images

    Images are resized to 320 x 320
    """
    def __init__(self, val=False, test=False, transf=transforms.ToTensor()):
        super().__init__()
        self.trans = transf
        self.mode='val' if val else 'test' if test else 'train'
        if val:
            self.data = valid_data
            self.path = valid_path
            self.label = valid_label
        elif test:
            self.data = test_data
            self.path = test_path
            self.label = test_label
        else:
            self.data = train_data
            self.path = train_path
            self.label = train_label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_name = self.data[idx][:-4] # remove ".jpg"
        data = Image.open(str(self.path) + "images/" + str(data_name) + ".jpg")
        data = data.resize((320, 320)) # Resize image to pass into model
        label_data = pd.read_csv(str(self.path) + "labels/" + str(data_name) + ".txt", delimiter=' ', header=None) #cls xc yc w h
        label = np.array([(label_data.iloc[0,2]-label_data.iloc[0,4]/2)*320, (label_data.iloc[0,2]+label_data.iloc[0,4]/2)*320]) # label is a two element vector as [y_c-h/2,y_c+h/2]
        return self.trans(data), label, data_name