from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as func
from PIL import Image
import os


# Defining path for the data
path = "PATH TO DATA FOLDER" # Fill in the path
train_path = path + "Train/"
test_path = path + "Test/"
result_path = "Result/"

train_data = os.listdir(train_path + "Train Data")
train_label = os.listdir(train_path + "Train Label")
test_data = os.listdir(test_path + "Test Data")
test_label = os.listdir(test_path + "Test Label")

class dataset(Dataset):
    """
    Dataset for train/test images

    Images are resized to 320 x 320
    """
    def __init__(self, test=False, transf=transforms.ToTensor()):
        super().__init__()
        self.test = test
        self.trans = transf
        if test:
            self.data = test_data
            self.labels = test_label
        else:
            self.data = train_data
            self.labels = train_label

    def __len__(self):
            return len(self.data)

    def __getitem__(self, idx):
        data_name = self.data[idx]
        if self.test:
            data = Image.open(test_path + "Test Data/" + data_name)
            label = Image.open(test_path + "Test Label/" + data_name)
            data = data.resize((320, 320))
            label = func.rgb_to_grayscale(label, num_output_channels=1) # convert label to grayscale
            label = label.resize((320,320))
            return self.trans(data), self.trans(label), data_name
        else:
            data = Image.open(train_path + "Train Data/" + data_name)
            label = Image.open(train_path + "Train Label/" + data_name)
            data = data.resize((320, 320))
            label = func.rgb_to_grayscale(label, num_output_channels=1)
            label = label.resize((320,320))
            return self.trans(data), self.trans(label), data_name