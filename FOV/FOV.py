import torch.nn as nn

class Dense(nn.Module):
    """
    Fully connected layer used in FOV
    """
    def __init__(self, in_units, out_units):
        super().__init__()
        self.fc=nn.Linear(in_units, out_units)
        self.relu=nn.ReLU()

    def forward(self,x):
        return self.relu(self.fc(x))

class FOV(nn.Module):
    """
    Predicts the field of view (upper and lower boundary)
    given an image
    """
    def __init__(self):
        super().__init__()
        self.conv1=nn.Sequential(nn.Conv2d(1,16,3),
                                nn.ReLU(),
                                nn.MaxPool2d(2))
        self.conv2=nn.Sequential(nn.Conv2d(16,32,3),
                                nn.ReLU(),
                                nn.MaxPool2d(2))
        self.conv3=nn.Sequential(nn.Conv2d(32,64,3),
                                nn.ReLU(),
                                nn.MaxPool2d(2))

        self.conv4=nn.Sequential(nn.Conv2d(64,64,3),
                                nn.ReLU(),
                                nn.MaxPool2d(2))
        self.flatten=nn.Flatten()
        self.fc1=Dense(576,100)
        self.fc2=Dense(100,50)
        self.fc3=Dense(50,2)

    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        for i in range(3): 
            x=self.conv4(x)
        x=self.flatten(x)
        x=self.fc1(x)
        x=self.fc2(x)
        x=self.fc3(x)

        return x