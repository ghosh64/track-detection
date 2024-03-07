import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch
import matplotlib.pyplot as plt
import os

from FOV import FOV
from FOV_dataset import dataset
from train_model import test, train_model
import matplotlib.pyplot as plt


def main():
    # Data loader
    device = torch.device("cuda:0")
    batch_size = 8
    train_data = dataset()
    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_data = dataset(val=True)
    val_data_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

    # Train model
    model = FOV()
    loss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    epochs = 200
    loss_list = train_model(model, train_data_loader, loss, optimizer, epochs, val_data_loader)

    # Test model
    test_data = dataset(test=True)
    test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    #test model
    loss = nn.MSELoss()
    test_loss = test(model, test_data_loader, loss)

    result_path = "PATH TO STORE PREDICT RESULT" # Fill in the path
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    model.eval()
    for data, _, img_name in test_data_loader:
        model.to(device)
        data = data.to(device)
        labels = labels.to(device)
        predictions = model(data)
        # Display prediction result
        for i in range(len(predictions)):
            predictions = predictions.cpu().detach().numpy()
            data = data.cpu().detach().numpy()
            
            plt.figure()
            test_img = data[i].permute(1,2,0).cpu().detach().numpy()
            y = predictions[i]
            plt.imshow(test_img)
            plt.axhline(y = y[0], color = 'w', linestyle = '--')
            plt.axhline(y = y[1], color = 'r', linestyle = 'dashed')
            plt.tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False)
            plt.savefig(result_path + "FOV_" + img_name[i])

            plt.close()


if __name__ == "__main__":
    main()