import torch.nn as nn
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
import os

from webacnn import WeBACNN
from dataset import dataset
from train_model import train_model

def main():
    model = WeBACNN()
    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-2)

    # Training dataset
    train_data_obj = dataset()
    train_data_load = DataLoader(train_data_obj, batch_size=8, shuffle=True, drop_last=True)

    # Train the model
    loss_list = train_model(model, train_data_load, loss, optimizer, epochs=50)

    # Evaluate the model against test data
    model.eval()

    # Test dataset
    test_data_obj = dataset(test=True)
    test_data_loader = DataLoader(test_data_obj, batch_size=8, shuffle=False)

    result_path = "PATH TO STORE PREDICT RESULT" # Fill in the path
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    # Do not need to test label (only predict on test dataset)
    for data, _, img_name in test_data_loader:
        device = torch.device('cuda:0')
        data = data.to(device)
        predictions, _, _ = model(data)

        # Display prediction result
        for i in range(len(predictions)):
            pred = predictions[i].permute(1,2,0).cpu().detach().numpy()
            
            thresh = 0.5 # Post-processing threshold
            plt.figure()
            thresh_pred = pred
            thresh_pred[thresh_pred < thresh] = 0 # post-processing, value less than threshold is set to 0
            thresh_pred[thresh_pred > (thresh)] = 1 # post-processing, value greater than threshold is set to 1
            plt.imshow(thresh_pred, cmap='gray')
            plt.tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False)
            plt.savefig(result_path + "mask_" + img_name[i])

            test_img=data[i].permute(1,2,0).cpu().detach().numpy()
            plt.imshow(test_img, alpha = 0.6)
            plt.tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False)
            plt.savefig(result_path + "result_" + img_name[i])

            plt.close()

if __name__ == "__main__":
    main()