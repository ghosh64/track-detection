from tqdm import tqdm
import torch

def train_model(model, data_loader, loss, optimizer, epochs):
    """
    Train the model

    Parameters:
        model: Model to be trained
        data_loader: Training data
        loss: Loss function 
        optimizer: Optimization algorithm
        epochs: Number of epochs to train

    Returns:
        loss_list (list): List containing the loss value for each epoch

    """

    device = torch.device('cuda:0')
    model.to(device)
    model.train()
    loss_list = []
    b_list = []
    for epoch in range(epochs):
        print("Epoch: ", (epoch + 1))
        model.train()
        for data, label, _ in tqdm(data_loader):
            data = data.to(device)
            label = label.to(device)
            output, _, _ = model(data) # Global and local results not needed for calculating loss 
            batch_loss = loss(output, label)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            b_list.append(batch_loss.cpu().detach().numpy())
        loss_list.append(sum(b_list)/len(b_list))
        print("Loss: ", sum(b_list)/len(b_list))
    return loss_list