from tqdm import tqdm
import torch

def test(model, test_data_loader, loss, return_pred=False):
    model.eval()
    overall_loss=0
    with torch.no_grad():
        for data, labels, _ in test_data_loader:
            device=torch.device('cuda:0')
            data=data.to(device, dtype=torch.float32)
            labels = labels.to(device,dtype=torch.float32)
            prediction = model(data)
            b_loss=loss(prediction, labels)
            overall_loss+=b_loss.cpu().detach().numpy()
    #average loss per batch
    return overall_loss/len(test_data_loader) if not return_pred else (overall_loss/len(test_data_loader))

def train_model(model, data_loader, loss, optimizer, epochs,val_data_loader=None):
    device=torch.device('cuda:0') # Use GPU to run
    model.to(device)
    model.train()
    loss_list=[]
    b_list=[]
    for epoch in range(epochs):
        print("Epoch: ", (epoch + 1))
        for data, label, _ in tqdm(data_loader):
            model.train()
            data = data.to(device, dtype=torch.float32)
            label = label.to(device, dtype=torch.float32)
            output=model(data)
            # print(output.shape)
            batch_loss=loss(output, label)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            b_list.append(batch_loss.cpu().detach().numpy())
        loss_list.append(sum(b_list)/len(b_list))
        if val_data_loader:
            val_loss=test(model, val_data_loader, loss)
        print("Loss: ", sum(b_list)/len(b_list)) # Prints value of loss
        if val_data_loader:
            print("Val Loss: ", val_loss)
    #average loss per batch
    return loss_list