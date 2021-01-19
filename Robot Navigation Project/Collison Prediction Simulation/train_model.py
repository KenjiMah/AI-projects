from Data_Loaders import Data_Loaders
from Networks import Action_Conditioned_FF

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torchsummary import summary
def train_model(no_epochs):

    batch_size = 16
    data_loaders = Data_Loaders(batch_size)
    model = Action_Conditioned_FF()
    PATH = 'C:/Users/Kenji Mah/Desktop/assignment_part4/saved/saved_model.pkl'
    loss_function = nn.MSELoss()
    losses = []

    min_loss = model.evaluate(model, data_loaders.test_loader, loss_function)
    losses.append(min_loss)
    
    summary(model)
    learning_rate = 0.01
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    for epoch_i in range(no_epochs):
        for idx, sample in enumerate(data_loaders.train_loader):
            # output = model.forward(sample['input'])
            optimizer.zero_grad()
            output = model(torch.tensor(sample['input'], dtype = torch.float32))
            loss = loss_function(output, torch.tensor(sample['label'], dtype = torch.float32))
            loss.backward()
            optimizer.step()
        losses.append(model.evaluate(model, data_loaders.train_loader,loss_function))
    #torch.save(model.state_dict(), PATH,_use_new_zipfile_serialization=False)
    
    test = model.evaluate(model, data_loaders.test_loader, loss_function)
    plt.plot(range(0,no_epochs+1),losses)
    plt.title('Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.xticks(range(0,no_epochs+1))
    print(test)
if __name__ == '__main__':
    no_epochs = 5
    train_model(no_epochs)
