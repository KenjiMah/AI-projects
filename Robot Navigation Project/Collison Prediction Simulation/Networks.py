import torch
import torch.nn as nn
import numpy as np

#import Data_Loaders as Data_Loaders

class Action_Conditioned_FF(nn.Module):
    def __init__(self,input_size = 6, output_size =1):
# STUDENTS: __init__() must initiatize nn.Module and define your network's
# custom architecture
        super(Action_Conditioned_FF,self).__init__()
        self.layer_1 = nn.Linear(input_size, 6) 
        self.layer_out = nn.Linear(6, output_size)
        self.relu = nn.ReLU()

    def forward(self, input):
# STUDENTS: forward() must complete a single forward pass through your network
# and return the output which should be a tensor
        hidden = self.relu(self.layer_1(input))
        output = self.layer_out(hidden)
        return output


    def evaluate(self, model, test_loader, loss_function):
# STUDENTS: evaluate() must return the loss (a value, not a tensor) over your testing dataset. Keep in
# mind that we do not need to keep track of any gradients while evaluating the
# model. loss_function will be a PyTorch loss function which takes as argument the model's
# output and the desired output.

        loss = 0
        for idx, sample in enumerate(test_loader):
            output = self.forward(torch.tensor(sample['input'], dtype = torch.float32))
            loss += loss_function(output, torch.tensor(sample['label'], dtype = torch.float32))
        return np.float(loss/len(test_loader))

def main():
    model = Action_Conditioned_FF()
    print(model.evaluate(model, Data_Loaders.Data_Loaders(16).train_loader, nn.MSELoss()))
if __name__ == '__main__':
    main()
