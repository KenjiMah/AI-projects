import torch
import torch.utils.data as data
import torch.utils.data.dataset as dataset
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class Nav_Dataset(dataset.Dataset):
    def __init__(self):
        self.data = np.genfromtxt('saved/training_data.csv', delimiter=',', dtype=np.float32)
# STUDENTS: it may be helpful for the final part to balance the distribution of your collected data
# Indicies of each class' observations
        
        class0 = self.data[np.where(self.data[:,-1] == 0)]
        class1 = self.data[np.where(self.data[:,-1] == 1)]
        downsampled = class0[np.random.choice(len(class0), size=len(class1), replace=False)]
        self.data = np.concatenate((class1, downsampled), axis=0)
        np.random.shuffle(self.data)

 #normalize data and save scaler for inference
        self.scaler = MinMaxScaler()
        self.normalized_data = self.scaler.fit_transform(self.data) #fits and transforms
        pickle.dump(self.scaler, open("saved/scaler.pkl", "wb")) #save to normalize at inference

    def __len__(self):
# STUDENTS: __len__() returns the length of the dataset
        return(len(self.normalized_data))
        

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            idx = idx.item()
# STUDENTS: for this example, __getitem__() must return a dict with entries {'input': x, 'label': y}
# x and y should both be of type float32. There are many other ways to do this, but to work with autograding
# please do not deviate from these specifications.
        #return({'input': torch.tensor(self.normalized_data[idx, :-1], requires_grad=True).float(), 'label': torch.tensor(self.normalized_data[idx, -1], requires_grad=True).float()})
        return({'input': self.normalized_data[idx, :-1], 'label': self.normalized_data[idx, -1]})
    
class Data_Loaders():
    def __init__(self, batch_size):
        self.nav_dataset = Nav_Dataset()
# STUDENTS: randomly split dataset into two data.DataLoaders, self.train_loader and self.test_loader
# make sure your split can handle an arbitrary number of samples in the dataset as this may vary
        result = []
        for i in range(len(self.nav_dataset)):
            result.append(self.nav_dataset.__getitem__(i))
        lengths = [round(len(result)*0.8), round(len(result)*0.2)]
        self.train_loader, self.test_loader = torch.utils.data.random_split(result, lengths)
        
def main():
    batch_size = 16
    data_loaders = Data_Loaders(batch_size)
    # STUDENTS : note this is how the dataloaders will be iterated over, and cannot be deviated from
    for idx, sample in enumerate(data_loaders.train_loader):
        _, _ = sample['input'], sample['label']
        print(sample)
        # print(sample['input'], sample['label'])
        # print(len(sample['input']))
        # print(sample['input'])
        # print(sample['label'])
        return
    for idx, sample in enumerate(data_loaders.test_loader):
        _, _ = sample['input'], sample['label']

if __name__ == '__main__':
    main()
