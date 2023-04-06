
import time
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import gc

import torchvision.transforms as standard_transforms
import util
import numpy as np
import copy

train_loader = DataLoader( )
val_loader = DataLoader( )
test_loader = DataLoader( )











class PCN(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 8, kernel_size=5, stride=1, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=1, dilation=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=1, dilation=1)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=1, dilation=1)
        self.relu = nn.ReLU(inplace=True)
        self.bnd1 = nn.BatchNorm2d(8)
        self.bnd2 = nn.BatchNorm2d(16)
        self.bnd3 = nn.BatchNorm2d(32)
        self.bnd4 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 4)

#TODO Complete the forward pass
    def forward(self, x):
        x1 = self.bnd1(self.relu(self.conv1(x)))
        x2 = self.bnd2(self.relu(self.conv2(x1)))
        x3 = self.bnd3(self.relu(self.conv3(x2)))
        x4 = self.bnd4(self.relu(self.conv4(x3)))
        xf1 = self.fc1(x4)
        xf2 = self.fc1(xf1)

        return xf2









class SimCLR_Loss(nn.Module):
    def __init__(self, batch_size, temperature):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):

        N = 2 * self.batch_size

        z = torch.cat((z_i, z_j), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)
        
        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)
        
        #SIMCLR
        labels = torch.from_numpy(np.array([0]*N)).reshape(-1).to(positive_samples.device).long() #.float()
        
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        
        return loss

def mask(input, m):

    for i in range(len(input)):
        input[i][m]=np.zeros()

    return input











epochs = 100


pcn_model = PCN()
pcn_model.apply(init_weights)

device = torch.device('cpu')  
optimizer = optim.Adam(pcn_model.parameters(), lr=0.001)
criterion1 = nn.CrossEntropyLoss()
criterion2 =  nn.SimCLR_Loss(batch_size = , temperature = 0.5)

pcn_model = pcn_model.to(device)


def train1(save_path):
    smallest_loss = 100000
    patience = 0 
    train_loss = []
    val_loss = []
    early_stopped_epoch = -1
    #scheduler =  optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader))
        
    for epoch in range(epochs):
        curr_train_loss = []
        ts = time.time()
        for iter, (inputs, labels) in enumerate(train_loader):
            

            optimizer.zero_grad()
            

            inputs =  inputs.to(device)
            labels =  labels.to(device) 
            outputs = pcn_model.forward(inputs) 
            loss = criterion1(outputs, labels)  
            curr_train_loss.append(loss.item())

            loss.backward()
            optimizer.step()
            print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.item()))

        #scheduler.step()
        #scheduler =  optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader))
        print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
        
        curr_loss = val1(epoch)
        train_loss.append(np.mean(curr_train_loss))
        val_loss.append(curr_loss)

            
        if curr_loss < smallest_loss:
            smallest_loss = curr_loss
            best_model = copy.deepcopy(fcn_model)
        else:
            patience += 1 

        if patience >= 10:
            early_stopped_epoch = epoch
            print(f'Early Stop at epoch {epoch}')
            torch.save(best_model.state_dict(), save_path)
            break 

    return train_loss, val_loss, early_stopped_epoch, best_model

def val1(epoch):
    pcn_model.eval() 
    
    losses = []

    accuracy = []

    with torch.no_grad(): 

        for iter, (inputs, labels) in enumerate(val_loader):
            inputs =  inputs.to(device)
            labels =  labels.to(device) 
            outputs = pcn_model.forward(inputs)
            losses.append(criterion1(outputs, labels))


    print(f"Loss at epoch: {epoch} is {np.mean(losses)}")

    fcn_model.train()

    return np.mean(losses)



def train2(save_path):
    smallest_loss = 100000
    patience = 0 
    train_loss = []
    val_loss = []
    early_stopped_epoch = -1
    #scheduler =  optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader))
        
    for epoch in range(epochs):
        curr_train_loss = []
        ts = time.time()
        for iter, (inputs, labels) in enumerate(train_loader):
            

            optimizer.zero_grad()
            
            inputs1 = copy.deepcopy(inputs)
            inputs1 =  inputs1.to(device)
            inputs2= copy.deepcopy(inputs)
            inputs2= mask(inputs2, 2)	
            inputs2 =  inputs2.to(device)            
            labels =  labels.to(device) 
            outputs1 = pcn_model.forward(inputs1)
            outputs2 = pcn_model.forward(inputs2) 
            loss = criterion2(outputs1, outputs2)  
            curr_train_loss.append(loss.item())

            loss.backward()
            optimizer.step()
            print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.item()))

        #scheduler.step()
        #scheduler =  optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader))
        print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
        
        curr_loss = val2(epoch)
        train_loss.append(np.mean(curr_train_loss))
        val_loss.append(curr_loss)

            
        if curr_loss < smallest_loss:
            smallest_loss = curr_loss
            best_model = copy.deepcopy(fcn_model)
        else:
            patience += 1 

        if patience >= 10:
            early_stopped_epoch = epoch
            print(f'Early Stop at epoch {epoch}')
            torch.save(best_model.state_dict(), save_path)
            break 

    return train_loss, val_loss, early_stopped_epoch, best_model

def val2(epoch):
    pcn_model.eval() 
    
    losses = []

    accuracy = []

    with torch.no_grad(): 

        for iter, (inputs, labels) in enumerate(val_loader):
            inputs =  inputs.to(device)
            labels =  labels.to(device) 
            outputs = pcn_model.forward(inputs)
            losses.append(criterion2(outputs, labels))


    print(f"Loss at epoch: {epoch} is {np.mean(losses)}")

    fcn_model.train()

    return np.mean(losses)


def modelTest(best_model):
    best_model.eval()

    losses = []

    accuracy = []

    with torch.no_grad():
        for iter, (inputs, labels) in enumerate(test_loader):

            inputs =  inputs.to(device)
            labels =  labels.to(device) 
            outputs = best_model.forward(inputs)
            losses.append(criterion1(outputs, labels))

    print(f"Test loss : is {np.mean(losses)}")


    best_model.train()  









