import matplotlib.pyplot as plt
import numpy as np
import torch
from datetime import datetime
import time
import torch.nn as nn
import os
import copy
from model_factory import get_model
import torchvision
import pandas as pd
from Dataloader import LSTMDataloader
from cnndata import CNNDataloader


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def load_data():
    value_file = 'new_train_proteins_filled.csv'
    target_file = 'train_clinical_data_reformat.csv'
    trainloader = CNNDataloader(value_file, target_file)
    train_loader=[]
    val_loader=[]
    for i, (prot, scores) in enumerate(trainloader):
        if i <= 59:
            train_loader.append([prot, scores])
        if i > 59:
            val_loader.append([prot, scores])

    #print(type(train_loader), len(train_loader), type(val_loader),  len(val_loader))
    return train_loader, val_loader

        
def z_score(p):
    return (p-np.mean(p))/np.std(p)

def smape_loss(y_true, y_pred):
        y_true = y_true + 1
        y_pred = y_pred + 1
        epsilon = 0.1
        numer = (y_pred - y_true).abs()
        denom = torch.maximum((y_true).abs() + (y_pred).abs() + epsilon, torch.ones_like (y_pred)* 0.5 + epsilon)
        smape = numer / (denom/2)
        smape = torch.where(torch.isnan(smape), torch.zeros_like(smape), smape)
        return torch.mean(smape) * 100 

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
        input[i][m]=torch.FloatTensor(np.zeros(227))

    return input


class Experiment(object):
    def __init__(self):
        
        self.__train_loader, self.__val_loader = load_data()

        self.__epochs = 200
        self.lr = 0.005
        self.__current_epoch = 0
        self.__patience = 20
        self.__early_stop = True
        self.__lowest_val_loss = 10000
        self.__training_losses = []
        self.__training_losses2 = []
        self.__training_losses3 = []
        self.__val_losses = []
        self.__val_losses2 = []
        self.__val_losses3 = []
        self.SimCLR =   SimCLR_Loss(batch_size = 2, temperature = 0.5)
        self.__model = get_model()
        self.__model.to(device)
        self.__criterion2 = torch.nn.MSELoss()
        self.__criterion = torch.nn.L1Loss(reduction='mean')

        self.__optimizer = torch.optim.Adam(self.__model.parameters(), lr=self.lr)
        self.__scheduler = torch.optim.lr_scheduler.ExponentialLR(self.__optimizer, gamma=0.9)
        

        self.__init_model()

    def __init_model(self):
        if torch.cuda.is_available():
            self.__model = self.__model.cuda().float()
            self.__criterion = self.__criterion.cuda()

    def run(self):
        start_epoch = self.__current_epoch
        count = 0 # for early stop
        for epoch in range(start_epoch, self.__epochs):  # loop over the dataset multiple times
            start_time = datetime.now()
            self.__current_epoch = epoch
            train_loss, train_loss2 ,train_loss3= self.__train()
            val_loss, val_loss2, val_loss3 = self.__val()
            self.__training_losses.append(train_loss)
            self.__training_losses2.append(train_loss2)
            self.__training_losses3.append(train_loss3)
            self.__val_losses.append(val_loss)
            self.__val_losses2.append(val_loss2)
            self.__val_losses3.append(val_loss3)
            print("Epoch: {}, Training loss: {:.6f},Training loss2: {:.6f}, Validation loss: {:.6f} ,Validation loss2: {:.6f}, Validation loss3: {:.6f}".format(epoch, train_loss, train_loss2, val_loss, val_loss2, val_loss3))

            self.__scheduler.step()
            
            if val_loss >= self.__lowest_val_loss:
                #print("curr val loss: {}, lowest: {}".format(val_loss, self.__lowest_val_loss))
                count+=1
            else:
                count=0
                self.__lowest_val_loss = val_loss
            
            if self.__early_stop and count == self.__patience:
                torch.save(self.__model.state_dict(), './model_weight/cnnweight')
                print("Early stopped at: {}".format(epoch))
                break
                
            
        self.plot_stats()


    def runs(self):
        start_epoch = self.__current_epoch
        count = 0 # for early stop
        for epoch in range(start_epoch, self.__epochs):  # loop over the dataset multiple times
            start_time = datetime.now()
            self.__current_epoch = epoch
            train_loss, train_loss2 ,training_loss3= self.__trains()
            val_loss, val_loss2, val_loss3 = self.__val()
            self.__training_losses.append(train_loss)
            self.__training_losses2.append(train_loss2)
            self.__training_losses3.append(train_loss3)
            self.__val_losses.append(val_loss)
            self.__val_losses2.append(val_loss2)
            self.__val_losses3.append(val_loss3)
            print("Epoch: {}, Training loss: {:.6f},Training loss2: {:.6f}, Validation loss: {:.6f} ,Validation loss2: {:.6f}, Validation loss3: {:.6f}".format(epoch, train_loss, train_loss2, val_loss, val_loss2, val_loss3))
            self.__scheduler.step()
            
            if val_loss >= self.__lowest_val_loss:
                #print("curr val loss: {}, lowest: {}".format(val_loss, self.__lowest_val_loss))
                count+=1
            else:
                count=0
                self.__lowest_val_loss = val_loss
            
            if self.__early_stop and count == self.__patience:
                print("Early stopped at: {}".format(epoch))
                break
                
            
        self.plot_stats()
    
    def __train(self):
        self.__model.train()
        training_loss = 0
        training_loss2 = 0
        training_loss3 = 0
        start_time = time.time()

        for i, (prot, scores) in enumerate(self.__train_loader):
            prot = z_score(prot)
            prot = torch.tensor(prot, dtype=torch.float32).to(device)
            scores=torch.tensor(scores, dtype=torch.float32).to(device)
            scores=scores.unsqueeze(0)

            self.__optimizer.zero_grad()
            
            out = self.__model(prot)
            loss = self.__criterion(out, scores)
            loss2 = self.__criterion2(out, scores)
            loss.backward()

            training_loss += loss.item()
            training_loss2 += loss2.item()
            training_loss3 += smape_loss(out, scores)
            self.__optimizer.step()

        training_loss = training_loss / len(self.__train_loader)
        training_loss2 = training_loss2 / len(self.__train_loader)
        training_loss3 = training_loss3 / len(self.__train_loader)
        return training_loss, training_loss2, training_loss3.item() 

    def __trains(self):
        self.__model.train()
        training_loss = 0
        training_loss2 = 0
        start_time = time.time()
        lastprot=[]
        lastscore=[]
        for i, (prot, scores) in enumerate(self.__train_loader):
            if i!=0:
                prot = z_score(prot)
                prot = torch.tensor(prot, dtype=torch.float32).to(device)
                inputs1 = copy.deepcopy(torch.Tensor([prot,lastprot]))

                inputs2= copy.deepcopy(torch.Tensor([prot,lastprot]))
                inputs2= mask(inputs2, 2)	


                self.__optimizer.zero_grad()
            
                out1 = self.__model(inputs1)
                out2 = self.__model(inputs2)
                loss = self.SimCLR(out1,out2)
                loss2 = self.__criterion2(out, scores)
                loss.backward()

                training_loss += loss.item()
                training_loss2 += loss2.item()
                self.__optimizer.step()

            lastscore=scores
            lastprot=prot
        training_loss = training_loss / len(self.__train_loader)
        training_loss2 = training_loss2 / len(self.__train_loader)
        return training_loss, training_loss2
            
    def __val(self):
        self.__model.eval()
        val_loss = 0
        val_loss2 = 0
        val_loss3 = 0
        iter_loss = 0
        iter_loss2 = 0
        start_time = time.time()
        with torch.no_grad():
            for i, (prot, scores) in enumerate(self.__val_loader):
                prot = z_score(prot)
                prot = torch.tensor(prot, dtype=torch.float32).to(device)
                scores=torch.tensor(scores, dtype=torch.float32).to(device)
                scores=scores.unsqueeze(0)
                
                out = self.__model(prot)
                loss = self.__criterion(out, scores)
                loss2 = self.__criterion2(out, scores)
                iter_loss += loss.item()
                iter_loss2 += loss2.item()

                val_loss += loss.item()
                val_loss2 += loss2.item()
                val_loss3 += smape_loss(out, scores)
            val_loss = val_loss / len(self.__val_loader)
            val_loss2 = val_loss2 / len(self.__val_loader)
            val_loss3 = val_loss3 / len(self.__val_loader)
        return val_loss, val_loss2, val_loss3.item() 
        
    def plot_stats(self):
        e = len(self.__training_losses)
        x_axis = np.arange(1, e + 1, 1)
        plt.figure()
        plt.plot(x_axis, np.array(self.__training_losses3)/100, label="Training Loss")
        plt.plot(x_axis, np.array(self.__val_losses3)/100, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.legend(loc='best')
        plt.title("SMAPE loss")
        plt.savefig("stat_plot.png")
        plt.show()

        
        
    
