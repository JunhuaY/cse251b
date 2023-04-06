# Build and return the model here based on the configuration.
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from torch.distributions.categorical import Categorical
import numpy as np


class CNN(nn.Module):        
    def __init__(self):
        super().__init__()

        self.Conv=nn.ModuleList( [nn.Conv2d(1,519,(n, 227)) for n in (2, 3, 4)] )

        self.fc=nn.Linear(519*3, 16)

    def relu_pool(self, In, conv):
        In=F.relu(conv(In)).squeeze(3)
        In=F.max_pool1d(In, In.size(2)).squeeze(2)
        return In

    def forward(self, out):
        out=out.unsqueeze(0)
        out=out.unsqueeze(1)
        out=torch.cat([self.relu_pool(out,conv) for conv in self.Conv],1)
        out=self.fc(out)
        return out



def get_model():

    model = CNN()
    return model

