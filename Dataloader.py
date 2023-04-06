from random import seed
import torch 
import torch.nn as nn 
from torch.nn import functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold


class LSTMDataloader(Dataset):
    def __init__(self,value_df, target_file):
        #value_df = pd.read_csv(value_file)
        target_df = pd.read_csv(target_file)
        
        x = value_df.iloc[:,3:].values
        y = []
        TD  = []
        for i in value_df.index.values:
            pateint = value_df.at[i ,'patient_id']
            visit = value_df.at[i ,'visit_month']
            clinical_info = list((target_df[target_df['patient_id']==pateint])['visit_month'])
            #print("patient: {}, visit: {}, match clinical: {}".format(pateint, visit, clinical_info))
            #print(pateint, visit, target_df[])
            if visit in clinical_info:
                td =torch.tensor(target_df[(target_df['patient_id'] == pateint) & (target_df['visit_month'] >= visit)]['visit_month'].values - visit,
                                 dtype=torch.int32)
                target = torch.tensor(target_df[(target_df['patient_id'] == pateint) & (target_df['visit_month'] >= visit)]
                                      [["updrs_1", "updrs_2","updrs_3","updrs_4"]].values, dtype=torch.float32)
                y.append(target)
                TD.append(td)
        max_size = max([v.shape[0] for i,v in enumerate(TD)])
        for i, v in enumerate(y):
            y[i] = F.pad(input=v, pad=(0, 0,0 ,max_size - v.shape[0]), mode='constant', value=-1)
            TD[i] = F.pad(input=TD[i], pad=(0,max_size - v.shape[0]), mode='constant', value=-1)
        self.x_train=x
        self.y_train=y
        self.td = TD
    def __len__(self):
        return len(self.x_train)
    def __getitem__(self,idx):
        return self.x_train[idx], self.y_train[idx], self.td[idx]

'''
# KFold dataset
kf = KFold(n_splits = 5, shuffle = True, seed=0)

if __name__ == "__main__":
    # Use case for KFold
    value_file = 'new_train_proteins.csv'
    target_file = 'train_clinical_data.csv'
    X_train, y_train = LSTMDataloader(value_file, target_file)
    for i, (train_index, val_index) in enumerate(kf.split(X_train, y_train)):
        print(f"Fold {i+1}")
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
        print(f"  Train: {len(X_train_fold)} samples")
        print(f"  Validation: {len(X_val_fold)} samples")
'''
