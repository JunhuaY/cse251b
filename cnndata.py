from random import seed
import torch 
import torch.nn as nn 
from torch.nn import functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import copy
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold


class CNNDataloader(Dataset):
    def __init__(self,value_file, target_file):
        value_df = pd.read_csv(value_file)
        target_df = pd.read_csv(target_file)
        
        x = value_df.iloc[:,3:].values
        patients = value_df.iloc[:,1].values
        visits = value_df.iloc[:,2].values
        y = target_df.iloc[:,4:].values
        ypatients = target_df.iloc[:,2].values
        yvisits = target_df.iloc[:,3].values


        lastpatient = 0
        X = []
        Y = []
        xv = []
        yv = []
        starting = 0
        curr=0
        zeroed =0
        missing=0
        j=0 
        pad = np.zeros(0)
        for i in range(len(patients)):
            if patients[i] != lastpatient:
                xv = []
                yv = []
                starting = visits[i]
                curr=1
                zeroed =0 
                lastpatient = patients[i]

                xv.append(np.array(np.hstack((x[i], pad)) ))
            else:	
                if curr==1:
                    if visits[i]-starting == 6:
                        xv.append(np.array(np.hstack((x[i], pad)) ))
                    else:
                        xv.append(np.array(np.zeros(227)))
                        zeroed+=1

                if curr==2:
                    if visits[i]-starting == 12:
                        xv.append(np.array(np.hstack((x[i], pad)) ))
                    else:
                        xv.append(np.array(np.zeros(227)))
                        zeroed+=1

                if curr==3:
                    if visits[i]-starting == 24:
                        xv.append(np.array(np.hstack((x[i], pad)) ))
                    else:
                        xv.append(np.array(np.zeros(227)))
                        zeroed+=1
                    if zeroed<=2:
                        clinical_info = np.array((target_df[(target_df['patient_id']==lastpatient) & (target_df['visit_month']==0) ])[["updrs_1","updrs_2","updrs_3","updrs_4"]].values)
                        if len(clinical_info) == 0:
                            yv=np.hstack((yv, [0,0,0,0]))
                        else:
                            #print(lastpatient , clinical_info[0])
                            yv=np.hstack((yv, clinical_info[0]))
                        clinical_info = np.array((target_df[(target_df['patient_id']==lastpatient) & (target_df['visit_month']==6) ])[["updrs_1","updrs_2","updrs_3","updrs_4"]].values)
                        if len(clinical_info) == 0:
                            yv=np.hstack((yv, [0,0,0,0]))
                        else:
                            #print(lastpatient , clinical_info[0])
                            yv=np.hstack((yv, clinical_info[0]))
                        clinical_info = np.array((target_df[(target_df['patient_id']==lastpatient) & (target_df['visit_month']==12) ])[["updrs_1","updrs_2","updrs_3","updrs_4"]].values)
                        if len(clinical_info) == 0:
                            yv=np.hstack((yv, [0,0,0,0]))
                        else:
                            #print(lastpatient , clinical_info[0])
                            yv=np.hstack((yv, clinical_info[0]))
                        clinical_info = np.array((target_df[(target_df['patient_id']==lastpatient) & (target_df['visit_month']==24) ])[["updrs_1","updrs_2","updrs_3","updrs_4"]].values)
                        if len(clinical_info) == 0:
                            yv=np.hstack((yv, [0,0,0,0]))
                        else:
                            #print(lastpatient , clinical_info[0])
                            yv=np.hstack((yv, clinical_info[0]))
                        Y.append(np.nan_to_num(np.array(yv)))
                        
                        X.append(copy.deepcopy(np.nan_to_num(xv)))

                if curr>=4:
                    continue

                curr+=1


        self.x_train=X
        self.y_train=Y

    def __len__(self):
        return len(self.x_train)
    def __getitem__(self,idx):
        return self.x_train[idx], self.y_train[idx]




