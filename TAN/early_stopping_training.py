import sys
import csv
import copy
import numpy as np
import re
import itertools
from collections import Counter
from utils import *
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
from networks import *
import pickle
from datetime import datetime
import random
from statistics import mode
import copy
import os
import pandas as pd
import matplotlib.pyplot as plt

D = None

random.seed(42)

if len(sys.argv) !=3:
    print("Usage :- python early_stopping_training.py <dataset name> <attention vairant>")
    sys.exit(-1)

version = sys.argv[2]
dataset = sys.argv[1]

def f_score(table):
    return "%.2f" % (100*table[0][0]/(table[0][1]+table[0][2]) + 100*table[1][0]/(table[1][1]+table[1][2]))


def train_bagging_tan_CV(version="tan-",n_epochs=50,batch_size=50,l2=0,dropout = 0.5,n_folds=5):

    NUM_EPOCHS = n_epochs
    loss_fn = nn.NLLLoss()
    n_models = 1
    print("\n\n starting cross validation \n\n")
    print("class : ",dataset, " :-")

    score = 0

    fold_sz = len(x_train)//n_folds
    foldwise_val_scores = []
    ensemble_models = []
    print("dataset size :- ",len(x_train))
    for fold_no in range(n_folds):
        print("Fold number {}".format(fold_no+1))
        best_val_score = 0
        model = LSTM_TAN(version,300,100,len(embedding_matrix),3,embedding_matrix,dropout=dropout).to(device)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad,model.parameters()),lr=0.0005,weight_decay = l2)
        if fold_no == n_folds-1:
            ul = len(x_train)
        else:
            ul = (fold_no+1)*fold_sz
        print("ll : {}, ul : {}".format(fold_no*fold_sz,ul))
        best_ensemble = []
        best_score = 0
        temp_ensemble = []
        temp_score = 0
        for _ in range(NUM_EPOCHS):
            ep_loss = 0
            target = torch.tensor(vector_target,dtype=torch.long).to(device)
            optimizer.zero_grad()

            #training

            model.train()
            loss = 0




            for i in range(fold_no*fold_sz):
                model.hidden = model.init_hidden()

                x = torch.tensor(np.array(x_train[i]),dtype=torch.long).to(device)
                y = torch.tensor([y_train[i]],dtype=torch.long).to(device)

                preds = model(x,target,verbose=False)

                x_ = loss_fn(preds,y)
                loss += x_
                ep_loss += x_
                if (i+1) % batch_size == 0:
                    loss.backward()
                    loss = 0
                    optimizer.step()
                    optimizer.zero_grad()

            for i in range(ul,len(x_train)):
                model.hidden = model.init_hidden()

                x = torch.tensor(np.array(x_train[i]),dtype=torch.long).to(device)
                y = torch.tensor([y_train[i]],dtype=torch.long).to(device)

                preds = model(x,target,verbose=False)

                x_ = loss_fn(preds,y)
                loss += x_
                ep_loss += x_
                if (i+1) % batch_size == 0:
                    loss.backward()
                    loss = 0
                    optimizer.step()
                    optimizer.zero_grad()

            optimizer.step()
            optimizer.zero_grad()

            #validation
            corr = 0
            with torch.no_grad():
                conf_matrix = np.zeros((2,3))
                for j in range(fold_sz*fold_no,ul):
                    x = torch.tensor(np.array(x_train[j]),dtype=torch.long).to(device)
                    y = torch.tensor([y_train[j]],dtype=torch.long).to(device)
                    model.eval()
                    preds = model(x,target,verbose=False)
                    label = np.argmax(preds.cpu().numpy(),axis=1)[0]
                    if label == y_train[j]:
                        corr+=1
                        if label <=1:
                            conf_matrix[label][0]+=1
                    if y_train[j] <=1:
                        conf_matrix[y_train[j]][2]+=1
                    if label <=1:
                        conf_matrix[label][1]+=1
                    ep_loss+=loss_fn(preds,y)
            val_f_score = float(f_score(conf_matrix))

            #if val_f_score > best_val_score:
            #    best_val_score = val_f_score
            #    best_model = copy.deepcopy(model)


            if _%10 ==0 and _ != 0:
                print("current last 10- score ",temp_score*1.0/10)
                if temp_score > best_score:
                    best_score = temp_score
                    best_ensemble = temp_ensemble
                    print("this is current best score now")
                temp_ensemble = []
                temp_score = 0

            temp_ensemble.append(copy.deepcopy(model))
            temp_score += val_f_score



            print("epoch number {} , val_f_score {}".\
            format(_+1,f_score(conf_matrix)))


        print("current last 10- score ",temp_score*1.0/10)
        if temp_score > best_score:
            best_score = temp_score
            best_ensemble = temp_ensemble
            print("this is current best score now")

        ensemble_models.extend(best_ensemble)

    with torch.no_grad():
        conf_matrix = np.zeros((2,3))
        for j in range(len(x_test)):
            x = torch.tensor(np.array(x_test[j]),dtype=torch.long).to(device)
            y = torch.tensor([y_test[j]],dtype=torch.long).to(device)
            all_preds = []
            for model in ensemble_models:
                model.eval()
                all_preds.append(np.argmax(model(x,target).cpu().numpy(),axis=1)[0])
            cnts = [0,0,0]
            for prediction in all_preds:
                cnts[prediction]+=1
            label = np.argmax(cnts)
            if label == y_test[j]:
                corr+=1
                if label <=1:
                    conf_matrix[label][0]+=1
            if y_test[j] <=1:
                conf_matrix[y_test[j]][2]+=1
            if label <=1:
                conf_matrix[label][1]+=1
            ep_loss+=loss_fn(preds,y)

    print("test_f_score {}".format(f_score(conf_matrix)))
    print(conf_matrix)
    return conf_matrix



dataset = sys.argv[1]

fin_matrix = np.zeros((2,3))


stances, word2emb, word_ind, ind_word, embedding_matrix, device,\
    x_train, y_train, x_test, y_test, vector_target, train_tweets, test_tweets  = load_dataset(dataset,dev = "cuda")


combined = list(zip(x_train, y_train))
random.shuffle(combined)
x_train[:], y_train[:] = zip(*combined)


fin_matrix += train_bagging_tan_CV(version=version,n_epochs=100,batch_size=50,l2=0.0,dropout = 0.6,n_folds=10)

print(f_score(fin_matrix))
