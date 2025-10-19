# -*- coding: utf-8 -*-
# @Author: zzt
# @Date:   2022-10-20 09:05:04
# @Last Modified by:   zzt
# @Last Modified time: 2022-11-29 21:18:50
import imp
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from models.model import MultiModalModel
from torch_geometric.loader import DataLoader 
from models.trainer import Trainer


class Training():
    def __init__(self,train_loader,val_loader,params) :
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.params = params
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    def bulid_trainner(self):
        lstm_input_dim = self.params["lstm_input_dim"]
        lstm_hidden_dim = self.params["lstm_hidden_dim"]
        lstm_layer_dim = self.params["lstm_layer_dim"]
        output_dim = self.params["output_dim"]

        dropout = self.params["dropout"]
        lr = self.params["lr"]
        momentum = self.params.get("momentum", 0.9)
        weight_decay = self.params["weight_decay"]
        epochs = self.params["n_epochs"]
 
        model = MultiModalModel(lstm_input_dim, lstm_hidden_dim, lstm_layer_dim, 
                                output_dim, dropout)
        model.to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        trainer = Trainer(model=model, optimizer=optimizer,train_loader= self.train_loader, valid_loader= self.val_loader)
        return trainer,epochs,model
    
    def train(self):
        trainer,epoch,model = self.bulid_trainner()
        (train_losses, train_scores), (valid_losses, valid_scores),best = trainer.run(epoch)
        return trainer,model,(train_losses, train_scores), (valid_losses, valid_scores),best

