# -*- coding: utf-8 -*-
# @Author: zzt
# @Date:   2022-08-17 11:52:03
# @Last Modified by:   zzt
# @Last Modified time: 2022-11-29 23:15:14

import imp
import math
import copy
import hiddenlayer as hl
import numpy as np
from tqdm import tqdm
from sklearn.metrics import r2_score , accuracy_score,balanced_accuracy_score
# pytorch
import torch

# Custom Trainer class
# Train and make prediction with the GNN models
class Trainer:
    def __init__(self, model, optimizer, train_loader, valid_loader):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # training model
    def train_one_epoch(self, epoch):
        # set model on training mode
        self.model.train()

        t_targets = []; p_targets = []; losses = []
        tqdm_iter = tqdm(self.train_loader, total=len(self.train_loader))
        for i, data in enumerate(tqdm_iter):
            x1,x2,x3,x4,y = data[0],data[1],data[2],data[3],data[4]
            x1,x2,x3,x4 = x1.to(self.device),x2.to(self.device),x3.to(self.device),x4.to(self.device)
            y = y.to(self.device)
            tqdm_iter.set_description(f"Epoch {epoch}")
            self.optimizer.zero_grad()
            
            outputs = self.model(x1,x2,x3,x4)
            targets = y
            
            loss = torch.nn.CrossEntropyLoss()
            loss = loss(outputs, targets.long())
            
            loss.backward()
            self.optimizer.step()

            y_true = self.process_output(targets)  # for one batch
            _,outputs = torch.max(outputs,dim=1)
            y_proba = self.process_output(outputs) # for one batch
            accuracy = accuracy_score(y_true, y_proba)
            t_targets.extend(list(y_true))
            p_targets.extend(list(y_proba))        
        
            tqdm_iter.set_postfix(train_loss=round(loss.item(), 2), train_accuracy=round(accuracy, 2))
            # continuous loss/auc update            
            losses.append(loss.item())
        #------------
        #torch.save(self.model.state_dict(),"model_saved\lstm_params.pkl")
        #---------------------
        epoch_ba = balanced_accuracy_score(t_targets, p_targets)

        epoch_a = accuracy_score(t_targets, p_targets)

        epoch_loss = sum(losses)/len(losses)
           

        return epoch_loss, epoch_ba,epoch_a, tqdm_iter


    def process_output(self, out):
        out = out.cpu().detach().numpy()
        return out

    
    def validate_one_epoch(self, progress):

        progress_tracker = progress["tracker"]
        train_loss = progress["loss"]
        train_accuracy = progress["ba"]
        
        # model in eval model
        self.model.eval()
        
        t_targets = []; p_targets = []; losses = []
        tqdm_iter = tqdm(self.valid_loader, total=len(self.valid_loader))
        for i, data in enumerate(tqdm_iter):
            x1,x2,x3,x4,y = data[0],data[1],data[2],data[3],data[4]
            x1,x2,x3,x4 = x1.to(self.device),x2.to(self.device),x3.to(self.device),x4.to(self.device)
            y = y.to(self.device)
            self.optimizer.zero_grad()
            
            outputs = self.model(x1,x2,x3,x4)
            targets = y
            
            loss = torch.nn.CrossEntropyLoss()
            loss = loss(outputs, targets.long())
            y_true = self.process_output(targets)  # for one batch
            _,outputs = torch.max(outputs,dim=1)
            y_proba = self.process_output(outputs) # for one batch
            
            t_targets.extend(list(y_true))
            p_targets.extend(list(y_proba))
            losses.append(loss.item())

            valid_accuracy = accuracy_score(y_true,y_proba)

            tqdm_iter.set_postfix(valid_loss=round(loss.item(), 2), valid_accuracy=round(valid_accuracy, 2))

            
        #print(y_proba)
        
        epoch_ba = balanced_accuracy_score(t_targets, p_targets)

        epoch_a = accuracy_score(t_targets, p_targets)

        epoch_loss = sum(losses)/len(losses)
                      
        progress_tracker.close()
        return epoch_loss, epoch_ba, epoch_a
            
    # runs the training and validation trainer for n_epochs
    def run(self, n_epochs=10):
        
        train_scores = []; train_losses = []
        valid_scores = []; valid_losses = []
        #-------------------------------
        history1 = hl.History()
        canvas1 = hl.Canvas()
        best_loss = math.inf
        best_score = float(0)
        best_a = float(0)
        best_params = None
        #-------------------------------
        for e in range(1, n_epochs+1):
            tl, tba,ta, progress_tracker = self.train_one_epoch(e)
            
            train_losses.append(tl)
            train_scores.append(tba)

            #---------------
            
            # validate this epoch
            progress = {"tracker": progress_tracker, "loss": tl, "ba": tba}  
            vl, vba,va = self.validate_one_epoch(progress)  # pass training progress tracker to validation func            
            valid_losses.append(vl)
            valid_scores.append(vba)
            if (vl<best_loss) & (vba>best_score) & (va>best_a):
                best_loss = vl
                best_score = vba
                best_a = va
                best_params = copy.deepcopy(self.model.state_dict())
            #self.model.load_state_dict(best_params)
            #--------------------------------
            history1.log(e,train_loss = tl,ba = tba,val_loss = vl,val_ba = vba)
            with canvas1:
                canvas1.draw_plot(history1["train_loss"])
                canvas1.draw_plot(history1["ba"])
                canvas1.draw_plot(history1["val_loss"])
                canvas1.draw_plot(history1["val_ba"])

        return (train_losses, train_scores), (valid_losses, valid_scores),best_params
            
        
    def predict(self, test_loader):
        # set model on evaluation mode
        self.model.eval()
        predictions = []
        target = []
        embedding = []
        tqdm_iter = tqdm(test_loader, total=len(test_loader))
        for i, data in enumerate(tqdm_iter):
            x1,x2,x3,x4,y = data[0],data[1],data[2],data[3],data[4]
            x1,x2,x3,x4 = x1.to(self.device),x2.to(self.device),x3.to(self.device),x4.to(self.device)
            y = y.to(self.device)

            tqdm_iter.set_description(f"Making prediction")
            with torch.no_grad():
                o = self.model(x1,x2,x3,x4)
                o = self.process_output(o)
            
                predictions.extend(o.tolist())
                target.extend(y.tolist())
                
            tqdm_iter.set_postfix(stage="test dataloader")
        tqdm_iter.close()
        return  embedding ,target , np.array(predictions)