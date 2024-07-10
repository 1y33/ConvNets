import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


class Trainer:
    def __init__(self,model,dataset,loss_fn):
        '''
        model : model you want to train
        dataset : the dataset you have, maybe not init here
        loss_fn : loss function
        '''
        self.model = model
        self.dataset = dataset
        self.loss_fn = loss_fn
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=5e-4)
        self.device = self.model.device


    
                
    
    def complete_train(self,train_ds,val_ds,nr_epochs):


        self.train_loss = []
        self.train_accuracy = []
        self.val_loss = []
        self.val_accuracy = []
        
        for epoch in tqdm(nr_epochs):
            ### Train the model
            self.model.train()
            loss,accuracy = self.train_step(train_ds)
            self.train_loss.append(loss)
            self.train_accuracy.append(accuracy)
            
            ### Val the model
            self.model.eval()
            loss,accuracy = self.val_step(val_ds)
            self.val_loss.append(loss)
            self.val_accuracy.append(accuracy)
            
            self.print_training(epoch,nr_epochs)

        self.plot_graph(nr_epochs)

        

    def loss_step(self,dataset):
        # Calculates the loss for an epoch in the dataset
        loss = 0
        correct = 0
        total = 0

        for input,target in dataset:
            input = input.to(self.device)
            target = target.to(self.device)

            preds = self.model(input)
            loss += self.loss_fn(preds,target)

            _, predicted = torch.max(preds, 1)
            correct = (preds == target).sum().item()
            total+= target.size(0)

        accuracy = correct/total * 100
        loss = loss / len(dataset)

        return loss,accuracy

    def train_step(self, dataset):
        self.optimizer.zero_grad()
        loss, accuracy = self.loss_step(dataset)
        loss.backward()
        self.optimizer.step()

        return loss.item(), accuracy

    def val_step(self, dataset):
        with torch.no_grad():
            loss, accuracy = self.loss_step(dataset)

        return loss.item(), accuracy

    def print_training(self, epoch, nr_epochs):
        print(f"Finished epoch : {epoch + 1}/{nr_epochs}")
        print(f"[Train] Loss :{self.train_loss[epoch]}, Accuracy :{self.train_accuracy[epoch]}")
        print(f"[Val] Loss :{self.val_loss[epoch]}, Accuracy :{self.val_accuracy[epoch]}")


    def plot_graph(self,nr_epochs):
        fig,(ax1,ax2) = plt.subplots(1,2,figsize = (10,5))

        ax1.plot(self.train_loss,nr_epochs,label="Train Loss")
        ax1.plot(self.val_loss,nr_epochs,label = "Val Loss")
        ax1.legend()
        ax1.set_title("Losses")

        ax2.plot(self.train_accuracy,nr_epochs,label = "Train Accuracy")
        ax2.plot(self.val_accuracy,nr_epochs,label = "Val Accuracy")
        ax2.legend()
        ax2.set_title("Accuracies")

        plt.show()