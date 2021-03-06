from math import floor
import torch
import random
import numpy as np
import torch.nn as nn
from torch.nn.init import xavier_uniform_
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import average_precision_score
import time

class ConvAttModel(nn.Module):
    
    def __init__(self, vocab_size=1000, kernel_size=10, num_filter_maps=16, embed_size=100, dropout=0.5, embedding=None, seed=19820618):
        super(ConvAttModel, self).__init__()
        set_seed(seed)
        self.embed_size = embed_size
        self.embed_drop = nn.Dropout(p=dropout)
        
        # embedding
        if embedding is None:
            self.embed = nn.Embedding(vocab_size+1, embed_size)
            xavier_uniform_(self.embed.weight)
        else:
            embedding_tensor = torch.FloatTensor(embedding)
            self.embed = nn.Embedding.from_pretrained(embedding_tensor)
        
        # conv layer
        self.conv = nn.Conv1d(self.embed_size, num_filter_maps, kernel_size, padding=int(floor(kernel_size/2)))
        xavier_uniform_(self.conv.weight)
        
        # context for attention
        self.u = nn.Linear(num_filter_maps, 1)
        xavier_uniform_(self.u.weight)
        
        # final layer
        self.final = nn.Linear(num_filter_maps, 1)
        xavier_uniform_(self.final.weight)
    
    def forward(self, text):
        text = text.transpose(1,-1)
        embedded = self.embed(text)
        embedded = self.embed_drop(embedded)
        embedded = embedded.transpose(1,-1)
        conved = torch.tanh(self.conv(embedded))
        attention = F.softmax(torch.matmul(self.u.weight, conved), dim=2)
        v = torch.matmul(attention, conved.transpose(1,-1))
        y = torch.sigmoid(self.final(v)).squeeze()
        return y, attention
        
def predict(model, data, max_batches=0):
    model.eval()
    y_pred = torch.LongTensor()
    y_true = torch.LongTensor()
    for i, (x, y) in enumerate(data):
        if max_batches > 0 and i > max_batches:
            break
        y_hat, attention = model(x)
        y_pred = torch.cat((y_pred, y_hat.detach().to('cpu')), dim=0)
        y_true = torch.cat((y_true, y.detach().to('cpu')), dim=0)
    return y_true, y_pred


def eval(model, data, max_batches=0):
    y_true, y_pred = predict(model, data, max_batches)
    return average_precision_score(y_true, y_pred)


def train(model, train_data, test_data, n_epochs, lr=0.001, weight_decay=0, seed=19820618):
    set_seed(seed)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCELoss()
    for epoch in range(n_epochs):
        start_time = time.time()
        model.train()
        train_loss = 0
        for i, (x, y) in enumerate(train_data):
            print(f"{i}/{len(train_data)}", end="\r")
            optimizer.zero_grad()
            y_hat, attention = model(x)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss = train_loss/len(train_data)
        print("Calculating train metrics", end="\r")
        train_aps = eval(model, train_data, len(test_data))
        print("Calculating test metrics ", end="\r")
        test_aps = eval(model, test_data, len(test_data))
        end_time = time.time()
        print(f"Epoch {epoch}\tTraining Loss: {train_loss:.6f}" + 
              f"\tTrain APS: {train_aps:.6f}\tTest APS: {test_aps:.6f}" +
              f"\tTime Taken: {end_time-start_time:.2f} seconds")

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)