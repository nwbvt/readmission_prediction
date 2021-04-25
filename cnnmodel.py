from math import floor
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import average_precision_score
import time
#from cnndata import *
#import cnndata
import data
from torch.utils.data import DataLoader

class CNNModel(nn.Module):
    
    def __init__(self, vocab_size=5000, kernel_size=10, num_filter_maps=16, embed_size=100, dropout=0.5):
        super(CNNModel, self).__init__()
        print("initializing CNNMOdel")
        self.embed_size = embed_size
        self.embed_drop = nn.Dropout(p=dropout)
        
        # embedding
        self.embed = nn.Embedding(vocab_size+1, embed_size)
        xavier_uniform_(self.embed.weight)
        
        # conv layer
        self.conv = nn.Conv1d(self.embed_size, num_filter_maps, kernel_size, padding=int(floor(kernel_size/2)))
        xavier_uniform_(self.conv.weight)
        
        # context for attention
        #self.u = nn.Linear(num_filter_maps, 1)
        #xavier_uniform_(self.u.weight)

        #self.pool = nn.MaxPool1d(kernel_size, stride=kernel_size)
        #todo xavier_uniform_ here too?

        # final layer
        self.final = nn.Linear(num_filter_maps, 1)
        xavier_uniform_(self.final.weight)
    
    def forward(self, text):
        text = text.transpose(1,-1).to(device)
        embedded = self.embed(text).to(device)
        embedded = self.embed_drop(embedded).to(device)
        embedded = embedded.transpose(1,-1).to(device)
        conved = torch.tanh(self.conv(embedded)).to(device)
        #attention = F.softmax(torch.matmul(self.u.weight, conved), dim=2) 
        #v = torch.matmul(attention, conved.transpose(1,-1))
        #y = torch.sigmoid(self.final(v)).squeeze() #this might need a transpose before it if dimensions dont work out.
        #print("conved shape: ", conved.shape)
        #conved = self.pool(conved)
        #print("maxpooled conved shape: ", conved.shape)
        #print("conved[0][0]: ", conved[0][0])
        val, indx = torch.max(conved,dim=2,keepdim=True)
        #print("max returned val: ", val)
        #print("max returned val shape: ", val.shape)
        conved = val
        y = torch.sigmoid(self.final(conved.transpose(1,-1))).squeeze()
        #print ("y transposed and squeezed shape: ", y.shape)
        #return y, attention # dont need to return attention
        return y #just returning y since we don't have attention anymore in CNN model
        
        
def eval(model, data):
    print("calling eval")
    model.eval()
    y_pred = torch.LongTensor().to(device)
    y_true = torch.LongTensor().to(device)
    for x, y in data:
        #y_hat, attention = model(x)
        x = x.to(device)
        y = y.to(device)
        y_hat = model(x)
        y_pred = torch.cat((y_pred, y_hat.detach().to(device)), dim=0)
        y_true = torch.cat((y_true, y.detach().to(device)), dim=0)
    return average_precision_score(y_true.cpu(), y_pred.cpu())


def train(model, train_data, test_data, n_epochs):
    print("starting train")
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.BCELoss()
    for epoch in range(n_epochs):
        print("starting epoch ", epoch)
        start_time = time.time()
        model.train()
        train_loss = 0
        for x, y in train_data:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            #y_hat, attention = model(x) # need to not get attention from return of forward
            y_hat = model(x) # adding this new invocation for plain CNN instead of the above which returned attention
            #print("calling criterion(y_hat, y), with shapes:", y_hat.shape, y.shape)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss = train_loss/len(train_data)
        train_aps = eval(model, train_data)
        test_aps = eval(model, test_data)
        end_time = time.time()
        print(f"Epoch {epoch}\tTraining Loss: {train_loss:.6f}" + 
              f"\tTrain APS: {train_aps:.6f}\tTest APS: {test_aps:.6f}" +
              f"\tTime Taken: {end_time-start_time:.2f} seconds")

model = CNNModel()
if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"  
device = torch.device(dev)  
model.to(device)

print("loading training dataset")
train_ds = data.load_dataset("data/train.csv", vocab_size=5000)
#train_ds = cnndata.load_dataset("data/train.csv", vocab_size=1000)
print("calling DataLoader for train dataset")
train_loader = DataLoader(train_ds, batch_size=10, collate_fn=data.collate, shuffle=True)
#train_loader = DataLoader(train_ds, batch_size=10, collate_fn=cnndata.collate, shuffle=True)
print("loading test dataset")
test_ds = data.load_dataset("data/test.csv", encoding=train_ds.encoding)
#test_ds = cnndata.load_dataset("data/test.csv", encoding=train_ds.encoding)
print("calling Dataloader for test dataset")
test_loader = DataLoader(test_ds, batch_size=10, collate_fn=data.collate, shuffle=False)
#test_loader = DataLoader(test_ds, batch_size=10, collate_fn=cnndata.collate, shuffle=False)
print("calling train on the model")

train(model, train_loader, test_loader, 20)