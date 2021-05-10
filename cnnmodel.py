from math import floor
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import average_precision_score
import time
import data
from torch.utils.data import DataLoader
import embedding_utils as em

class CNNModel(nn.Module):
    
    def __init__(self, vocab_size=1000, kernel_size=10, num_filter_maps=16, embed_size=100, dropout=0.5, embedding=None):
        super(CNNModel, self).__init__()
        self.embed_size = embed_size
        self.embed_drop = nn.Dropout(p=dropout)
        
         # embedding
        if embedding is None:
            self.embed = nn.Embedding(vocab_size+1, embed_size)
            xavier_uniform_(self.embed.weight)
        else:
            embedding_tensor = torch.FloatTensor(embedding)
            self.embed = nn.Embedding.from_pretrained(embedding_tensor)
        
        self.conv1 = nn.Conv1d(self.embed_size, num_filter_maps, kernel_size, padding=int(floor(kernel_size/2)))
        xavier_uniform_(self.conv1.weight)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # final layer
        self.final = nn.Linear(num_filter_maps, 1)
        xavier_uniform_(self.final.weight)
    
    def forward(self, text):
        text = text.transpose(1,-1)
        embedded = self.embed(text)
        embedded = self.embed_drop(embedded)
        embedded = embedded.transpose(1,-1)
        conved1 = self.conv1(embedded)
        conved1 = torch.tanh(conved1)
        conved1 = self.pool(conved1)

        y = torch.sigmoid(self.final(conved1.transpose(1,-1))).squeeze()

        return y 
        
        

def eval(model, data, max_batches=0):
    model.eval()
    y_pred = torch.LongTensor()
    y_true = torch.LongTensor()
    for i, (x, y) in enumerate(data):
        if max_batches > 0 and i > max_batches:
            break
        y_hat = model(x)
        y_pred = torch.cat((y_pred, y_hat.detach().to('cpu')), dim=0)
        y_true = torch.cat((y_true, y.detach().to('cpu')), dim=0)
    return average_precision_score(y_true, y_pred)



def train(model, train_data, test_data, n_epochs):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    for epoch in range(n_epochs):
        start_time = time.time()
        model.train()
        train_loss = 0
        for i, (x, y) in enumerate(train_data):
            print(f"{i}/{len(train_data)}", end="\r")
            optimizer.zero_grad()
            y_hat = model(x)
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


if __name__ == '__main__':

    epochs = 20

    print("Starting CNN Run with Word2Vec embedding")
    w2v = em.load_embedding('embedding/word2vec_250d_1win.wordvectors')
    print("loading training dataset")
    train_ds = data.load_dataset("data/train.csv", encoding=w2v.key_to_index.get)
    train_loader = DataLoader(train_ds, batch_size=10, collate_fn=data.collate, shuffle=True)
    print("loading test dataset")
    test_ds = data.load_dataset("data/test.csv", encoding=w2v.key_to_index.get)
    test_loader = DataLoader(test_ds, batch_size=10, collate_fn=data.collate, shuffle=False)
    vocab_size, embed_size = w2v.vectors.shape
    model = CNNModel(vocab_size=vocab_size, embed_size=embed_size, num_filter_maps = 16, embedding=w2v.vectors)
    print("training the model")
    train(model, train_loader, test_loader, epochs)


    print("\n\nStarting CNN Run with embedding from scratch")
    model = CNNModel()
    print("loading training dataset")
    train_ds = data.load_dataset("data/train.csv", vocab_size=1000)
    train_loader = DataLoader(train_ds, batch_size=10, collate_fn=data.collate, shuffle=True)
    print("loading test dataset")
    test_ds = data.load_dataset("data/test.csv", encoding=train_ds.encoding)
    test_loader = DataLoader(test_ds, batch_size=10, collate_fn=data.collate, shuffle=False)
    print("training the model")
    train(model, train_loader, test_loader, epochs)


