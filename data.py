import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch import tensor
import torch
import numpy as np
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
import re

NOTE_FILE="data/NOTEEVENTS.csv"
ADMISSION_FILE="data/ADMISSIONS.csv"
OUTLOC="data"

NON_CHARS = re.compile("[^a-z0-9]+")

def clean(line):
    line = line.lower()
    line = NON_CHARS.sub(' ', line)
    return line

class MIMICNotes(Dataset):
    """Dataset for mimic notes"""
    
    def __init__(self, data, readmit_cutoff = 30, encoding=None, vocab=None):
        assert encoding or vocab
        self.data = data
        self.encoding = encoding or vocab.get
        self.labels = data.DAYS_TO_READMIT < readmit_cutoff
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        datum = clean(self.data.iloc[index].TEXT)
        text = datum.split()
        encoded = [self.encoding(word) for word in text]
        encoded = np.array([w for w in encoded if w is not None])
        label = self.labels.iloc[index]
        return tensor(encoded, dtype=torch.long), tensor(label, dtype=torch.float)

def load_dataset(filename=f"{OUTLOC}/train.csv", df=None, readmit_cutoff=30,
                 encoding=None, vocab=None, vocab_size=1000, sample=1, split=None, random_state=19820618):
    df = df or pd.read_csv(filename)
    if sample < 1:
        df = df.sample(frac=sample)
    if encoding is None and vocab is None:
        vocab = get_vocab(df.TEXT, vocab_size)
    if split is not None:
        train, test = train_test_split(df, test_size=split, random_state=random_state)
        return MIMICNotes(train, readmit_cutoff, encoding, vocab), MIMICNotes(test, readmit_cutoff, encoding, vocab)
    else:
        return MIMICNotes(df, readmit_cutoff, encoding, vocab)

def get_vocab(text_data, vocab_size=1000):
    split = text_data.str.split()
    vocab = Counter([word.lower() for line in split for word in line])
    words = vocab.most_common(vocab_size)
    return {word: i for i, (word, count) in enumerate(words)}

def one_hot_encoder(vocab):
    n = len(vocab)
    def encode(x):
        return vocab.get(x, n)
    return encode

def collate(data):
    x,y = zip(*data)
    x = pad_sequence(x, batch_first=True)
    y = torch.stack(y)
    return x, y

def load_discharges(notes_file=NOTE_FILE):
    notes = pd.read_csv(notes_file)
    discharge_data = notes[notes.CATEGORY == "Discharge summary"]
    return discharge_data

def load_readmit_times(admit_file=ADMISSION_FILE):
    admits = pd.read_csv(admit_file)
    admits.index = admits.HADM_ID
    by_patient = admits.groupby("SUBJECT_ID")
    def time_to_readmit(df):
        df = df.sort_values(by="ADMITTIME")
        return (pd.to_datetime(df.ADMITTIME).shift(-1) - pd.to_datetime(df.DISCHTIME)).apply(lambda f: f.days)
    readmit = by_patient.apply(time_to_readmit)
    readmit.index = readmit.index.droplevel("SUBJECT_ID")
    readmit.name = "DAYS_TO_READMIT"
    return readmit

def run():
    parser = argparse.ArgumentParser(description="generate data")
    parser.add_argument("--notes", "-n", help="the location of the note events", default=NOTE_FILE)
    parser.add_argument("--admissions", "-a", help="the location of the admission events", default=ADMISSION_FILE)
    parser.add_argument("--out", "-o", help="the location to output the data", default=OUTLOC)
    parser.add_argument("--split", "-s", help="the split size of the test data", type=float, default=0.1)

    args = parser.parse_args()
    discharges = load_discharges(args.notes)
    readmits = load_readmit_times(args.admissions)
    data = discharges.join(readmits, on="HADM_ID")
    train, test = train_test_split(data, test_size=args.split, random_state=6181982)
    with open(f"{args.out}/train.csv", 'w') as f:
        train.to_csv(f)
    with open(f"{args.out}/test.csv", 'w') as f:
        test.to_csv(f)

if __name__ == "__main__":
    run()