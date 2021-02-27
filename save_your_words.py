import numpy as np
import pandas as pd
import pickle
import spacy
import en_core_web_sm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import torchtext
from torchtext.data import get_tokenizer
from torchtext.data import SubwordField, Field, BPTTIterator
from torchtext.data import Field, BucketIterator, TabularDataset

from sklearn.model_selection import train_test_split

np.random.seed(0)
torch.manual_seed(0)

df = pd.read_table('data/2016_Oct_10--2017_Jan_08_full.txt', names=('score', 'sentence1', 'sentence2'))
en = spacy.load('en_core_web_sm')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
customize_threshold = 0.5

def main():
    # create the field object
    EN_TEXT = Field(tokenize=tokenize_en)

    train_file = 'data/train.csv'
    validation_file = 'data/val.csv'
    
    data = load_data(df, 0.5)
    train, val = train_test_split(data, test_size=0.1)
    train.to_csv(train_file, index=False)
    val.to_csv(validation_file, index=False)
    
    data_fields = [('sentence1', EN_TEXT), ('sentence2', EN_TEXT)]
    train, val = TabularDataset.splits(path='./', train=train_file, validation=validation_file, format='csv', fields=data_fields)
    
    # build  the vocabulary
    EN_TEXT.build_vocab(train, val)
    
    # generate the iterator
    train_iter = BucketIterator(train, batch_size=20, sort_key=lambda x: len(x.sentenc1), shuffle=True)
    
    # =============================================================================
    # to print out the tokenized value in the first batc, use the following codes:
    #    
    # batch = next(iter(train_iter))
    # print(batch.sentence1)
    # =============================================================================

def tokenize_en(sentence):
    return [token.text for token in en.tokenizer(sentence)]

def load_data(df, threshold):
    # remove the rows whose score is less than the threshold
    df['remove'] = [row <= threshold for row in df['score']]
    rslt_df = df[df['remove'] != True] 
    return rslt_df[['sentence1', 'sentence2']]


if __name__ == "__main__":
    main()
