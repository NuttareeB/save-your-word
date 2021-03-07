# A part of this code is inspired by Professor Lee (Stefan Lee) from OSU course CS539.
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
    EN_TEXT = Field(tokenize=tokenize_en, init_token = '<sos>', eos_token = '<eos>', lower = True)

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
    
    BATCH_SIZE = 20
    # generate the iterator
    train_iter, val_iter = BucketIterator.splits((train, val), batch_size=BATCH_SIZE, device = device, sort_key=lambda x: len(x.sentence1), shuffle=True)
    
    # =============================================================================
    # to print out the tokenized value in the first batc, use the following codes:
    #    
    # batch = next(iter(train_iter))
    # print(batch.sentence1)
    # =============================================================================

    input_size = 0
    dest_size = 0
    word_embed_dim = 256
    hidden_dim = 512
    dropout_rate = 0.5

    enc = Encoder(input_size, hidden_dim, hidden_dim, word_embed_dim, dropout_rate)
    dec = Decoder(dest_size, hidden_dim, hidden_dim, word_embed_dim, dropout_rate)
    model = Seq2Seq(enc, dec, device).to(device)
    
    epoch = 10
    
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(params = parameters)
    
    tag_pad_idx = EN_TEXT.vocab.stoi[EN_TEXT.pad_token]
    crit = nn.CrossEntropyLoss(ignore_index = tag_pad_idx).to(device)
    
    # start training
    for i in range(epoch):
        t_loss, t_acc = training(model, train_iter, optimizer, crit, tag_pad_idx)
        

def tokenize_en(sentence):
    return [token.text for token in en.tokenizer(sentence)]

def load_data(df, threshold):
    # remove the rows whose score is less than the threshold
    df['remove'] = [row <= threshold for row in df['score']]
    rslt_df = df[df['remove'] != True] 
    return rslt_df[['sentence1', 'sentence2']]

def training(model, train_iter, optimizer, crit, tag_pad_idx):
    model.train()
    epoch_train_loss = 0
    epoch_test_loss = 0
    
    for batch in train_iter:
        
        batch_sentence1 = batch.sentence1

def evaluate(model, val_iter, crit, epoch):
    
    model.eval()
    
    epoch_loss = 0
    
    with torch.no_grad():
        for i, batch in enumerate(val_iter):
            src = batch.sentence1
            trg = batch.sentence2

            output = model(src, trg) 

            output_dim = output.shape[-1]
            
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            loss = crit(output, trg)

            epoch_loss += loss.item()
    return epoch_loss / len(val_iter)





class Encoder(nn.Module):
    def __init__(self, input_size, encode_hidden_dim, decode_hidden_dim, embedding_dim, dropout=0.5):
        super().__init__()
        
        self.encode_hidden_dim = encode_hidden_dim

        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim,
            hidden_size=encode_hidden_dim,
            num_layers=1,
            bidirectional=True)
        self.fc = nn.Linear(encode_hidden_dim * 2, decode_hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input):

        embedded = self.dropout(self.embedding(input))

        encode_hts, _ = self.lstm(embedded)

        last_forward = enc_hidden_states[-1, :, :self.encode_hidden_dim]
        first_backward = enc_hidden_states[0, :, self.encode_hidden_dim:]

        out = F.relu(self.fc(torch.cat((last_forward, first_backward), dim = 1)))

        return encode_hts, out

class Decoder(nn.Module):
    def __init__(self, output_dim, encode_hidden_dim, decode_hidden_dim, embedding_dim, dropout=0.5,):
        super().__init__()

        self.output_dim = output_dim
        
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, decode_hidden_dim)
        self.fc_out = nn.Linear((encode_hidden_dim * 2) + decode_hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input, hidden, encoder_outputs):

        input = input.unsqueeze(0)        
        embedded = self.dropout(self.embedding(input))
        
        output, hidden = self.lstm(embedded, hidden.unsqueeze(0))
        
        prediction = self.fc_out(output)
        
        return prediction

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, input_sentence, target_sentence):
        
        batch_size = input_sentence.shape[1]
        target_len = target_sentence.shape[0]
        target_vocab_size = self.decoder.output_dim
        
        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(self.device)
        
        encoder_outputs, hidden = self.encoder(input_sentence)
                
        for t in range(1, target_len):
            output, hidden, a = self.decoder(target_sentence[t-1], hidden, encoder_outputs)
            outputs[t] = output

        return outputs



if __name__ == "__main__":
    main()
