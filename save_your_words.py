# A part of this code is inspired by Professor Lee (Stefan Lee) from OSU course CS539.
import numpy as np
import pandas as pd
import pickle
import math
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

    input_size = len(EN_TEXT.vocab)
    dest_size = len(EN_TEXT.vocab)
    word_embed_dim = 256
    hidden_dim = 512
    dropout_rate = 0.5

    attn = SingleQueryScaledDotProductAttention(hidden_dim, hidden_dim)

    enc = Encoder(input_size, hidden_dim, hidden_dim, word_embed_dim, dropout_rate)
    dec = Decoder(dest_size, hidden_dim, hidden_dim, attn, word_embed_dim, dropout_rate)
    model = Seq2Seq(enc, dec, device).to(device)
    
    epoch = 10
    
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(params = parameters)
    
    tag_pad_idx = EN_TEXT.vocab.stoi[EN_TEXT.pad_token]
    crit = nn.CrossEntropyLoss(ignore_index = tag_pad_idx).to(device)
    
    # start training
    for i in range(epoch):
        t_loss = training(model, train_iter, optimizer, crit)
        print("epcoh:", i+1, "training loss:", t_loss)
        

def tokenize_en(sentence):
    return [token.text for token in en.tokenizer(sentence)]

def load_data(df, threshold):
    # remove the rows whose score is less than the threshold
    df['remove'] = [row <= threshold for row in df['score']]
    rslt_df = df[df['remove'] != True] 
    return rslt_df[['sentence1', 'sentence2']]

def training(model, iterator, optimizer, crit):
    model.train()
    epoch_loss = 0
    
    for batch in iterator:
        
        src = batch.sentence1
        trg = batch.sentence2
        
        optimizer.zero_grad()
        
        output = model(src, trg)
        
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        
        loss = crit(output, trg)
        
        loss.backward() 
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss/ len(iterator)

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

class SingleQueryScaledDotProductAttention(nn.Module):    
    def __init__(self, enc_hid_dim, dec_hid_dim, kq_dim=512):
        super().__init__()
        self.linear_q = nn.Linear(dec_hid_dim, kq_dim)
        self.linear_k = nn.Linear(enc_hid_dim*2, kq_dim)
        self.kq_dim = kq_dim
        self.dec_hid_dim = dec_hid_dim

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, hidden, encoder_outputs):

        q = self.linear_q(hidden)
        k_t = self.linear_k(encoder_outputs)
        v_t = encoder_outputs

        q_batch = q.unsqueeze(1)
        k = torch.transpose(k_t, 0, 1)
        k = torch.transpose(k, 1, 2)
        s = torch.bmm(q_batch, k) / math.sqrt(self.kq_dim)

        alpha = self.softmax(s)
        attended_val = torch.bmm(alpha, torch.transpose(v_t, 0, 1))
        
        alpha = alpha[:,-1,:]
        attended_val = attended_val[:,-1,:]

        assert attended_val.shape == (hidden.shape[0], encoder_outputs.shape[2])
        assert alpha.shape == (hidden.shape[0], encoder_outputs.shape[0])
        
        return attended_val, alpha


class Encoder(nn.Module):
    def __init__(self, input_size, encode_hidden_dim, decode_hidden_dim, embedding_dim, dropout=0.5):
        super().__init__()
        
        self.encode_hidden_dim = encode_hidden_dim

        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.rnn = nn.GRU(input_size=embedding_dim,
            hidden_size=encode_hidden_dim,
            num_layers=1,
            bidirectional=True)
        self.fc = nn.Linear(encode_hidden_dim * 2, decode_hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input):

        embedded = self.dropout(self.embedding(input))

        encode_hts, _ = self.rnn(embedded)

        last_forward = encode_hts[-1, :, :self.encode_hidden_dim]
        first_backward = encode_hts[0, :, self.encode_hidden_dim:]

        out = F.relu(self.fc(torch.cat((last_forward, first_backward), dim = 1)))

        return encode_hts, out

class Decoder(nn.Module):
    def __init__(self, output_dim, encode_hidden_dim, decode_hidden_dim, attn, embedding_dim, dropout=0.5,):
        super().__init__()

        self.output_dim = output_dim   
        print("decode_hidden_dim:", decode_hidden_dim)
        print("embedding_dim:", embedding_dim)
        
        self.attention = attn
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, decode_hidden_dim)
        self.fc_out = nn.Linear((encode_hidden_dim * 2) + decode_hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input, hidden, encoder_outputs):

        input = input.unsqueeze(0)        
        embedded = self.dropout(self.embedding(input))

        attended_feature, a = self.attention(hidden.squeeze(0), encoder_outputs) 

        output, hidden = self.rnn(embedded, hidden.unsqueeze(0))
        
        prediction = self.fc_out(torch.cat((output, attended_feature.unsqueeze(0)), dim = 2))
        
        return prediction, hidden.squeeze(0), a

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
