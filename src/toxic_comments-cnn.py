import sys, os, re, csv, codecs, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model, Sequential
from keras import initializers, regularizers, constraints, optimizers, layers
from setuptools.dist import sequence

#split train and test sets
#will just create two files

def build_model(num_words, max_vec_size, inp_len):
    model = Sequential()
    model.add(Embedding(num_words, max_vec_size, input_length=inp_len))

def get_data(file_path):
    print("getting data..")
    
    data = pd.read_csv(file_path)
    data['split'] = np.random.randn(data.shape[0],1)
    msk = np.random.rand(len(data)) <= 0.7
    
    train=data[msk]
    test=data[~msk]
    
    return((train, test, data))

    
def get_tokenizer(train, nwords):
    print("getting tokenizer..")
    
    t = Tokenizer(num_words=nwords)
    texts = train["comment_text"]
    t.fit_on_texts(texts)
    sequences = t.texts_to_sequences(texts)
    
    return (t,sequences)

def pad_data(sequences,maxln):
    print("padding data..")
    
    #we could set a max_len here but chose not to, let it default to max
    data  = pad_sequences(sequences)
    return data

def main():
    num_words = 20000
    maxlen = 300
    max_vec_size = 128
    
    file_path='./data.csv'
    data_sets_tup = get_data(file_path)
    train = data_sets_tup[0]
    test = data_sets_tup[1]
    all_data = data_sets_tup[2]
    print("Sets should be split into test and train now..")
    print("Train size: ", len(train))
    print("Test size: ", len(test))
    print("############## Null testing ############\n", train.isnull().any(),test.isnull().any())
    
    classes_lst = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]
    
    #for now I'll leave all the dependent values in their own arrays
    y_toxic = train["toxic"].values
    y_severe = train["severe_toxic"].values
    y_obscene = train["obscene"].values
    y_threat = train["threat"].values
    y_insult = train["insult"].values
    y_identity = train["identity_hate"].values
    
    comments_train_lst = train["comment_text"]
    comments_test_lst = test["comment_text"]
    
    #TODO: do tokenization
    tokenizer_tup = get_tokenizer(train, num_words)
    t = tokenizer_tup[0]
    sequences = tokenizer_tup[1]
    
    train_padded = pad_data(sequences, maxlen) 
    
    print(train_padded)
   
    inp_len = len(train_padded[0])
    print("len(): ", inp_len)
    #TODO: add all layers for network
    model = build_model(num_words, max_vec_size, inp_len)
    
    pass

main()
