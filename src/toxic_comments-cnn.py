import sys, os, re, csv, codecs, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers

#split train and test sets
#will just create two files

def get_data(file_path):
    data = pd.read_csv(file_path)
    data['split'] = np.random.randn(data.shape[0],1)
    msk = np.random.rand(len(data)) <= 0.7
    
    train=data[msk]
    test=data[~msk]
    
    return((train, test))

def main():
    file_path='./data.csv'
    data_sets_tup = get_data(file_path)
    train = data_sets_tup[0]
    test = data_sets_tup[1]
    print("Sets should be split into test and train now..")
    print("Train size: ", len(train))
    print("Test size: ", len(test))
    print("############## Null testing ############\n", train.isnull().any(),test.isnull().any())
    
    classes_lst = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]
    
    y = train[list_classes].values
    
    comments_train_lst = train["comment_text"]
    comments_test_lst = test["comment_text"]
    
    
main()
