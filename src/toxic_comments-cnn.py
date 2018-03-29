import sys, os, re, csv, codecs, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D, Flatten
from keras.models import Model, Sequential
from keras import initializers, regularizers, constraints, optimizers, layers
from setuptools.dist import sequence
from keras.layers.convolutional import Conv1D

#split train and test sets
#will just create two files

def build_model(num_words, max_vec_size, inp_len):
    '''
        This should build the model, but right now it isn't really ready to be 
        trained because I don't think the layers are correct
    '''
    print("Building model..")
    
    model = Sequential()
    model.add(Embedding(num_words, max_vec_size, input_length=inp_len))
    #TODO: add convolutional layers
    model.add(Conv1D(64,3, activation='sigmoid'))
    model.add(Conv1D(32,3, activation='sigmoid'))
    model.add(Conv1D(16,3, activation='sigmoid'))
    #I had this pooling layer but maybe not needed?
    #model.add(MaxPooling1D(5))
    model.add(Flatten())
    #dropout to prevent overfitting
    model.add(Dropout(0.2))
    
    model.add(Dense(180, activation='sigmoid'))
    model.add(Dropout(0.2))
    
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', 
    metrics=['accuracy'])
    
    return model
    
def get_data(file_path):
    print("getting data..")
    
    data = pd.read_csv(file_path)
    data['split'] = np.random.randn(data.shape[0],1)
    msk = np.random.rand(len(data)) <= 0.7
    
    train=data[msk]
    test=data[~msk]
    
    return((train, test, data))

    
def get_tokenizer(train_comments, nwords):
    print("getting tokenizer..")
    
    t = Tokenizer(num_words=nwords)
    texts = train_comments
    t.fit_on_texts(texts)
    sequences = t.texts_to_sequences(texts)
    
    return (t,sequences)

def pad_data(sequences,maxln):
    print("padding data..")
    
    #we could set a max_len here but chose not to, let it default to max
    data  = pad_sequences(sequences, maxlen = maxln)
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
    print("############## Testing for null vals ############\n", 
    all_data.isnull().any(),test.isnull().any())
    
    classes_lst = ["toxic","severe_toxic","obscene","threat","insult",
    "identity_hate"]
    
    
    
    #for now I'll leave all the dependent values in their own arrays
    y_train_toxic = train["toxic"].values
    y_train_severe = train["severe_toxic"].values
    y_train_obscene = train["obscene"].values
    y_train_threat = train["threat"].values
    y_train_insult = train["insult"].values
    y_train_identity = train["identity_hate"].values
    
    #same for test set
    y_test_toxic = test["toxic"].values
    
    
    
    comments_train_lst = train["comment_text"]
    comments_test_lst = test["comment_text"]
    
    #tokenizing and padding for training set
    tokenizer_train_tup = get_tokenizer(comments_train_lst, num_words)
    tokenizer_train = tokenizer_train_tup[0]
    sequences_train = tokenizer_train_tup[1]
    #the max is really about 1400, but just 
    #to be safe
    train_padded = pad_data(sequences_train, 1500)
    print("len(train_padded[0]): ", len(train_padded[0]))
    
    #tokenizing and padding for testing set
    tokenizer_test_tup = get_tokenizer(comments_test_lst, num_words)
    tokenizer_test = tokenizer_test_tup[0]
    sequences_test = tokenizer_test_tup[1]
    test_padded = pad_data(sequences_test, 1500)
    
    
     
    
    print("###### Padded data ######")
    print("train_padded: ", train_padded)
    print("\ntest_padded: ",  test_padded)
    print("\n########################")
    

    
    
    
    
    
    #setting x vals for test and train
    x_train = train_padded
    x_test = test_padded
    
    #y vals for test and train
    #for now, we're just doing one column (toxic), until we know it works and we
    #can do more
    y_train = y_train_toxic
    y_test = y_test_toxic
    
    print("Printing values that will go into fit..")
    print("x_train: ", x_train)
    print("Shape: (", len(x_train[0]), ",", len(x_train), ")")
    print("\n","x_test: ", x_test)
    
    print("\n","y_train: ", y_train)
    print("\n","y_test: ", y_test)
    
    ###### SHOULD BE DEPRECATED. TODO: REMOVE IF UNNECESSARY #######
    inp_len = len(train_padded[0])
    print("len(): ", inp_len)
    ################################################################
    
    model = build_model(num_words, max_vec_size, 1500)
    
    print("Fitting model..")
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=2, 
    batch_size=100, verbose=1)
    
    print(history.history.keys())
    
    #found the following code on machinelearningmastery
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
main()
