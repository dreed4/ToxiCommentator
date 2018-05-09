import sys, os, re, csv, codecs, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D, Flatten, MaxPooling1D
from keras.models import Model, Sequential
from keras import initializers, regularizers, constraints, optimizers, layers
from setuptools.dist import sequence
from keras.layers.convolutional import Conv1D
from sklearn.metrics import confusion_matrix
from sympy.matrices.densetools import row

#split train and test sets
#will just create two files

def build_model2(num_words, max_vec_size, inp_len):
    model = Sequential()
    model.add(Embedding(num_words, max_vec_size, input_length=inp_len)) 
    model.add(Conv1D(256, 5, activation='relu'))
    model.add(MaxPooling1D(5))
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(MaxPooling1D(5))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(6, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer='adam', 
    metrics=['accuracy'])
    
    return model
def build_model1(num_words, max_vec_size, inp_len):
    '''
        
    '''
    print("Building model..")
    
    model = Sequential()
    model.add(Embedding(num_words, max_vec_size, input_length=inp_len))
    #previous model1 (should revert to this after testing) acc=~85%
    #model.add(Conv1D(64,3, activation='relu'))
    #model.add(Conv1D(32,3, activation='relu'))
    #model.add(Conv1D(16,3, activation='relu'))
    
    model.add(Conv1D(128,3, activation='relu'))
    model.add(MaxPooling1D(4))
    model.add(Conv1D(128,3, activation='relu'))
    model.add(MaxPooling1D(4))
    model.add(Conv1D(128,3, activation='relu'))
    model.add(MaxPooling1D(4))
    model.add(Conv1D(64,3,activation='relu'))
    model.add(MaxPooling1D(4))
    model.add(Flatten())
    #dropout to prevent overfitting; originally 0.2
    #how should we set this
    model.add(Dropout(0.3))
    
    #this layer was sigmoid, but changed to relu
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.3))
    
    model.add(Dense(6, activation='sigmoid'))
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
    num_eps=3
    
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
    y_train_toxic = np.array(train["toxic"].values)
    y_train_severe = np.array(train["severe_toxic"].values)
    y_train_obscene = np.array(train["obscene"].values)
    y_train_threat = np.array(train["threat"].values)
    y_train_insult = np.array(train["insult"].values)
    y_train_identity = np.array(train["identity_hate"].values)
    
    
    print("shapes: ", y_train_toxic.shape, y_train_severe.shape, y_train_obscene.shape, y_train_threat.shape, y_train_insult.shape, y_train_identity.shape)
    #stack into giant 2d array
    y_train_column_stack =np.column_stack((y_train_toxic, y_train_severe, y_train_obscene, y_train_threat, y_train_insult, y_train_identity))
    
    #same for test set
    y_test_toxic = np.array(test["toxic"].values)
    y_test_severe = np.array(test["severe_toxic"].values)
    y_test_obscene = np.array(test["obscene"].values)
    y_test_threat = np.array(test["threat"].values)
    y_test_insult = np.array(test["insult"].values)
    y_test_identity = np.array(test["identity_hate"].values)
    
    #empty list
    weights = []
    
    
    print("shapes: ", y_test_toxic.shape, y_test_severe.shape, y_test_obscene.shape, y_test_threat.shape, y_test_insult.shape, y_test_identity.shape)
    y_test_column_stack =np.column_stack((y_test_toxic, y_test_severe, y_test_obscene, y_test_threat, y_test_insult, y_test_identity))
    
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
    #y_train = y_train_toxic
    #y_test = y_test_toxic
    
    #y vals for test and train need to be one-hot vectors
    y_train = y_train_column_stack
    y_test = y_test_column_stack
    
    
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
    
    model = build_model1(num_words, max_vec_size, 1500)
    
    print("Fitting model..")
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=num_eps, 
    batch_size=100, verbose=1)
    
    
    labels = ['toxic', 'severe', 'obscene', 'threat', 'insult', 'identity']
    
    #confusion matrix building
    y_pred = model.predict(x_test)
    y_pred = (y_pred > 0.5)
    
    y_pred_transpose = y_pred.transpose()
    
    y_test_transpose = y_test.transpose()
    
    for i in range(len(y_pred_transpose)):
        this_pred = y_pred_transpose[i]
        this_test = y_test_transpose[i]
        this_label = labels[i]
        
        this_cm = confusion_matrix(this_test, this_pred)
        
        tn, fp, fn, tp = this_cm.ravel()
        
        print(this_label, " confusion matrix: \n    ", this_cm)
        print('true negatives = ', tn, ', false positives = ', fp, ', false negatives = ', fn, ', true positives = ', tp, '\n')
    
    print("\n",history.history.keys())
    print("acc: ", history.history['acc'])
    print("val_acc: ", history.history['val_acc'])
    
    print("loss: ", history.history['loss'])
    print("val_loss: ", history.history['val_loss'])
    
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
