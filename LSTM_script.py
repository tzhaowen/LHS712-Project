import os
import pickle
import numpy as np
from numpy.random import seed
seed(1111)
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, SpatialDropout1D, LSTM, Dense
from sklearn.model_selection import StratifiedKFold
from keras.callbacks.callbacks import EarlyStopping
import pandas as pd
import itertools
import tensorflow as tf
tf.random.set_seed(2222)

def main():
    
    # read the data from pickle file
    with open('train.pkl','rb') as f:
        train_data = pickle.load(f)
    
    with open('validation.pkl','rb') as f:
        test_data = pickle.load(f)
    
    # cross validation
    

    # Will be using the cleaned-up text and apply keras tokenizer
    train_X  = train_data['processed_text'].to_numpy()
    test_X  = test_data['processed_text'].to_numpy()
    train_Y  = train_data['EncodedLabel'].to_numpy()
    test_Y  = test_data['EncodedLabel'].to_numpy()

    # tokenize
    MAX_NB_WORDS = 20000
    
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    tokenizer.fit_on_texts(np.concatenate([train_X,test_X],axis =0))
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    train_XX = tokenizer.texts_to_sequences(train_X)
    test_XX = tokenizer.texts_to_sequences(test_X)
    # find the maxium length of the texts for them to be padded
    #for sequence in train_XX:
    #    if len(sequence)>MAX_SEQUENCE_LENGTH:
    #        MAX_SEQUENCE_LENGTH = len(sequence)
    MAX_SEQUENCE_LENGTH = 140
    train_XX = pad_sequences(train_XX, maxlen=MAX_SEQUENCE_LENGTH)
    test_XX = pad_sequences(test_XX, maxlen=MAX_SEQUENCE_LENGTH)
    
    
    
    # EMBEDDING_DIM = [2, 5, 10, 15, 20, 25, 50, 100]
    # EMBEDDING_DIM = [5, 10]
    EMBEDDING_DIM = [50]
    # DROP_OUT = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.35, 0.5, 0.75]
    # DROP_OUT = [0.2, 0.5]
    DROP_OUT = [0.2]
    performance_df = pd.DataFrame(index=range(len(EMBEDDING_DIM)*len(DROP_OUT)) , columns = ['drop_out','embedding_dim','cv_scores'])
    row_n = 0

    cv_score = popi_run(train_XX,train_Y,EMBEDDING_DIM[0],DROP_OUT[0],test_without_cv = 1,test_XX = test_XX,test_Y = test_Y)
    for embedding_dim, drop_out in itertools.product(EMBEDDING_DIM,DROP_OUT):
        print('Running iteration: embedding_dim = {} drop_out = {}'.format(embedding_dim,drop_out))
        cv_score = popi_run(train_XX,train_Y,embedding_dim,drop_out)
        performance_df.loc[row_n,'drop_out'] = drop_out
        performance_df.loc[row_n,'embedding_dim'] = embedding_dim
        performance_df.loc[row_n,'cv_scores'] = cv_score
        row_n +=1

    performance_df.to_csv('performance_LSTM.csv',header=True, index=False)
    

def popi_run(train_XX, train_Y, embedding_dim, drop_out,test_without_cv = 0,test_XX = None,test_Y = None):
    epochs = 40
    batch_size = 64
    CV_seed = [123]
    K = 5
    MAX_NB_WORDS = 20000
    cv_scores = []
    for i,seed in enumerate(CV_seed):
        skf = StratifiedKFold(n_splits=K,random_state=seed)
        if test_without_cv == 1:
            model = Sequential()
            model.add(Embedding(MAX_NB_WORDS, embedding_dim, input_length=train_XX.shape[1]))
            model.add(SpatialDropout1D(drop_out))
            model.add(LSTM(100, dropout=drop_out, recurrent_dropout=drop_out))
            model.add(Dense(4, activation='softmax'))
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
            # transform data to onehot
            train_y = label_to_onehot(train_Y)
            val_y = label_to_onehot(test_Y)
            history = model.fit(train_XX, train_y, epochs=epochs, 
            validation_data=(test_XX, val_y),batch_size=batch_size,
            callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
            pred_y = model.predict(test_XX)
            positive_lab = 0 # f-1 score for a class
            accr = binary_accuracy(val_y,pred_y,positive_lab)
            cv_scores.append(accr)
            print('[INFO] CV F1-score result: {} ({})'.format(np.mean(cv_scores),np.std(cv_scores)))
        else:
            for train_index, val_index in skf.split(train_XX, train_Y): # train validation split
                model = Sequential()
                model.add(Embedding(MAX_NB_WORDS, embedding_dim, input_length=train_XX.shape[1]))
                model.add(SpatialDropout1D(drop_out))
                model.add(LSTM(100, dropout=drop_out, recurrent_dropout=drop_out))
                model.add(Dense(4, activation='softmax'))
                model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

                train_x, val_x = train_XX[train_index], train_XX[val_index]
                train_y, val_y = train_Y[train_index], train_Y[val_index]


                # transform data to onehot
                train_y = label_to_onehot(train_y)
                val_y = label_to_onehot(val_y)

                history = model.fit(train_x, train_y, epochs=epochs, 
                validation_data=(val_x, val_y),batch_size=batch_size,
                callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

                pred_y = model.predict(val_x)

                positive_lab = 0 # f-1 score for a class
                accr = binary_accuracy(val_y,pred_y,positive_lab)

                cv_scores.append(accr)

    print('[INFO] CV F1-score result: {} ({})'.format(np.mean(cv_scores),np.std(cv_scores)))
    return np.mean(cv_scores)



def label_to_onehot(train_Y):
    onehot_train_Y = np.zeros([len(train_Y),4])
    for i in range(len(onehot_train_Y)):
        onehot_train_Y[i,train_Y[i]]=1

    return onehot_train_Y

def binary_accuracy(true_Y,pred_Y,positive_lab):

    pred_Y = np.argmax(pred_Y,axis=1).flatten()
    true_Y = np.argmax(true_Y,axis=1).flatten()
    bool1 = true_Y==positive_lab # real positive
    bool2 = pred_Y==positive_lab # test positive
    bool3 = np.array(bool1)&np.array(bool2) # true positive
    precision = sum(bool3)/sum(bool1)
    recall = sum(bool1)/sum(bool2)
    # return f-1 score for positive label
    return 2*(precision * recall)/(precision + recall)

if __name__ == '__main__':
    main()