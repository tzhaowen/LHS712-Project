
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.2** of this notebook. 
# 
# ---

# # LHS 712 Classification

# ### Loading data  

import os
import re
import pandas as pd
import numpy as np
from numpy.random import seed
seed(1111)
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt
from itertools import product
import lightgbm as lgbm
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from sklearn.svm import SVC
from collections import Counter
import pickle

from nltk.corpus import stopwords
from pattern.en import suggest

DRUGLIST = ['oxycodone','oxycontin','percocet','diazepam','valium','olanzapine',
'zyprexa','amphetamine','adderall','methadone','dolophine','alprazolam','xanax',
'risperidone','risperdal','lisdexamphetamine','vyvanse','morphine','avinza','clonazepam',
'klonopin','aripiprazol','Aripiprazol','abilify','methylphenidate','ritalin','tramadol',
'conzip','lorazepam','ativan','asenapine','saphris','gaba','hydrocodone','vicodin',
'zohydro','quetiapine','seroquel','gabapentin','neurontin','buprenorphine','suboxone',
'pregabalin','lyrica']

def main(max_depth=4,learning_rate=0.05):

    
    # ----------------- PARAMETERS ------------
    CV_seed = [123]
    SVC_seed = [1234]
    K = 5               # K-fold cross-validation 
    test_size = 0.2 

    # ----------------- READING DATA --------------
    with open('train.pkl','rb') as f:
        train_data = pickle.load(f)
    
    with open('validation.pkl','rb') as f:
        test_data = pickle.load(f)

    

    #------------------  Training/test split ---------------------
    # train_data, test_data = train_test_split(dataset, test_size=test_size)
    train_X  = train_data['processed_text'].to_numpy()
    test_X  = test_data['processed_text'].to_numpy()
    train_Y  = train_data['EncodedLabel'].to_numpy()
    test_Y  = test_data['EncodedLabel'].to_numpy()
    test_IDs = test_data['id'].to_numpy()


def SVC_model()
    #------------------  Cross validation ---------------------
    CV_scores = []
    params= {
    'objective' :'multiclass',
    "num_class" : 4,
    'learning_rate' : learning_rate,
    'n_estimators':5000,
    'max_depth':max_depth, 
    'boosting_type' : 'gbdt',
    'verbose': -1,
    'seed':29
    }

    for seed in CV_seed:
        skf = StratifiedKFold(n_splits=K,random_state=seed)
        for train_index, val_index in skf.split(train_X, train_Y):
            # print("TRAIN:", train_index, "TEST:", val_index)
            train_x, val_x = train_X[train_index], train_X[val_index]
            train_y, val_y = train_Y[train_index], train_Y[val_index]

            # ------- transform -------
            train_x, val_x = data_transformer(train_x, val_x)

            # pred_y = LGBM(train_x, train_y,val_x,val_y,params)
            pred_y = SVM_classifier(train_x, train_y,val_x,val_y)
            pred_y_label = np.argmax(pred_y,axis=1)
            val_y_label = [1 if x == 0 else 0 for x in val_y]
            pred_y_b = [1 if x == 0 else 0 for x in pred_y_label]
            score = f1_score(val_y_label,pred_y_b)
            CV_scores.append(score)

    print('CV result: {}, {}'.format(np.mean(CV_scores), np.std(CV_scores)))
    print(CV_scores)

    # ------------------- performance in testin ----------
    train_X, test_X = data_transformer(train_X, test_X)

    pred_y = LGBM(train_X, train_Y,test_X,test_Y,params)
    
    pred_y_label = np.argmax(pred_y,axis=1)
    test_Y_label = [1 if x == 0 else 0 for x in test_Y]
    pred_y_b = [1 if x == 0 else 0 for x in pred_y_label]
    score = f1_score(test_Y_label,pred_y_b)

    print('Test result: {}'.format(score))
    return np.mean(CV_scores)



def data_transformer(train_x,val_x):
    vectorizer = TfidfVectorizer(stop_words='english')
    train_x = vectorizer.fit_transform(train_x)
    val_x = vectorizer.transform(val_x)

    # vectorizer = CountVectorizer(stop_words='english')
    # train_x_count = vectorizer.fit_transform(train_x).toarray()
    # val_x_count = vectorizer.transform(val_x).toarray()

    # count_those = ['workout','yoga',]

    return train_x, val_x


def LGBM(train_x, train_y,val_x,val_y,params):
    # making lgbm datasets for train and valid
    d_train = lgbm.Dataset(train_x, train_y)
    d_valid = lgbm.Dataset(val_x, val_y)

    # training with early stop
    bst = lgbm.train(params, d_train, 1000, valid_sets=[d_valid], verbose_eval=-1, early_stopping_rounds=100)

    pred_y = bst.predict(val_x)
    return pred_y


def SVM_classifier(train_x, train_y,val_x,val_y,gamma, C):
    '''
    Tune gamma and C, see: https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html
    '''
    
    clf = SVC(gamma='scale',C = 1, random_state = 1234,probability=True)
    clf.fit(train_x, train_y)
    return clf.predict(val_x)




if __name__ == '__main__':
    

    f = open('./lightGBM-1000.csv','w')
    print('depth','lr','score',sep=',',file=f)
    for learning_rate,max_depth in product(np.arange(0.01,0.1,0.01),np.arange(1,10,1)):
        cv = main(max_depth,learning_rate)
        print(max_depth,learning_rate,cv,sep=',',file=f)
        print(max_depth,learning_rate,cv,sep=',')

    f.close()
    



