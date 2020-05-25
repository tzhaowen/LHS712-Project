import os
import re
import pandas as pd
import numpy as np
import itertools
from sklearn import preprocessing
from collections import Counter

from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
from pattern.en import suggest


DRUGLIST = ['oxycodone','oxycontin','percocet','diazepam','valium','olanzapine',
'zyprexa','amphetamine','adderall','methadone','dolophine','alprazolam','xanax',
'risperidone','risperdal','lisdexamphetamine','vyvanse','morphine','avinza','clonazepam',
'klonopin','aripiprazol','Aripiprazol','abilify','methylphenidate','ritalin','tramadol',
'conzip','lorazepam','ativan','asenapine','saphris','gaba','hydrocodone','vicodin',
'zohydro','quetiapine','seroquel','gabapentin','neurontin','buprenorphine','suboxone',
'pregabalin','lyrica']

def main(max_depth=4,learning_rate=0.05):
    # ----------------- READING DATA --------------
    train_data = pd.read_csv('./data/train.csv',header=0,sep=',',encoding = "ISO-8859-1")
    test_data = pd.read_csv('./data/validation.csv',header=0,sep=',',  encoding = "ISO-8859-1")

    dataset = train_data

    #------------------  Process the class labels ---------------------

    # Read class data
    train_Y = train_data['class'].to_numpy()
    test_Y  = test_data['class'].to_numpy()
    # remove the spaces
    train_Y = [re.sub('\s','',x) for x in train_Y]
    test_Y = [re.sub('\s','',x) for x in test_Y]
    # encode labels
    le = preprocessing.LabelEncoder()
    le.fit(train_Y)
    train_Y = le.transform(train_Y)
    test_Y = le.transform(test_Y)

    train_data.loc[:,'EncodedLabel'] = train_Y
    test_data.loc[:,'EncodedLabel'] = test_Y

    train_X = train_data['unprocessed_text'].to_numpy()
    test_X  = test_data['unprocessed_text'].to_numpy()

    # ----------- processing texts ---------------
    train_X = data_preprocessing(train_X)
    test_X = data_preprocessing(test_X)

    train_data.loc[:,'processed_text'] = train_X
    test_data.loc[:,'processed_text'] = test_X

    tknzr = TweetTokenizer()
    train_X = [tokenzation(x,tknzr) for x in train_X]
    test_X = [tokenzation(x,tknzr) for x in test_X]
    train_data.loc[:,'tokens'] = train_X
    test_data.loc[:,'tokens'] = test_X

    import pickle

    with open('train.pkl','wb') as f:
        pickle.dump( train_data,file=f)
    
    with open('validation.pkl','wb') as f:
        pickle.dump( test_data,file=f)

def reduce_lengthening(text):
    pattern = re.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1\1", text)
    
def data_preprocessing(X):
    '''
    text pre-processing 
    '''
    # convert all characters to lower case
    X = [x.lower() for x in X]
    # remove random characters out of ASCII in the text
    X = [x.encode("ascii","ignore") for x in X]
    X = [x.decode() for x in X]
    # remove the meaningless "_U" in the text
    X = [re.sub('_u',' ', x) for x in X]
    # replace @username with 
    X = [re.sub('@\w+','username',x) for x in X]
    # remove website links
    X = [re.sub(r'(http:\/\/www\.|https:\/\/www\.|http:\/\/|https:\/\/)?[a-z0-9]+([\-\.]{1}[a-z0-9]+)*\.[a-z]{2,5}(:[0-9]{1,5})?(\/.*)?','', x+' ') for x in X]
    # remove symbol
    X = [re.sub('[/(){}\[\]\|@,;]',' ', x) for x in X]
    X = [re.sub('[^0-9a-z ]',' ', x) for x in X]
    # consolidate multiple spaces
    X = [re.sub(' +',' ', x) for x in X]

    # spell correction
    for i,x in enumerate(X):
        print("[INFO] this is {}/{} tweet! ".format(i,len(X)))
        words = x.split()
        for j,word in enumerate(words):
            if word not in DRUGLIST:
                word = reduce_lengthening(word)
                try:
                    suggestion = suggest(word)[0]
                except:
                    suggestion = suggest(word)[0]

                if suggestion[1]>0.8: # do not change words with low confidence
                    words[j] = suggestion[0]
                else:
                    pass
                    # print(word,suggestion)
            else:
                word = 'drugname'# replace the drugnames with drugname
        X[i] = ' '.join(words)

    # remove stop words
    STOPWORDS = set(stopwords.words('english'))
    for i,x in enumerate(X):
        X[i] = ' '.join([word for word in x.split() if word not in STOPWORDS])
    return X

def tokenzation(text,tknzr):
    # tokenization
    return tknzr.tokenize(text)

if __name__ == '__main__':
    main()