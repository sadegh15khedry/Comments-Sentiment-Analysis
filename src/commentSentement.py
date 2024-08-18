#part 1 of project
import spacy
import pandas as pd
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import multilabel_confusion_matrix
import unicodedata
from numpy import nan
from keras.utils.np_utils import to_categorical
from keras import utils as np_utils
from pandas import read_csv
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras import layers
from sklearn.preprocessing import LabelEncoder

#parameters
mode = 'train'



if mode == 'train':
    #preprossesing dataset------------------------------------------------------------------
    dataset = pd.read_excel (r'BookFinal.xlsx')
    print(dataset.shape[0])


    #find missing data-------------------------------------------
    blanks = []
    for i in range(dataset.shape[0]):
        if (dataset.iloc[i,12]!=1 and dataset.iloc[i,12]!=0 and dataset.iloc[i,12]!=-1):
            blanks.append(i)

    print(len(blanks))
    dataset.drop(blanks,inplace=True)
    print(dataset.shape[0])

    #spliting dataset ------------------------------------
    x = dataset['comment']
    y = dataset['label']
    #one hot encoding our labels 1==>2 , -1==>0 , 0==>1
    encoder = LabelEncoder()
    encoder.fit(y)
    y = encoder.transform(y)
    y = np_utils.to_categorical(y)
    #print (y)

    sentences_train, sentences_test, y_train, y_test = train_test_split(x,y,test_size=.3)
    #print(sentences_train.shape)

    #vecrorizing comments---------------------------------------
    vectorizer = CountVectorizer(min_df=0, lowercase=False)
    vectorizer.fit(dataset['comment'].values.astype('U'))
    x_train = vectorizer.transform(sentences_train.values.astype('U'))
    x_test  = vectorizer.transform(sentences_test.values.astype('U'))

    #one hot encoding our output
    #encoder = LabelEncoder()
    #encoder.fit(y_train)
    #y_train = encoder.transform(y_train)
    #y_train = np_utils.to_categorical(y_train)
    #y_test = encoder.transform(y_test)
    #y_test = np_utils.to_categorical(y_test)
    #print(x_train.shape[1])
    #print(x_test.shape[1])

    #making our model--------------------------
    input_dim = x_train.shape[1]

    model = Sequential()
    model.add(layers.Dense(256, input_dim=input_dim, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(68, activation='relu'))
    model.add(layers.Dense(68, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(8, activation='relu'))
    model.add(layers.Dense(3, activation='sigmoid'))


    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.summary()


    model.fit(x_train, y_train,
              epochs=1,
              batch_size=15)



    #testing our model
    loss, accuracy = model.evaluate(x_test, y_test, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))




