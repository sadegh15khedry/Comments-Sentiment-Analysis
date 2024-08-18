#part 2 and 3 of project
#by changing the mode variable you can see the outcome for pruducts or users
import spacy
import pandas as pd
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import multilabel_confusion_matrix
import unicodedata
from numpy import nan
from pandas import read_csv
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras import layers
from sklearn.preprocessing import LabelEncoder



#preprossesing dataset------------------------------------------------------------------
dataset = pd.read_excel (r'BookFinal.xlsx')


blanks = []
for i in range(dataset.shape[0]):
    if (dataset.iloc[i,12]!=1 and dataset.iloc[i,12]!=0 and dataset.iloc[i,12]!=-1):
        blanks.append(i)

dataset.drop(blanks,inplace=True)




#by changing the mode variable you can see the outcome for pruducts or users
mode = 'product'

if mode == 'user':
    col = 3
elif mode == 'product':
    col = 0
else:
    col =0

mylist = []
myIds = []
for i in range(dataset.shape[0]):

    #lets see what we got
    sumOfLabels=0
    countOfLabels=0
    negetiveCommentCount=0
    posetiveCommentCount=0
    zeroCommentCount = 0
    for j in range(dataset.shape[0]):
        #checking we cal before
        if(dataset.iloc[i,col] in myIds):
            print('done that before')
            break
        elif (dataset.iloc[i,col] == dataset.iloc[j,col]):
            countOfLabels += 1
            sumOfLabels += dataset.iloc[j,12]
            #checkin value of comment
            if dataset.iloc[j,12] == 1:
                posetiveCommentCount +=1
            elif dataset.iloc[j,12] == -1:
                negetiveCommentCount += 1
            elif dataset.iloc[j,12] == 0:
                zeroCommentCount += 1


    #thereshhold section is here and can be used if neccecry
    #dataset.drop(mydel)
    #if sumOfLabel>0:
    #   sumofLabels = 0
    #elif sumOfLabel==0:
    #   sumofLabels = 0
    #elif sumOfLabel==0:
    #   sumofLabels = 0
    #else
    #   sumOfLabels = -1

    print ('id =  '+str(dataset.iloc[i,col])+'   count of labels = '+str(countOfLabels)+'   negetive label count = '+str(negetiveCommentCount)+'     zero label count = '+str(zeroCommentCount)+'     positive label count = '+str(posetiveCommentCount)+'   sumOflabels = '+str(sumOfLabels))
    mylist.append([dataset.iloc[i,col], sumOfLabels])
    myIds.append(dataset.iloc[i,col])

print(mylist)
