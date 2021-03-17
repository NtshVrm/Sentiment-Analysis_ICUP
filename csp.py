import re
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import string
import nltk
import warnings
from PIL import Image
warnings.filterwarnings("ignore",category=DeprecationWarning)

train  = pd.read_csv('train_75.csv')
test = pd.read_csv('test_25.csv')

train.head()

combi = train.append(test, ignore_index=True)
# remove twitter handles (@user)
combi['tidy_tweet']=combi['tweet'].str.replace("@user"," ")
# remove special characters, numbers, punctuations

combi['tidy_tweet'] = combi['tidy_tweet'].str.replace("[^a-zA-Z#]", " ")

combi['tidy_tweet'] = combi['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))

#make each tweet into a list of words to extract #'s

combi['list_tidy_tweet']=combi['tidy_tweet'].apply(lambda x: x.split(' '))




#now extract all words with prefix '#' 
hashtags=[]
check=[]
def all_hash():
    for i in range(0,200):
        l=(combi['list_tidy_tweet'][i])
    
        for j in l:
            listofwords=j
            listofwords=listofwords.split(' ')
            for k in listofwords:
                check.append(k)
                for m in check:
                    if "#" in m:
                        if m not in hashtags:
                            hashtags.append(m)
                        
all_hash()
print(hashtags,'\n')
print('\n')

##TO GET POSITIVE HASHTAGS
pos=[]
positivehash=[]
def pos_hash():
    combi['pos_tidy_tweet']=combi['tidy_tweet'][combi['label']==0]


    for i in range(0,200):
    
        if type(combi['pos_tidy_tweet'][i])!=float:
            pos.append(combi['pos_tidy_tweet'][i])   
    for k in pos:
        for j in k.split(' '):
            if '#' in j:
                if j not in positivehash:
                    positivehash.append(j)
pos_hash()
print(positivehash,'\n')
print('\n')
            
##TO EXTRACT NEGATIVE HASHATGS
neg=[]
negativehash=[]
def neg_hash():
    combi['neg_tidy_tweet']=combi['tidy_tweet'][combi['label']==1]

    for i in range(0,200):
    
        if type(combi['neg_tidy_tweet'][i])!=float:
            neg.append(combi['neg_tidy_tweet'][i])
    for k in neg:
        for j in k.split(' '):
            if '#' in j:
                if j not in negativehash:
                    negativehash.append(j)

neg_hash()
print(negativehash)

combi.head()










