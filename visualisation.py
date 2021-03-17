import re
import numpy as np 
import matplotlib.pyplot as plt 
import string
import nltk
from PIL import Image
from wordcloud import WordCloud
from nltk.stem import LancasterStemmer # For more aggressive stemming(it will legit find the closest word)one can use LancasterStemmer
import seaborn as sns
import warnings
warnings.filterwarnings(action="ignore", category=DeprecationWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)
import pandas as pd 

train  = pd.read_csv('train_75.csv')
test = pd.read_csv('test_25.csv')

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
    for i in range(0,10):
        l=(combi['list_tidy_tweet'][i])
    
        for j in l:
            listofwords=j
            listofwords=listofwords.split(' ')
            for k in listofwords:
                check.append(k)
                for m in check:
                    if "#" in m:
                        #if m not in hashtags:
                            hashtags.append(m)
                        
all_hash()
#print(hashtags,'\n')

##TO GET POSITIVE HASHTAGS
pos=[]
positivehash=[]
def pos_hash():
    combi['pos_tidy_tweet']=combi['tidy_tweet'][combi['label']==0]


    for i in range(0,10):
    
        if type(combi['pos_tidy_tweet'][i])!=float:
            pos.append(combi['pos_tidy_tweet'][i])   
    for k in pos:
        for j in k.split(' '):
            if '#' in j:
                #if j not in positivehash:
                    positivehash.append(j)
pos_hash()
#print(positivehash,'\n')
print('\n')
            
##TO EXTRACT NEGATIVE HASHATGS
neg=[]
negativehash=[]
def neg_hash():
    combi['neg_tidy_tweet']=combi['tidy_tweet'][combi['label']==1]

    for i in range(0,10):
    
        if type(combi['neg_tidy_tweet'][i])!=float:
            neg.append(combi['neg_tidy_tweet'][i])
    for k in neg:
        for j in k.split(' '):
            if '#' in j:
                #if j not in negativehash:
                    negativehash.append(j)

neg_hash()
#print(negativehash)




##'''PASTE THIS IN CSP.py AND RUN IT'''### 


#------------------------------BAR GRAPH FOR REG HASHTAGS--------------------------------------


#TO PLOT FRQUENCY DIST OF HASHTAGS( top 10)


def all_bar():
    a = nltk.FreqDist(hashtags)
    a=dict(a)
    #print(dict(a))
    d = pd.DataFrame({'Hashtag': list(a.keys()),
                      'Count': list(a.values())})
    # selecting top 10 most frequent hashtags     
    d = d.nlargest(columns="Count", n = 10) 
    plt.figure(figsize=(16,5))
    ax = sns.barplot(data=d, x= "Hashtag", y = "Count")
    ax.set(ylabel = 'Count')
    plt.show()
#all_bar()

#TO PLOT FREQUENCY DIST OF POSITIVE HASHTAGS

def pos_bar():
    a = nltk.FreqDist(positivehash)
    a = dict(a)

    d = pd.DataFrame({'Hashtag': list(a.keys()),
                  'Count': list(a.values())})
    # selecting top 10 most frequent hashtags     
    d = d.nlargest(columns="Count", n = 10) 
    plt.figure(figsize=(16,5))
    ax = sns.barplot(data=d, x= "Hashtag", y = "Count")
    ax.set(ylabel = 'Count')
    plt.show()
#pos_bar()

#TO PLOT FREQ DIST OF NEGATIVE HASHTAGS
def neg_bar():
    a = nltk.FreqDist(negativehash)

    d = pd.DataFrame({'Hashtag': list(a.keys()),
                      'Count': list(a.values())})
    # selecting top 10 most frequent hashtags     
    d = d.nlargest(columns="Count", n = 10) 
    plt.figure(figsize=(16,5))
    ax = sns.barplot(data=d, x= "Hashtag", y = "Count")
    ax.set(ylabel = 'Count')
    plt.show()
#neg_bar()



#--------------------------WORDCLOUD VISUALISATION------------------------------#
stemmer=LancasterStemmer()#To remove a particular prefix or suffix one can use RegexpStemmer

tokenized_tweet=combi['tidy_tweet'].apply(lambda x: x.split())
tokenized_tweet = tokenized_tweet.apply(lambda x:[stemmer.stem(i) for i in x])
    # print(tokenized_tweet)
for i in range(len(tokenized_tweet)):
    tokenized_tweet[i]=' '.join(tokenized_tweet[i])
combi['tidy_tweet'] = tokenized_tweet


def wordcloud_tweets():
    all_words = ' '.join([text for text in combi['tidy_tweet']]) 
    # for positive label is 0 and for negative label is 1. If you want both, then dont mention label itself.
    
    wordcloud=WordCloud(width=800,height=500,max_font_size=110).generate(all_words)

    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud,interpolation="bilinear")
    plt.axis('off') # To remove x and y axes
    plt.show()

def wordcloud_pos_tweets():
    pos_words = ' '.join([text for text in combi['tidy_tweet'][combi['label']==0]]) 
    # for positive label is 0 and for negative label is 1. If you want both, then dont mention label itself.
    
    wordcloud=WordCloud(width=800,height=500,max_font_size=110).generate(pos_words)

    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud,interpolation="bilinear")
    plt.axis('off') # To remove x and y axes
    plt.show()
def wordcloud_neg_tweets():
    neg_words = ' '.join([text for text in combi['tidy_tweet'][combi['label']==1]]) 
    # for positive label is 0 and for negative label is 1. If you want both, then dont mention label itself.
    
    wordcloud=WordCloud(width=800,height=500,max_font_size=110).generate(neg_words)

    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud,interpolation="bilinear")
    plt.axis('off') # To remove x and y axes
    plt.show()

    
#FOR REG HASHTAGS
def all_wc():
    a=nltk.FreqDist(hashtags)
    hasht=dict(a)
    mask1= np.array(Image.open('twitter.png'))
    wc = WordCloud(width=800, height=800, random_state=21, max_font_size=110,mask=mask1).generate_from_frequencies(hasht)

    plt.figure(figsize=(10, 7))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis('off')
    plt.show()
#all_wc()


#POSITIVE
def pos_wc():
    a=nltk.FreqDist(positivehash)
    poshash=dict(a)
    #mask2= np.array(Image.open('cloud.png'))
    wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate_from_frequencies(poshash)
    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.show()
#pos_wc()

##NEGATIVE HASHTAGS
def neg_wc():

    a=nltk.FreqDist(negativehash)
    neghash=dict(a)
    #mask2= np.array(Image.open('loc.png'))
    wordcloud = WordCloud(width=800, height=800, random_state=21, max_font_size=110).generate_from_frequencies(neghash)
    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.show()

#neg_wc()
