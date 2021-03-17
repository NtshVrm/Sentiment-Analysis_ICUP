import re # Regular Expressions
import pandas as pd
import numpy as np
import nltk
import string
from wordcloud import WordCloud,STOPWORDS
import matplotlib.pyplot as plt
train=pd.read_csv('train_E6oV3lV.csv')
test=pd.read_csv('test_tweets_anuFYb8.csv')
# train.head()# Gives only first 4-5 lines with id(sr.no),label(0 or 1) and tweet(which needs to be cleaned and processed). Just for checking purpose.
combi=train.append(test,ignore_index=True)# Prints first few of train and last few of test COMBINED
def remove_pattern(input_txt,pattern):# Definition of a function to remove certain words,characters,punctuations etc which will be called in future.
    r=re.findall(pattern,input_txt)
    for i in r:
        input_txt=re.sub(i,'',input_txt)
    return input_txt
combi['tidy_tweet']=np.vectorize(remove_pattern)(combi['tweet'],"@[\w]*")# Learn Vectorize, To remove certain things preceeded by @* etc
combi['tidy_tweet']=combi['tidy_tweet'].str.replace("[^a-zA-Z#]"," ")# To remove punctuations and special characters
combi['tidy_tweet']=combi['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))# To remove words with 3 letters or less
# By applying the following code, all relevant and useful data is retained whereas all the unwanted is removed from the raw data.
tokenized_tweet=combi['tidy_tweet'].apply(lambda x: x.split())# Done to split the cleaned tweets into individual words or you can directly import word_tokenize from nltk.tokenize for this purpose and then put the required text required in the parantheses of word_tokenize.
# For stemming and wordcloud
from nltk.stem import LancasterStemmer # For more aggressive stemming(it will legit find the closest word)one can use LancasterStemmer
stemmer=LancasterStemmer()#To remove a particular prefix or suffix one can use RegexpStemmer
tokenized_tweet=tokenized_tweet.apply(lambda x:[stemmer.stem(i) for i in x])
# print(tokenized_tweet)
for i in range(len(tokenized_tweet)):
    tokenized_tweet[i]=' '.join(tokenized_tweet[i])
combi['tidy_tweet']=tokenized_tweet
all_words=' '.join([text for text in combi['tidy_tweet'][combi['label']==0]]) # for positive label is 0 and for negative label is 1. If you want both, then dont mention label itself.
wordcloud=WordCloud(width=800,height=500,max_font_size=110).generate(all_words)
plt.figure(figsize=(10,5))
plt.imshow(wordcloud,interpolation="bilinear")
plt.axis('off') # To remove x and y axes
plt.show()



