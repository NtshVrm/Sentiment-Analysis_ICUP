import numpy as np 
import re
from collections import defaultdict
import warnings
warnings.filterwarnings(action="ignore", category=DeprecationWarning) 
warnings.filterwarnings(action='ignore', category=FutureWarning)
import pandas as pd 

train = pd.read_csv('train_75.csv') 
test = pd.read_csv('test_25.csv')

def preprocess(tweet):
    cleaned_str = re.sub('[^a-z\s]+',' ',tweet, flags=re.IGNORECASE)
    cleaned_str = re.sub('(\s+)',' ',cleaned_str) #multiple spaces are replaced by a single space.
    cleaned_str = cleaned_str.lower()
    
    return cleaned_str

class naiveBayes:

    def __init__(self, no_of_classes):
        self.classes = no_of_classes
    
    def form_BoW(self, tweet, dict_index):
        if isinstance(tweet, np.ndarray):
            tweet = tweet[0] #tweet[0] because that passes all the tweets in that category.
        for i in tweet.split():
            self.bow_dicts[dict_index][i]+=1
    
    def train(self, labels, ds):
        self.examples = ds
        self.labels = labels
        #array of BoW dicts
        self.bow_dicts = np.array([defaultdict(lambda:0) for i in range(self.classes.shape[0])]) 
        
        if not isinstance(self.examples,np.ndarray):
            self.examples = np.array(self.examples)
        if not isinstance(self.labels,np.ndarray):
            self.labels = np.array(self.labels)
        
        #forming the BoW for each category 
        for index, category in enumerate(self.classes):
            all_category_tweets = self.examples[self.labels==category]

            cleaned_examples = [preprocess(i) for i in all_category_tweets]
            cleaned_examples = pd.DataFrame(data = cleaned_examples)

            np.apply_along_axis(self.form_BoW , 1, cleaned_examples, index)

            prob_classes = np.zeros(self.classes.shape[0])
            all_words = list()

            category_word_count = np.zeros(self.classes.shape[0])

        for index, category in enumerate(self.classes):
            prob_classes[index]=np.sum(self.labels==category)/float(self.labels.shape[0])

            #calculating frequency of each word 
            count = list(self.bow_dicts[index].values())

            category_word_count[index] = np.sum(np.array(list(self.bow_dicts[index].values())))
            
            all_words += self.bow_dicts[index].keys() #updates the list for all words 

        #combining all words (unique) to form vocabulary 
        self.vocabulary = np.unique(np.array(all_words))
        self.vocab_length = self.vocabulary.shape[0]

        denominators = np.array([category_word_count[index] + self.vocab_length + 1 for index, category in enumerate(self.classes)])
        '''
        A list of tuples containing information for each category.
        '''

        self.cats_info=[(self.bow_dicts[index],prob_classes[index],denominators[index]) for index,cat in enumerate(self.classes)]                               
        self.cats_info=np.array(self.cats_info)        
    
    def getProb(self, test_example):
        likelihood_prob = np.zeros(self.classes.shape[0]) #to store probability wrt to each class

        #finding prob wrt to each class
        for index, category in enumerate(self.classes):
            for test_token in test_example.split():
                token_count = self.cats_info[index][0].get(test_token, 0) + 1 #adding 1 

                #getting prob of this token

                token_prob = token_count/float(self.cats_info[index][2])
                likelihood_prob[index] += np.log(token_prob)
        
        final_prob = np.zeros(self.classes.shape[0])
        for index, category in enumerate(self.classes):
            final_prob[index] = likelihood_prob[index] + np.log(self.cats_info[index][1])

        return final_prob

    def test(self, test_ds):
        predictions = list()

        for ex in test_ds:
            cleaned_ex = preprocess(ex)

            post_prob = self.getProb(cleaned_ex)

            predictions.append(self.classes[np.argmax(post_prob)])
    

        return np.array(predictions)

train_data = train['tweet']
train_labels = train['label']

test_data = test['tweet']
test_labels = test['label']

nb = naiveBayes(np.unique(train_labels))

nb.train(train_labels, train_data)


def final():
    print('-------------------------------------------Training in progress-----------------------------------')

    print('...')
    print('...')

    print('--------------------------------------------Training complete-------------------------------------')
    pred_prob = nb.test(test_data)

    test_acc = np.sum(pred_prob==test_labels)/float(test_labels.shape[0])

    print('Accuracy of classifying:', test_acc*100,'%')
    print()

    usr_in = input('Please enter a tweet: ')
    print()
    out = nb.test([usr_in])

    print('Prediction made: ')
    print()
    if out[0]==0:
        print('Not Hate Speech. :)')
    else:
        print('Hate Speech. -_-')                 


