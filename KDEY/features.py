#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import glob
import sys

if len(sys.argv) < 2:
    print ("Please enter the path to the data directory")


f = {}
for root, directories, filenames in os.walk(sys.argv[1]): #Change to Data_MPHI for MPHI dataset
    for directory in directories:             
        f[os.path.join(root, directory)] = {}
    for filename in filenames:
        f[root][os.path.splitext(filename)[0]] = os.path.join(root,filename)
print (f)


# In[4]:


import csv

lexicon = {}
with open('./MPQA/lexicon_easy.csv',"r") as csvfile:
    data = csv.reader(csvfile)
    for row in data:
        row[1] = int(row[1])
        row[2] = int(row[2])
        lexicon[row[0]] = {}
        lexicon[row[0]]['subj'] = row[1]
        lexicon[row[0]]['sent'] = row[2]


# In[5]:

# In[6]:


from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.corpus import sentiwordnet as swn
from nltk import sent_tokenize, word_tokenize, pos_tag
from nltk import ngrams,bigrams



def mpqa_subj(tweet):
    feat = 0
    score = 0
    tokens = word_tokenize(tweet)
    for token in tokens: 
        if token in lexicon:
            score+= lexicon [token]['subj']
    if score > 2 or score < -2:
        feat = 1
    return feat

def pot_adj(tweet):
    feat = 0
    tokens = word_tokenize(tweet)
    for token in tokens:
        synsets = wn.synsets(token)
        for s in synsets:
            if s.pos() == 'a':
                feat = 1
    return feat

        


# In[7]:



lemmatizer = WordNetLemmatizer()

def penn_to_wn(tag):
    """
    Convert between the PennTreebank tags to simple Wordnet tags
    """
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    return None


def swn_polarity(tweet):
    """
    Return a sentiment polarity: 0 = negative, 1 = positive
    """
 
    sentiment = 0.0
    tokens_count = 0
 
 
    tagged_sentence = pos_tag(word_tokenize(tweet))

    for word, tag in tagged_sentence:
        wn_tag = penn_to_wn(tag)
        if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
            continue

        lemma = lemmatizer.lemmatize(word, pos=wn_tag)
        if not lemma:
            continue

        synsets = wn.synsets(lemma, pos=wn_tag)
        if not synsets:
            continue

        # Take the first sense, the most common
        synset = synsets[0]
        swn_synset = swn.senti_synset(synset.name())

        if swn_synset.pos_score() - swn_synset.neg_score()>0:
            sentiment+=1
        elif swn_synset.pos_score() - swn_synset.neg_score()<0:
            sentiment+=-1
        tokens_count += 1

    # judgment call ? Default to positive or negative
    if not tokens_count:
        return 0
 
    return sentiment

def sentiword_mpqa_sentiment(tweet):
    feat = [0,0]
    feat[0] = swn_polarity(tweet)
    tokens = word_tokenize(tweet)
    for token in tokens:
        if token in lexicon:
            feat[1]+= lexicon[token]['sent']
    return feat

def target_yn(tweet,target):
    if target in tweet:
        return 1
    else:
        return 0

def word_ngrams(tweet,target):
    feat= [0,0,0]
    tweet_sent = tweet.split()
    target_sent =target.split()
    for k in range(1,4):
        for target_gram in ngrams(target_sent,k):
            for tweet_gram in ngrams(tweet_sent,k):
                if tweet_gram == target_gram:
                    feat[k-1] = 1
    return feat

def char_ngrams(tweet,target):
    feat= [0,0,0]
    tweet_sent = ''.join(e for e in tweet if e.isalnum())
    target_sent = ''.join(e for e in target if e.isalnum())
    for k in range(2,5):
        for target_gram in ngrams(target_sent,k):
            for tweet_gram in ngrams(tweet_sent,k):
                if tweet_gram == target_gram:
                    feat[k-2] = 1
    return feat


# In[8]:


import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

data = {}
for key in sorted(f.keys()):
    features = {}
    
    #Setting up the file names
    train_file = f[key]['train_clean']
    test_file = f[key]['test_clean']
    
    #Set up the TF-IDF Vectorizer
    corpus = []
    with open(train_file,'r') as fr:
        lines = fr.readlines()
        for line in lines:
            row = line.split('\t')
            if row[0] == 'ID':
                continue
            tweet = row[2].lower()
            corpus.append(tweet)
    with open(test_file,'r') as fr:
        lines = fr.readlines()
        for line in lines:
            row = line.split('\t')
            if row[0] == 'ID':
                continue
            tweet = row[2].lower()
            corpus.append(tweet)
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)
    X = X.todense()
    
    #Train Feature Extraction
    train_ph1 = []
    train_ph2 = []
    i = 0
    with open(train_file,'r') as fr:
        lines = fr.readlines()
        for line in lines:
            feature_vector_1 = []
            feature_vector_2 = []
            row = line.split('\t')
            if row[0] == 'ID':
                continue
            tweet = row[2].lower()
            target = row[1].lower()
            
            #First phase features
            feature_vector_1.append(mpqa_subj(tweet))
            feature_vector_1.append(pot_adj(tweet))
            #Uncomment the following line to add count vectorizer
            #feature_vector_1.extend(X[i].tolist()[0])

            #Second Phase features
            senti = sentiword_mpqa_sentiment(tweet)
            feature_vector_2.extend(senti)
            feature_vector_2.append(target_yn(tweet,target))
            feature_vector_2.extend(word_ngrams(tweet,target))
            feature_vector_2.extend(char_ngrams(tweet,target))
            #Uncomment the following line to add count vectorizer
            #feature_vector_2.extend(X[i].tolist()[0])
            
            i+=1
            
            category = row[3].rstrip()
#             print (category)
            if category == 'NONE':
                feature_vector_1.append(1)
                train_ph1.append(feature_vector_1)
            else:
                feature_vector_1.append(0)
                train_ph1.append(feature_vector_1)
                if category == 'AGAINST':
                    feature_vector_2.append(0)
                    train_ph2.append(feature_vector_2)
                else:
                    feature_vector_2.append(1)
                    train_ph2.append(feature_vector_2)
    
    #Load the test data and calculate features
    test = []
    test_y = []
    with open(test_file,'r') as fr:
        lines = fr.readlines()
        for line in lines:
            feature_vector_1 = []
            feature_vector_2 = []
            row = line.split('\t')
            if row[0] == 'ID':
                continue
            tweet = row[2].lower()
            target = row[1].lower()

            #First phase features
            feature_vector_1.append(mpqa_subj(tweet))
            feature_vector_1.append(pot_adj(tweet))
            #Uncomment the following line to add count vectorizer
            #feature_vector_1.extend(X[i].tolist()[0])

            #Second Phase features
            senti = sentiword_mpqa_sentiment(tweet)
            feature_vector_2.extend(senti)
            feature_vector_2.append(target_yn(tweet,target))
            feature_vector_2.extend(word_ngrams(tweet,target))
            feature_vector_2.extend(char_ngrams(tweet,target))
            #Uncomment the following line to add count vectorizer
            #feature_vector_2.extend(X[i].tolist()[0])
            
            i+=1 
            
            test.append((np.array(feature_vector_1,dtype=np.int32),np.array(feature_vector_2,dtype=np.int32)))
            
            category = row[3].rstrip()
            if category == 'NONE':
                test_y.append(2)
            else:
                if category == 'AGAINST':
                    test_y.append(1)
                else:
                    test_y.append(0)
#     print (train_ph1[0])
    train_ph1 = np.array(train_ph1, dtype = np.int32)
    train_ph2 = np.array(train_ph2,dtype = np.int32)
#     test = np.array(test,dtype = np.int32)
    test_y = np.array(test_y,dtype = np.int32)
    print (key)
    print (train_ph1.shape)
    print (train_ph2.shape)
    print (test_y.shape)
    data[key] = (train_ph1,train_ph2,test,test_y)
    print ()
    
        
               

# In[9]:


from sklearn.svm import LinearSVC
from sklearn.utils import shuffle

def predict(clf1,clf2,x):
    ph1 = [x[0]]
#     print (ph1)
    ph2 = [x[1]]
#     print (ph2)
    p1 = clf1.predict(ph1)
#     print (p1)
    if p1[0] == 1:
        return 2
    else:
        p2 = clf2.predict(ph2)
        if p2[0] == 1:
            return 0
        else:
            return 1
        
def score(y_true,y_pred):
    fav = [0,0,0]
    ag = [0,0,0]
    tot = [fav,ag]
    corr = 0
    for y_t,y_p in zip(y_true,y_pred):
        if y_t < 2:
            tot[y_t][2]+=1
        if y_p < 2:
            tot[y_p][1]+=1
        if y_t == y_p and y_t < 2:
            tot[y_t][0]+=1
        if y_t == y_p:
            corr+=1

        r0 = tot[0][0]/(tot[0][2]+1e-5)
        p0 = tot[0][0]/(tot[0][1]+1e-5)
        r1 = tot[1][0]/(tot[1][2]+1e-5)
        p1 = tot[1][0]/(tot[1][1]+1e-5)
        f0 = 2*r0*p0/(r0+p0+1e-5)
        f1 = 2*r1*p1/(r1+p1+1e-5)
        
        f_avg = (f0+f1)/2
    return tot,f_avg


# In[ ]:


import numpy as np
from sklearn.model_selection import GridSearchCV

parameters = {
    'C' : np.logspace(start = 0.001,stop = 5,num = 10),
    'dual' : [True, False]
}
tot = np.array([[0,0,0],[0,0,0]])
for key in data.keys():
    train_ph1 = shuffle(data[key][0])
    train_ph2 =shuffle(data[key][1])
    test = data[key][2]
    test_y = data[key][3]
    
    #Phase 1 Training
    train_ph1_x = train_ph1[:,:-1]
    train_ph1_y = train_ph1[:,-1]
#     print(train_ph1_x)
#     print(train_ph1_y)
    svc = LinearSVC(max_iter = 100000)
    clf1  = GridSearchCV(svc, parameters, cv=5)
    clf1.fit(train_ph1_x,train_ph1_y)
    
    #Phase 2 Training
    train_ph2_x = train_ph2[:,:-1]
    train_ph2_y = train_ph2[:,-1]
    
    clf2 = GridSearchCV(svc, parameters, cv=5)
    clf2.fit(train_ph2_x,train_ph2_y)
    
    #Testing
    preds = []
    for x in test:
        preds.append(predict(clf1,clf2,x))
#     print (preds)
#     print (test_y)
    print (key+":")
    arr, sc = score(test_y,preds)
    print ("Score - %f"%sc)
    print ()
    tot = tot + np.array(arr)
    cat = key.replace(sys.argv[1],"")
    with open(key+'/predict_'+cat+'.txt','w') as pf:
        for pred in preds:
            if pred == 0:
                pf.write("FAVOR\n")
            elif pred == 1:
                pf.write("AGAINST\n")
            elif pred == 2:
                pf.write("NONE\n")
    


# In[ ]:


r0 = tot[0][0]/(tot[0][2]+1e-5)
p0 = tot[0][0]/(tot[0][1]+1e-5)
r1 = tot[1][0]/(tot[1][2]+1e-5)
p1 = tot[1][0]/(tot[1][1]+1e-5)
f0 = 2*r0*p0/(r0+p0+1e-5)
f1 = 2*r1*p1/(r1+p1+1e-5)

f_avg = (f0+f1)/2
print ("Overall Score - %f"%f_avg)


# In[ ]:




