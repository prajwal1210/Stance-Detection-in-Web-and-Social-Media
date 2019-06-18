#!/usr/bin/env python
# coding: utf-8

# In[9]:


import csv
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import wordninja
import re
import json
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import pandas as pd


# In[2]:
stemmer = PorterStemmer()

def load_glove_embeddings_set():
    word2emb = []
    WORD2VEC_MODEL = "../glove.6B.300d.txt"
    fglove = open(WORD2VEC_MODEL,"r")
    for line in fglove:
        cols = line.strip().split()
        word = cols[0]
        word2emb.append(word)
    fglove.close()
    return set(word2emb)

def create_normalise_dict(no_slang_data = "noslang_data.json", emnlp_dict = "emnlp_dict.txt"):
    print("Creating Normalization Dictionary")
    with open(no_slang_data, "r") as f:
        data1 = json.load(f)

    data2 = {}

    with open(emnlp_dict,"r") as f:
        lines = f.readlines()
        for line in lines:
            row = line.split('\t')
            data2[row[0]] = row[1].rstrip()

    normalization_dict = {**data1,**data2}
    #print(normalization_dict)
    return normalization_dict

word_dict,norm_dict = load_glove_embeddings_set(),create_normalise_dict()


# In[3]:


def sent_process(sent):
    sent = re.sub(r"[^A-Za-z0-9(),!?\'\`#]", " ", sent)
    sent = re.sub(r"#SemST", "", sent)
    sent = re.sub(r"#([A-Za-z0-9]*)", r"# \1 #", sent)
    #sent = re.sub(r"# ([A-Za-z0-9 ]*)([A-Z])(.*) #", r"# \1 \2\3 #", sent)
    #sent =  re.sub(r"([A-Z])", r" \1", sent)
    sent = re.sub(r"\'s", " \'s", sent)
    sent = re.sub(r"\'ve", " \'ve", sent)
    sent = re.sub(r"n\'t", " n\'t", sent)
    sent = re.sub(r"\'re", " \'re", sent)
    sent = re.sub(r"\'d", " \'d", sent)
    sent = re.sub(r"\'ll", " \'ll", sent)
    sent = re.sub(r",", " , ", sent)
    sent = re.sub(r"!", " ! ", sent)
    sent = re.sub(r"\(", " ( ", sent)
    sent = re.sub(r"\)", " ) ", sent)
    sent = re.sub(r"\?", " ? ", sent)
    sent = re.sub(r"\s{2,}", " ", sent)
    sent = sent.strip()
    word_tokens = sent.split()
    normalised_tokens = []
    for word in word_tokens:
        if word in norm_dict:
        #if False:
            normalised_tokens.extend(norm_dict[word].lower().split(" "))
            print(word," normalised to ",norm_dict[word])
        else:
            normalised_tokens.append(word.lower())
    wordninja_tokens = []
    for word in normalised_tokens:
        if word in word_dict:
            wordninja_tokens+=[word]
        else:
            wordninja_tokens+=wordninja.split(word)
    return " ".join(wordninja_tokens)


# In[4]:



def svc_param_selection(X, y, nfolds):
    Cs = [0.001, 0.01, 0.1, 1, 10,100 ]
    gammas = [0.001, 0.01, 0.1, 1]
    param_grid = [{'C': Cs, 'gamma' : gammas , 'kernel' : ['rbf']},{'C': Cs , 'gamma' : gammas , 'kernel' : ['linear']}]
    grid_search = GridSearchCV(svm.SVC(), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_


# In[21]:


def train(topic,bow=False,senti=False,sta=False,ent=False):
    save_file = topic+"_"
    print(topic)
    y_train = []
    y_test = []
    sentences = []
    features_ent = []
    features_sta = []
    features_senti = []
    senti_dict = {'Neutral' : 0, 'Positive' : 1, 'Negative' : 2}
    with open("./final_feature_set/{}_train.csv".format(topic),"r",encoding='latin-1') as f:
            reader = csv.DictReader(f, delimiter=',')
            done = False
            for row in reader:
                sentences.append(row['sentence'])
                if ent:
                    features_ent.append([row['ent_nut'],row['ent_pos'],row['ent_neg']])
                    if not done:
                        save_file = save_file + "ent_"
                if sta:
                    features_sta.append([row['sta_nut'],row['sta_sup'],row['sta_opp']])
                    if not done:
                        save_file = save_file + "sta_"
                if senti:
                    features_senti.append([senti_dict[row['senti']]])
                    if not done:
                        save_file = save_file + "senti_"
                done = True
                y_train.append(row['label'])
    L = len(sentences)
    with open("./final_feature_set/{}_test.csv".format(topic),"r",encoding='latin-1') as f:
            reader = csv.DictReader(f, delimiter=',')
            for row in reader:
                sentences.append(row['sentence'])
                if ent:
                    features_ent.append([row['ent_nut'],row['ent_pos'],row['ent_neg']])
                if sta:
                    features_sta.append([row['sta_nut'],row['sta_sup'],row['sta_opp']])
                if senti:
                    features_senti.append([senti_dict[row['senti']]])
                y_test.append(row['label'])
                
    all_features = []
    if bow:
        new_sentences = []
        for sent in sentences:
            tokens = word_tokenize(sent)
            tokens = [stemmer.stem(token) for token in tokens]
            ret = " ".join(w for w in tokens)
            new_sentences.append(ret)
        save_file = save_file + "bow_"
        vectorizer = CountVectorizer(stop_words='english',ngram_range=(1,1),min_df = 2)
        features_bow = vectorizer.fit_transform(new_sentences)
        all_features.append(features_bow.toarray())
    #features_bow_train = np.array(features_bow[:L].toarray())
    #features_bow_test = np.array(features_bow[L:].toarray())
    
    if ent:
        #features_ent_train = np.array(features_ent[:L])
        #features_ent_test = np.array(features_ent[L:])
        all_features.append(features_ent)
    
    if sta:
        #features_ent_train = np.array(features_ent[:L])
        #features_ent_test = np.array(features_ent[L:])
        all_features.append(features_sta)
    
    
    if senti:
        #features_senti_train = np.array(features_senti[:L])
        #features_senti_test = np.array(features_senti[L:])
        all_features.append(features_senti)
    
    dataset = np.concatenate(all_features,axis=1)
    train_dataset = dataset[:L]
    test_dataset = dataset[L:]
    
    best_params = svc_param_selection(train_dataset,y_train,nfolds=5)
    print(best_params)
    if best_params['kernel'] == 'rbf':
        model = svm.SVC(kernel='rbf' ,C = best_params['C'], gamma = best_params['gamma'],probability=True)
    else:
        model = svm.SVC(kernel='linear' ,C = best_params['C'],probability=True)
    
    
    model.fit(train_dataset,y_train)
    
    y_pred = model.predict(test_dataset)
    print(classification_report(y_test,y_pred,labels=['support','oppose'],target_names=['support','oppose']))
    #cm = confusion_matrix(y_test,y_pred,labels=['0','1','2'])
    conf_score = model.predict_proba(test_dataset)

    #print
    df = pd.DataFrame(np.concatenate([np.array(sentences[L:]).reshape(-1,1),np.array(y_pred).reshape(-1,1),np.array(conf_score)],axis=1))
    df.to_csv(save_file+".csv",header=False,index=False)
    return y_pred,y_test
                


# In[22]:

'''
#BOW
print ('BOW---------')
y_pred,y_test = [],[]
for dataset in ['AT','CC','FM','LA','HC']:
    a,b = train(dataset,bow = True)
    y_pred.extend(a)
    y_test.extend(b)
    print(len(a),len(b))
print(classification_report(y_test,y_pred,labels=['support','oppose'],target_names=['support','oppose']))'''


# In[23]:

print ("STA---------")
y_pred,y_test = [],[]
for dataset in ['AT','CC','FM','LA','HC']:
    a,b = train(dataset,sta=True)
    y_pred.extend(a)
    y_test.extend(b)
    print(len(a),len(b))
    
print(classification_report(y_test,y_pred,labels=['support','oppose'],target_names=['support','oppose']))


# In[7]:

print("ENT----------")
y_pred,y_test = [],[]
for dataset in ['AT','CC','FM','LA','HC']:
    a,b = train(dataset,ent=True)
    y_pred.extend(a)
    y_test.extend(b)
    print(len(a),len(b))
print(classification_report(y_test,y_pred,labels=['support','oppose'],target_names=['support','oppose']))


# In[8]:

print("SENTI---------")
y_pred,y_test = [],[]
for dataset in ['AT','CC','FM','LA','HC']:
    a,b = train(dataset,senti=True)
    y_pred.extend(a)
    y_test.extend(b)
    print(len(a),len(b))
print(classification_report(y_test,y_pred,labels=['support','oppose'],target_names=['support','oppose']))


# In[9]:

print("ENT_SENTI--------")
y_pred,y_test = [],[]
for dataset in ['AT','CC','FM','LA','HC']:
    a,b = train(dataset,ent=True,senti=True,sta=False)
    y_pred.extend(a)
    y_test.extend(b)
    print(len(a),len(b))
print(classification_report(y_test,y_pred,labels=['support','oppose'],target_names=['support','oppose']))


# In[10]:

print("ENT_STA---------")
y_pred,y_test = [],[]
for dataset in ['AT','CC','FM','LA','HC']:
    a,b = train(dataset,ent=True,senti=False,sta=True)
    y_pred.extend(a)
    y_test.extend(b)
    print(len(a),len(b))
print(classification_report(y_test,y_pred,labels=['support','oppose'],target_names=['support','oppose']))


# In[11]:

print("SENTI_STA-----------")
y_pred,y_test = [],[]
for dataset in ['AT','CC','FM','LA','HC']:
    a,b = train(dataset,ent=False,senti=True,sta=True)
    y_pred.extend(a)
    y_test.extend(b)
    print(len(a),len(b))
print(classification_report(y_test,y_pred,labels=['support','oppose'],target_names=['support','oppose']))


# In[12]:

print("ENT_SENTI_STA---------")
y_pred,y_test = [],[]
for dataset in ['AT','CC','FM','LA','HC']:
    a,b = train(dataset,ent=True,senti=True,sta=True)
    y_pred.extend(a)
    y_test.extend(b)
    print(len(a),len(b))
print(classification_report(y_test,y_pred,labels=['support','oppose'],target_names=['support','oppose']))


# In[ ]:




