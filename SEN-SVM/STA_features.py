#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import time
import numpy as np
import pandas as pd
import string
import csv
from scipy import stats
import random
import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import wordninja
from collections import defaultdict, Counter
import math
import sys


# In[ ]:


def load_glove_embeddings_set():
    word2emb = []
    WORD2VEC_MODEL = "glove.6B.300d.txt"
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


# In[ ]:


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
            #print(word," normalised to ",norm_dict[word])
        else:
            normalised_tokens.append(word.lower())
    wordninja_tokens = []
    for word in normalised_tokens:
        if word in word_dict:
            wordninja_tokens+=[word]
        else:
            wordninja_tokens+=wordninja.split(word)
    return " ".join(wordninja_tokens)


# In[ ]:





# In[13]:




def build_lexicon(name):
    def pmi(x,y,z,t):
    	res=(x/(y*(z/t)+(math.sqrt(x)*math.sqrt(math.log(0.9)/(-2)))))
    	return math.log(res,2)
        

    
    def prob(word1,nava,total):
        count_prob=0
        if word1 in nava:
            count_prob += nava[word1]
        return((count_prob+1))

    def prob_cond(word1,seed,stance_seed,stance,total):
        count_prob=0
        for i in range(len(seed)):
            if(seed[i]==word1):
                if(stance_seed[i]==stance):
                    count_prob=count_prob+1
        return((count_prob+1))


    def prob_cond1(word1,word2,Features,total):
        count_prob=0
        #for i in range(length_Features):
        #    flag1=0
        #    flag2=0
        #    for word in Features['co_relation'][i]:
        #        if(word==word1):
        #            flag1=1
        #        if(word==word2):
        #            flag2=1
        #    if(flag1==1 and flag2==1):
        #            count_prob=count_prob+1
        #seed and non-seed lexicon formation       
        return((co_relation[(word1,word2)]+1))

    print("building lexicon for ", name)
    raw=pd.read_csv('./MPHI_Preprocessed/'+name+'/train.csv')

    #Features Extraction
    porter=PorterStemmer()

    Stop_words=set(stopwords.words('english'))
    Features=raw[['sentence']]
    Tweet=Features['sentence'].copy()

    Features['sentence']=Tweet.apply(sent_process)
    Features['tokenized_sents'] = Features.apply(lambda row: (row['sentence'].split()), axis=1)
    Features['pos_tag']=Features.apply(lambda row:nltk.pos_tag(row['tokenized_sents'],tagset='universal'),axis=1)
    Features['stance']=raw['stance']
    length_Features=len(Features['sentence'])
    
    co_relation=defaultdict(int)
    co_relation2 = []
    for i in range(length_Features):
        line=[]
        for word,tag in Features['pos_tag'][i]:
            if(tag=='NOUN' or tag=='ADJ' or tag=='VERB' or tag=='ADV'):
                if(word not in Stop_words):
                    line.append(porter.stem(word))
        for i in range(len(line)):
            for j in range(i+1,len(line)):
                co_relation[(line[i],line[j])]+=1
                co_relation[(line[j],line[i])]+=1
        co_relation2.append(line)

    Features['co_relation']=co_relation2

    FAVOR=[]
    AGAINST=[]
    NONE=[]
    for i in range(length_Features):
        if(Features['stance'][i]=='support'):
            for word,tag in Features['pos_tag'][i]:
                if(tag=='NOUN' or tag=='ADJ' or tag=='VERB' or tag=='ADV'):
                    if(word not in Stop_words):
                        FAVOR.append(porter.stem(word))
        else:
            if(Features['stance'][i]=='oppose'):
                for word,tag in Features['pos_tag'][i]:
                    if(tag=='NOUN' or tag=='ADJ' or tag=='VERB' or tag=='ADV'):
                        if(word not in Stop_words):
                            AGAINST.append(porter.stem(word))
            else:
                if(Features['stance'][i]=='neutral'):
                    for word,tag in Features['pos_tag'][i]:
                        if(tag=='NOUN' or tag=='ADJ' or tag=='VERB' or tag=='ADV'):
                            if(word not in Stop_words):
                                NONE.append(porter.stem(word))

    len_sup=len(FAVOR)
    len_opp=len(AGAINST)
    len_nut=len(NONE)

    len_co=[]
    for i in range(length_Features):
        len_co.append(len(Features['co_relation'][i]))

    Features['len_nava']=len_co

    nava=[]
    for i in range(length_Features):
        for word,tag in Features['pos_tag'][i]:
            if(tag=='NOUN' or tag=='ADJ' or tag=='VERB' or tag=='ADV'):
                if(word not in Stop_words):
                    nava.append(word.lower())
    nava_stem=[]
    for word in nava:
        nava_stem.append(porter.stem(word))
    uni_nava_stem=list(set(nava_stem))
    nava_stem = Counter(nava_stem)


    total=len(nava_stem)
    length=len(uni_nava_stem)

    print(total,length)

    seed=[]
    non_seed=[]
    seed_stance=[]
    for i in range(len(Features)):
        for j in range(int(0.75*Features['len_nava'][i])):
            seed.append(Features['co_relation'][i][j])
            seed_stance.append(Features['stance'][i])
        for j in range(int(0.75*Features['len_nava'][i]),Features['len_nava'][i]):
            non_seed.append(Features['co_relation'][i][j])
    uni_seed=list(set(seed))
    uni_non_seed=list(set(non_seed))

    '''for i in range(len(Features)):
        x=[]
        x=random.sample(Features['co_relation'][i],int(0.75*Features['len_nava'][i]))
        for j in range(len(x)):
            seed.append(x[j])
            seed_stance.append(Features['stance'][i])
        for j in range(Features['len_nava'][i]):
            if(Features['co_relation'][i][j] not in x):
                non_seed.append(Features['co_relation'][i][j])
    uni_seed=list(set(seed))
    uni_non_seed=list(set(non_seed))'''

    len_seed=len(seed)
    len_uni_seed=len(uni_seed)
    len_non_seed=len(non_seed)
    len_uni_non_seed=len(uni_non_seed)

    len_seed_sup=0
    len_seed_opp=0
    len_seed_nut=0
    for i in range(len(seed_stance)):
        if(seed_stance[i]=='support'):
            len_seed_sup=len_seed_sup+1
        else:
            if(seed_stance[i]=='oppose'):
                len_seed_opp=len_seed_opp+1
            else:
                len_seed_nut=len_seed_nut+1
    print(len_seed_nut,len_seed_opp,len_seed_sup)

    


    prob_sup=len_seed_sup/(len_seed_sup+len_seed_opp+len_seed_nut)
    prob_opp=len_seed_opp/(len_seed_sup+len_seed_opp+len_seed_nut)
    prob_nut=len_seed_nut/(len_seed_sup+len_seed_opp+len_seed_nut)

    prob_word=[]
    for word in uni_seed:
        prob_word.append(prob(word,nava_stem,total))

    prob_cond_word={}
    prob_supp_word=[]
    prob_opp_word=[]
    prob_neu_word=[]

    for word in uni_seed:
        prob_supp_word.append(prob_cond(word,seed,seed_stance,'support',(len_seed_sup+len_seed_opp+len_seed_nut)))
        prob_opp_word.append(prob_cond(word,seed,seed_stance,'oppose',(len_seed_sup+len_seed_opp+len_seed_nut)))
        prob_neu_word.append(prob_cond(word,seed,seed_stance,'neutral',(len_seed_sup+len_seed_opp+len_seed_nut)))

    prob_cond_word={'word':list(uni_seed),'prob_word':prob_word,'prob_supp_word':prob_supp_word,'prob_opp_word':prob_opp_word,'prob_neu_word':prob_neu_word}
    Seed_lexicon = pd.DataFrame(data=prob_cond_word)



    print(Seed_lexicon)

    pmi_AGAINST=[]
    pmi_FAVOR=[]
    pmi_NONE=[]
    '''for i in range(len_uni_seed):
        pmi_AGAINST.append(pmi(prob_opp_word[i],prob_word[i],prob_opp))
        pmi_FAVOR.append(pmi(prob_supp_word[i],prob_word[i],prob_sup))
        pmi_NONE.append(pmi(prob_neu_word[i],prob_word[i],prob_nut))'''

    for i in range(len_uni_seed):
        pmi_AGAINST.append(pmi(prob_opp_word[i],prob_word[i],len_seed_opp,len_seed))
        pmi_FAVOR.append(pmi(prob_supp_word[i],prob_word[i],len_seed_sup,len_seed))
        pmi_NONE.append(pmi(prob_neu_word[i],prob_word[i],len_seed_nut,len_seed))


    Seed_lexicon['pmi_AGAINST']=list(pmi_AGAINST)
    Seed_lexicon['pmi_FAVOR']=list(pmi_FAVOR)
    Seed_lexicon['pmi_NONE']=list(pmi_NONE)

    stance=[]
    for i in range(len_uni_seed):
        if((Seed_lexicon['pmi_FAVOR'][i] > Seed_lexicon['pmi_AGAINST'][i]) and (Seed_lexicon['pmi_FAVOR'][i] > Seed_lexicon['pmi_NONE'][i])):
            stance.append('support')
        else:
            if((Seed_lexicon['pmi_AGAINST'][i] > Seed_lexicon['pmi_FAVOR'][i]) & (Seed_lexicon['pmi_AGAINST'][i] > Seed_lexicon['pmi_NONE'][i])):
                stance.append('oppose')
            else:
                stance.append('neutral')

    Seed_lexicon['Stance']=list(stance)

        #NON SEED LEXICON
    score_non_seed_opp=[]
    score_non_seed_sup=[]
    score_non_seed_nut=[]

    opp_seed_word=[]
    nut_seed_word=[]
    sup_seed_word=[]
    for i in range(len_uni_seed):
            if(Seed_lexicon['Stance'][i]=='support'):
                sup_seed_word.append(Seed_lexicon['word'][i])
            else:
                if(Seed_lexicon['Stance'][i]=='oppose'):
                    opp_seed_word.append(Seed_lexicon['word'][i])
                else:
                    nut_seed_word.append(Seed_lexicon['word'][i])

    #opp_seed_word=set(opp_seed_word)
    #nut_seed_word=set(nut_seed_word)
    #sup_seed_word=set(sup_seed_word)

    len_opp_words=len(opp_seed_word)
    len_nut_words=len(nut_seed_word)
    len_sup_words=len(sup_seed_word)

    pmi_non_seed={}

    start1=time.time()
    print("COMPUTING...")
    k=0
    for word in uni_non_seed:
        list_=[]
        for i in range(len_sup_words):
            l=pmi(prob_cond1(word,sup_seed_word[i],Features,total),prob(word,nava_stem,total),prob(sup_seed_word[i],nava_stem,total),total)
            if(l<0):
                list_.append(1)
            else:
                list_.append(l)
        score_non_seed_sup.append(stats.gmean(list_))
        #print(k)
        k=k+1
    print("score_non_seed_sup_complete :)")
    end1=time.time()
    time1=end1-start1
    print(time1)

    start2=time.time()
    k=0
    for word in uni_non_seed:
        list_=[]
        for i in range(len_opp_words):        
            l=pmi(prob_cond1(word,opp_seed_word[i],Features,total),prob(word,nava_stem,total),prob(opp_seed_word[i],nava_stem,total),total)
            if(l<0):
                list_.append(1)
            else:
                list_.append(l)
        score_non_seed_opp.append(stats.gmean(list_))
        #print(k)
        k=k+1
    print("score_non_seed_opp_complete :)")
    end2=time.time() 
    time2=end2-start2
    print(time2)

    start3=time.time()
    k=0

    #print("~~~~",nut_seed_word)
    print(len(uni_non_seed),len_nut_words)
    for word in uni_non_seed:
        list_=[]
        #s2 = time.time()
        for i in range(len_nut_words):
            #s1 = time.time()
            l=pmi(prob_cond1(word,nut_seed_word[i],Features,total),                  prob(word,nava_stem,total),                  prob(nut_seed_word[i],nava_stem,total),total)
            #print(time.time()-s1)
            if(l<0):
                list_.append(1)
            else:
                list_.append(l)
        score_non_seed_nut.append(stats.gmean(list_))
        #print(time.time()-s2)
        #print(k)
        k=k+1
    print("score_non_seed_nut_complete :)")   
    end3=time.time()
    print("Process Complete :)")
    time3=end3-start3
    print(time3)

    total_time=time1+time2+time3
    print(total_time)

    prob_cond_word={'word':list(uni_non_seed),'score_non_seed_opp':score_non_seed_opp,'score_non_seed_sup':score_non_seed_sup,'score_non_seed_nut':score_non_seed_nut}
    NonSeed_lexicon = pd.DataFrame(data=prob_cond_word)

    #Tweet Vector Formation
    lex_word=[]
    lex_word.extend(list(Seed_lexicon['word']))
    lex_word.extend(list(NonSeed_lexicon['word']))

    pmi_sup=[]
    pmi_sup.extend(list(Seed_lexicon['pmi_FAVOR']))
    pmi_sup.extend(list(NonSeed_lexicon['score_non_seed_sup']))

    pmi_opp=[]
    pmi_opp.extend(list(Seed_lexicon['pmi_AGAINST']))
    pmi_opp.extend(list(NonSeed_lexicon['score_non_seed_opp']))

    pmi_nut=[]
    pmi_nut.extend(list(Seed_lexicon['pmi_NONE']))
    pmi_nut.extend(list(NonSeed_lexicon['score_non_seed_nut']))

    Lexicon = dict()
    for i in range(len(lex_word)):
        Lexicon[lex_word[i]] = {'pmi_sup':pmi_sup[i],'pmi_opp':pmi_opp[i],'pmi_nut':pmi_nut[i]}

    print("Lexicon formed")
    return Lexicon

    #Lexicon={'word':lex_word,'pmi_sup':pmi_sup,'pmi_opp':pmi_opp,'pmi_nut':pmi_nut}
    #Lexicon = pd.DataFrame(data=Lexicon)


    
    
    


# In[14]:


#Lexicon = build_lexicon('SC')


# In[26]:


def produce_features(name,Lexicon):
    #train_features
    for l in ['train','test']:
        raw=pd.read_csv('./MPHI_Preprocessed/'+name+'/{}.csv'.format(l))
        Stop_words=set(stopwords.words('english'))
        Features=raw[['sentence']]
        Tweet=Features['sentence'].copy()

        Features['preprocessed_sentence']=Tweet.apply(sent_process)
        Features['tokenized_sents'] = Features.apply(lambda row: (row['preprocessed_sentence'].split()), axis=1)

        porter = PorterStemmer()
        start=time.time()
        #word_sup_vect=[]
        #word_opp_vect=[]
        #word_nut_vect=[]

        data = [['sentence','pmi_sup','pmi_opp','pmi_nut']]
        len_lexicon_word=len(Lexicon)

        for i in range(len(Features['sentence'])):
            sum1=0
            sum2=0
            sum3=0
            total_lex=0
            temp = []
            for word in Features['tokenized_sents'][i]:
                #for j in range(len_lexicon_word):

                w = porter.stem(word)
                if w in Lexicon:
                    sum1=sum1+Lexicon[w]['pmi_sup']
                    sum2=sum2+Lexicon[w]['pmi_opp']
                    sum3=sum3+Lexicon[w]['pmi_nut']
                    total_lex=total_lex+1
            #word_sup_vect.append(sum1/total_lex)
            #word_opp_vect.append(sum2/total_lex)
            #word_nut_vect.append(sum3/total_lex)
            data.append([Features['sentence'][i],sum1/total_lex,sum2/total_lex,sum3/total_lex])

        my_df = pd.DataFrame(data)
        my_df.to_csv('./pmi/pmi_{}_{}.csv'.format(name,l+'1'),header=False,index=False)


    end=time.time()
    print(end-start)


# In[27]:
produce_features('HRT',build_lexicon('HRT'))

'''for dataset in ['AT','LA','CC','HC','FM']:
    produce_features(dataset,build_lexicon(dataset))'''


# In[ ]:





# In[ ]:




