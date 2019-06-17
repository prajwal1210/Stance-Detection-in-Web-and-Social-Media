#!/usr/bin/env python
# coding: utf-8
import json
import os
import glob
import sys
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk.stem import PorterStemmer,WordNetLemmatizer
import re
import wordninja
import numpy as np
import csv
import argparse
import os
import shutil
import copy

parser = argparse.ArgumentParser()
parser.add_argument('-nd','--non-dl',dest='nondl', action='store_true')
args = parser.parse_args()


wnl = WordNetLemmatizer()
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

#Creating Normalization Dictionary
with open("./noslang_data.json", "r") as f:
    data1 = json.load(f)

data2 = {}
with open("./emnlp_dict.txt","r") as f:
    lines = f.readlines()
    for line in lines:
        row = line.split('\t')
        data2[row[0]] = row[1].rstrip()

normalization_dict = {**data1,**data2}

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets.
    Every dataset is lower cased.
    Original taken from https://github.com/dennybritz/cnn-text-classification-tf
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`#]", " ", string)
    string = re.sub(r"#SemST", "", string)
    string = re.sub(r"#([A-Za-z0-9]*)", r"# \1 #", string)
    #string = re.sub(r"# ([A-Za-z0-9 ]*)([A-Z])(.*) #", r"# \1 \2\3 #", string)
    #string =  re.sub(r"([A-Z])", r" \1", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()


def split(word, word2emb):
    if word in word2emb:
        return [word]
    return wordninja.split(word)

def load_glove_embeddings():
    word2emb = {}
    WORD2VEC_MODEL = "./glove.6B.300d.txt"
    fglove = open(WORD2VEC_MODEL,"r")
    for line in fglove:
        cols = line.strip().split()
        word = cols[0]
        embedding = np.array(cols[1:],dtype="float32")
        word2emb[word]=embedding
    fglove.close()
    return word2emb

word2emb = load_glove_embeddings()
raw_folders = ['./Data_SemE','./Data_MPCHI']
processed_folders = ['./Data_SemE_P','./Data_MPCHI_P']

for folder in processed_folders:
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.mkdir(folder)

for dataset,new_folder in zip(raw_folders,processed_folders):
    f = {}
    print (dataset)
    for root, directories, filenames in os.walk(dataset):
        for directory in directories:             
            f[os.path.join(root, directory)] = {}
        for filename in filenames:
            f[root][os.path.splitext(filename)[0]] = os.path.join(root,filename)
    print (f)

    correct = 0
    
    for key in sorted(f.keys()):
        
        cat = key.replace(dataset,"")
        new_cat_folder = new_folder+"/"+cat
        os.mkdir(new_cat_folder)

        for k in ["train","test"]:
            n_count = 0
            s_words = 0
            new_lines = []
            old_lines = []
            with open(f[key][k],"r") as fp:
                lines = fp.readlines()
                for line in lines:
                    x = line.split("\t")
                    old_sent = copy.deepcopy(x[2])
                    old_lines.append(old_sent)
                    sent = clean_str(x[2])
                    word_tokens = sent.split(' ')

                    #Normalization
                    normalized_tokens = []
                    for word in word_tokens:
                        if word in normalization_dict.keys():
                            normalized_tokens.append(normalization_dict[word])
                            n_count+=1
                        else:
                            normalized_tokens.append(word)

                    #Word Ninja Splitting
                    normalized_tokens_s = []
                    for word in normalized_tokens:
                        normalized_tokens_s.extend(split(word,word2emb))

                    final_tokens = normalized_tokens_s
                    
                    if args.nondl == True:                  
                        #Stop Word Removal
                        filtered_tokens = []
                        for w in normalized_tokens_s:
                            if w not in stop_words:
                                filtered_tokens.append(w)
                            else:
                                s_words+=1

                        # Stemming using Porter Stemmer
                        stemmed_tokens = []
                        for w in filtered_tokens:
                            stemmed_tokens.append(ps.stem(w))
                        final_tokens = stemmed_tokens
                    
                    new_sent = ' '.join(final_tokens)
                    x[2] = new_sent
                    if (len(x) == 3):
                        if correct == 0:
                            x.append('NONE\n')
                            correct+=1
                        else:
                            x.append('FAVOR\n')
                    new_line = '\t'.join(x)
                    new_lines.append(new_line)
                    
                print ("%s %s- (%d,%d)"%(cat,k,n_count,s_words))
                
                #Write to a txt file
                with open(new_cat_folder+"/"+k+"_clean.txt","w") as wf:
                    wf.writelines(new_lines)
                with open(new_cat_folder+"/"+k+"_preprocessed.csv","w") as csvf:
                    writer = csv.writer(csvf)
                    writer.writerow(["Tweet", "Stance","Index","Original Tweet"])
                    for i,line in enumerate(new_lines):
                        try:
                            writer.writerow([line.split("\t")[2],line.split("\t")[3][:-1],int(line.split("\t")[0]),old_lines[i]])
                        except:
                            print (line.split('\t'))











