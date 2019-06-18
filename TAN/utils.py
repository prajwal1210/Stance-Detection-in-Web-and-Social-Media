import csv
import copy
import numpy as np
import re
import itertools
from collections import Counter,defaultdict
import torch
import json
from collections import Counter
import wordninja
"""

Tokenization/string cleaning for all datasets.
Every dataset is lower cased.
Original taken from https://github.com/dennybritz/cnn-text-classification-tf

string = re.sub(r"[^A-Za-z0-9(),!?\'\`#]", " ", string)
string = re.sub(r"#SemST", "", string)
string = re.sub(r"#([A-Za-z0-9]*)", r"# \1 #", string)
#string = re.sub(r"# ([A-Za-z0-9 ]*)([A-Z])(.*) #", r"# \1 \2\3 #", string)
string =  re.sub(r"([A-Z])", r" \1", string)
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

"""
def clean_str2(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets.
    Every dataset is lower cased.
    Original taken from https://github.com/dennybritz/cnn-text-classification-tf
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()



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


def create_normalise_dict(no_slang_data = "./noslang_data.json", emnlp_dict = "./emnlp_dict.txt"):
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

def normalise(normalization_dict,sentence):
    normalised_tokens = []
    word_tokens = sentence.split()
    for word in word_tokens:
        if word in normalization_dict:
        #if False:
            normalised_tokens.extend(normalization_dict[word].lower().split(" "))
            #print(word," normalised to ",normalization_dict[word])
        else:
            normalised_tokens.append(word.lower())
    #print(normalised_tokens)
    return normalised_tokens


def load_dataset(dataset,dev = "cuda"):
    def split(word):
        if word in word2emb:
        #if True:
            return [word]
        return wordninja.split(word)


    assert dataset in ['VC', 'HC', 'HRT', 'LA', 'CC', 'SC', 'EC', 'MMR', 'AT', 'FM'], "unknown dataset"

    folder = "Data_SemE_P"

    if dataset == 'EC':
        topic = 'E-ciggarettes are safer than normal ciggarettes'
        folder = "Data_MPCHI_P"
    elif dataset == 'SC':
        topic = 'Sun exposure can lead to skin cancer'
        folder = "Data_MPCHI_P"
    elif dataset == 'VC':
        topic = 'Vitamin C prevents common cold'
        folder = "Data_MPCHI_P"
    elif dataset == 'HRT':
        topic = 'Women should take HRT post menopause'
        folder = "Data_MPCHI_P"
    elif dataset == 'MMR':
        topic = 'MMR vaccine can cause autism'
        folder = "Data_MPCHI_P"
    elif dataset == 'AT' :
        topic = "atheism"
    elif dataset == 'HC' :
        topic = "hillary clinton"
    elif dataset == 'LA' :
        topic = "legalization of abortion"
    elif dataset == 'CC' :
        topic = "climate change is a real concern"
    elif dataset == 'FM' :
        topic = "feminist movement"
    elif dataset == 'VCA':
        topic = "vaccines cause autism"
    elif dataset == 'VTI':
        topic = "vaccines treat influenza"
    print(topic)

    normalization_dict = create_normalise_dict(no_slang_data = "./noslang_data.json", emnlp_dict = "./emnlp_dict.txt")

    target = normalise(normalization_dict,clean_str(topic))
    stances = {'FAVOR' : 0, 'AGAINST' : 1, 'NONE' : 2}

    train_x = []
    train_y = []

    with open("../Preprocessing/{}/{}/train_preprocessed.csv".format(folder,dataset),"r",encoding='latin-1') as f:
        reader = csv.DictReader(f, delimiter=',')
        for row in reader:
            if row['Stance'] in stances:
                train_x.append(row['Tweet'].split(' '))
                train_y.append(stances[row['Stance']])

    test_x = []
    test_y = []

    with open("../Preprocessing/{}/{}/test_preprocessed.csv".format(folder,dataset),"r",encoding='latin-1') as f:
        reader = csv.DictReader(f, delimiter=',')
        for row in reader:
            if row['Stance'] in stances:
                test_x.append(row['Tweet'].split(' '))
                test_y.append(stances[row['Stance']])


    word2emb = load_glove_embeddings()


    word_ind = {}

    # for i,sent in enumerate(train_x):
    #     final_sent = []
    #     j = 0
    #     while j < len(sent):
    #         final_sent += split(sent[j])
    #         j+=1
    #     train_x[i] = final_sent
    #
    # for i,sent in enumerate(test_x):
    #     final_sent = []
    #     j = 0
    #     while j < len(sent):
    #         final_sent += split(sent[j])
    #         j+=1
    #     test_x[i] = final_sent
    #

    for sent in train_x:
        for word in sent:
            if word not in word_ind and word in word2emb:
                word_ind[word] = len(word_ind)

    for sent in test_x:
        for word in sent:
            if word not in word_ind and word in word2emb:
                word_ind[word] = len(word_ind)

    for word in target:
        if word not in word_ind and word in word2emb:
            word_ind[word] = len(word_ind)



    UNK = len(word_ind)
    PAD = len(word_ind)+1


    ind_word = {v:k for k,v in word_ind.items()}


    print("Number of words - {}".format(len(ind_word)))


    # In[12]:



    # In[13]:

    #x_train = np.full((len(train_x),MAX_LEN),PAD)
    x_train = []
    OOV = 0
    oovs = []

    for i,sent in enumerate(train_x):
        temp = []
        for j,word in enumerate(sent):
            if word in word_ind:
                temp.append(word_ind[word])
            else:
                #print(word)
                temp.append(UNK)
                OOV+=1
                oovs.append(word)
        x_train.append(temp)

    print("OOV words :- ",OOV)
    a = Counter(oovs)
    print(a)

    # In[14]:

    y_train = np.array(train_y)
    y_test = np.array(test_y)


    # In[15]:

    x_test = []

    for i,sent in enumerate(test_x):
        temp = []
        for j,word in enumerate(sent):
            if word in word_ind:
                temp.append(word_ind[word])
            else:
                temp.append(UNK)

        x_test.append(temp)




    embedding_matrix = np.zeros((len(word_ind) + 2, 300))
    embedding_matrix[len(word_ind)] = np.random.randn((300))
    for word in word_ind:
        embedding_matrix[word_ind[word]] = word2emb[word]




    print("Number of training examples :- ",len(x_train))
    print("Sample vectorised sentence :- ",x_train[0])

    device = torch.device(dev)
    print("Using this device :- ", device)





    vector_target = []
    for w in target:
        if w in word_ind:
            vector_target.append(word_ind[w])
        else:
            vector_target.append(UNK)


    print("vectorised target:-")
    print(vector_target)

    return stances, word2emb, word_ind, ind_word, embedding_matrix, device,\
     x_train, y_train, x_test, y_test, vector_target, train_x, test_x






def load_glove_embeddings():
    word2emb = {}
    WORD2VEC_MODEL = "../Preprocessing/glove.6B.300d.txt"
    fglove = open(WORD2VEC_MODEL,"r")
    for line in fglove:
        cols = line.strip().split()
        word = cols[0]
        embedding = np.array(cols[1:],dtype="float32")
        word2emb[word]=embedding
    fglove.close()
    return word2emb
