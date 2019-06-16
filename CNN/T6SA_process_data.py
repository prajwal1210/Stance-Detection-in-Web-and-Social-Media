# -*- coding: utf-8 -*-
import numpy as np
import cPickle
from collections import defaultdict
import sys, re
import pandas as pd

MAX_SENTANCE_LEN = 100
EMBEDDING_DIMENSION = 300
def process_T6SA_data(line):
    tmp = line.split('\t')
    msg=''.join(tmp[2])
    msg=msg.replace("#SemST","")
    msg=msg.replace("\r\n","\n")
    msg=msg.lower()
    msg=msg+"\n"
    stance=''.join(tmp[3])
    if stance=="AGAINST\n":
        stance=0
    if stance=="FAVOR\n":
        stance=1
    if stance=="NONE\n":
        stance=2   
    if stance == "UNKNOWN\n":
        stance= 3
    if msg == "not available\n":
        msg = ""
    return [msg,stance]

def build_data_cv(data_folder, test_folder, cv=10, clean_string=True):
    """
    Loads data and split into 10 folds.
    """
    revs,revs_test = [],[]
    pos_file = data_folder[0]
    neg_file = data_folder[1]
    cen_file = data_folder[2]
    print pos_file
    print neg_file
    print cen_file
    vocab = defaultdict(float)
    """ data should be preprocessed into three files - pos, neg and none. """
    with open(pos_file, "rb") as f:
        for line in f:       
            rev = []
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum  = {"y":1, 
                      "text": orig_rev,                             
                      "num_words": len(orig_rev.split()),
                      "split": np.random.randint(0,cv)}
            revs.append(datum)
    with open(neg_file, "rb") as f:
        for line in f:       
            rev = []
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum  = {"y":0, 
                      "text": orig_rev,                             
                      "num_words": len(orig_rev.split()),
                      "split": np.random.randint(0,cv)}
            revs.append(datum)
    with open(cen_file, "rb") as f:
        for line in f:       
            rev = []
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum  = {"y":2, 
                      "text": orig_rev,                             
                      "num_words": len(orig_rev.split()),
                      "split": np.random.randint(0,cv)}
            revs.append(datum)
 
    ''' if the parameter test_folder is "null", 10% of the dataset will be used for test.
        Otherwise test set will be used, so that prediction can be performed.
    '''
    if test_folder!="null":
        print("Test Data loading....") 
        with open(test_folder, "r") as f:
            for line in f:
                if line.split('\t')[0] == "ID":
                    continue
                rev = []
                ''' the name of test_folder is specific ''' 
                if test_folder==test_folder:
                    msg,stance = process_T6SA_data(line)
                else:
                    revs_test = []
                    break
                if msg == "":
                    continue
                rev.append(msg.strip())
                if clean_string:
                    orig_rev = clean_str(" ".join(rev))
                else:
                    orig_rev = " ".join(rev).lower()
                words = set(orig_rev.split())
                for word in words:
                    vocab[word] += 1
                datum  = {"y":stance, 
                      "text": orig_rev,                             
                      "num_words": len(orig_rev.split()),
                      "split": np.random.randint(0,cv)}
                if datum["num_words"] > MAX_SENTANCE_LEN:
                    continue
                revs_test.append(datum)  
    return revs, revs_test, vocab
    
def get_W(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+1, k))            
    W[0] = np.zeros(k)
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map

def load_bin_vec(fname1, fname2, vocab):
    """
    2 word2vec files can be used together. fname2's word2vec will cover fname1's word2vec if they 
    have same words.
    If only one word2vec file is used, then fname2 is "null".
    """
    word_vecs = {}
    """Loads 300x1 word vecs from Google (Mikolov) word2vec"""
    with open(fname1, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)   
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
            else:
                f.read(binary_len)
    f.close()

    if fname2 != "null":	
        with open(fname2, "rb") as f:
            header = f.readline()
            vocab_size, layer1_size = map(int, header.split())
            binary_len = np.dtype('float32').itemsize * layer1_size
            for line in xrange(vocab_size):
                word = []
                while True:
                    ch = f.read(1)
                    if ch == ' ':
                        word = ''.join(word)
                        break
                    if ch != '\n':
                        word.append(ch)   
                if word in vocab:
                    word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
                else:
                    f.read(binary_len)
        f.close()
    return word_vecs

def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)  

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


if __name__=="__main__":    
    w2v_file1 = sys.argv[1]  
    w2v_file2 = sys.argv[2]	
    test_folder = sys.argv[3]
    path = sys.argv[4]
    data_folder = [path+"/T6SA_stance.pos",path+"/T6SA_stance.neg",path+"/T6SA_stance.none"]
    print test_folder
    print "loading data...",        
    revs, revs_test, vocab = build_data_cv(data_folder, test_folder, cv=10, clean_string=True)
    print pd.DataFrame(revs)
    max_l = np.max(pd.DataFrame(revs)["num_words"])
    print "data loaded!"
    print "number of train sentences: " + str(len(revs))
    print "number of out-test sentences: " + str(len(revs_test))
    print "vocab size: " + str(len(vocab))
    print "max sentence length: " + str(max_l)
    print "loading word2vec vectors...",
    w2v = load_bin_vec(w2v_file1, w2v_file2, vocab)
    print "word2vec loaded!"
    print "num words already in word2vec: " + str(len(w2v))
    add_unknown_words(w2v, vocab, k=EMBEDDING_DIMENSION)
    W, word_idx_map = get_W(w2v, k=EMBEDDING_DIMENSION)
    rand_vecs = {}
    add_unknown_words(rand_vecs, vocab,k=EMBEDDING_DIMENSION)
    W2, _ = get_W(rand_vecs,k=EMBEDDING_DIMENSION)

    cPickle.dump([revs, revs_test, W, W2, word_idx_map, vocab], open(path+"/mr.p", "wb"))
    print "dataset created!"
    
