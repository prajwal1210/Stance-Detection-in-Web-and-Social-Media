import re

f = open('lexicon_easy.csv', 'w')

# Read in the lexicon. Here's an example line:
#
# type=weaksubj len=1 word1=abandoned pos1=adj stemmed1=n priorpolarity=negative
#
# For now, just use a regular expression to grab the word and the priorpolarity parts.
with open('subjclueslen1-HLTEMNLP05.tff', 'r') as file:
    for line in file.readlines():
        # print (line)
        m = re.search('.*type=(\S+).*word1=(\S+).*priorpolarity=(\S+)', line)
        print(m.group(2))
        sub_score = 0
        senti_score = 0
        if m.group(3) == 'positive':
            senti_score = 1
            if m.group(1) == 'weaksubj':
                sub_score = 1
            elif m.group(1) == 'strongsubj':
                sub_score = 2
        elif m.group(3) == 'negative':
            senti_score = -1
            if m.group(1) == 'weaksubj':
                sub_score = -1
            elif m.group(1) == 'strongsubj':
                sub_score = -2
        f.write("%s,%d,%d\n" % (m.group(2), sub_score,senti_score))

f.close()