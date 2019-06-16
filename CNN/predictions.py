import os
import glob
import sys
import csv
dir = glob.glob('./Data/*')

f = {}
for root, directories, filenames in os.walk('./Data/'):
    for directory in directories:             
        f[os.path.join(root, directory)] = {}
    for filename in filenames:
        f[root][os.path.splitext(filename)[0]] = os.path.join(root,filename)

for key in f.keys():
    data = []
    cat = key.replace("./Data/","")
    gf = f[key]['test_clean']
    pf = key+'/predict_'+cat+'.txt'
    with open(gf,'r') as rf:
        lines = rf.readlines()
        for line in lines:
            row = line.split('\t')
            tweet = row[2]
            if tweet == "tweet":
                continue
            data.append([tweet])
    with open(pf,'r') as rf:
        lines = rf.readlines()
        for i,line in enumerate(lines):
            label = line.rstrip()
            data[i].append(label)
    with open("./Predictions/predict_"+cat+".csv","w") as wf:
        csvw = csv.writer(wf)
        csvw.writerow(['tweet','predictions'])
        csvw.writerows(data)
    
