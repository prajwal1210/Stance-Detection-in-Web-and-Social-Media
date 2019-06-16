import sys
import os

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
    print (tot)
    return f_avg


gold_f = sys.argv[1]
predictions_f = sys.argv[2]

true_y = []
with open(gold_f,"r") as f:
    lines = f.readlines()
    for line in lines:
        row = line.split('\t')
        cat = row[3].rstrip()
	    #print (cat)
        if cat == "Stance":
            continue
        elif cat == "AGAINST":
            cat = 1
        elif cat == "FAVOR":
            cat = 0
        elif cat == "NONE":
            cat = 2
        true_y.append(cat)


pred_y = []
with open(predictions_f,"r") as f:
    lines = f.readlines()
    for line in lines:
        cat = line.rstrip()
	#print (cat)
        if cat == "AGAINST":
            cat = 1
        elif cat == "FAVOR":
            cat = 0
        elif cat == "NONE":
            cat = 2
        pred_y.append(cat)
#print(len(pred_y))
print (score(true_y,pred_y))
