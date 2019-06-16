import os
import glob
import sys
import pickle as pkl

if len(sys.argv) < 2:
    print "Please enter the path to the data directory"

f = {}
for root, directories, filenames in os.walk(sys.argv[1]):
    for directory in directories:             
        f[os.path.join(root, directory)] = {}
    for filename in filenames:
        f[root][os.path.splitext(filename)[0]] = os.path.join(root,filename)

for key in sorted(f.keys()):
    cat = key.replace(sys.argv[1],"")
    print(cat)
    param_key = "predict_"+cat+"._hyperparams"
    param_file = f[key][param_key]
    params = pkl.load(open(param_file,'rb'))
    print params
    print "\n"

