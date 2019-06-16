import os
import glob
import sys

if len(sys.argv) < 2:
    print "Please enter the path to the data directory"

f = {}
for root, directories, filenames in os.walk(sys.argv[1]):
    for directory in directories:             
        f[os.path.join(root, directory)] = {}
    for filename in filenames:
        f[root][os.path.splitext(filename)[0]] = os.path.join(root,filename)
    

print f

for key in sorted(f.keys()):
    cat = key.replace(sys.argv[1],"")
    print(cat)
    os.system('python process4T6SA.py '+f[key]['train_clean'])
    os.system('python T6SA_process_data.py GoogleNews-vectors-negative300.bin null '+ f[key]['test_clean']+' '+key)
    os.system('python T6SA_conv_net_sentence.py -nonstatic -word2vec -predict '+key+'/predict_'+cat+'.txt ')
    # print ('python evaluate.py '+f[key]['test_clean']+' '+key+'/predict_'+cat+'.txt >'+key+'/score.txt')
    os.system('python evaluate.py '+f[key]['test_clean']+' '+key+'/predict_'+cat+'.txt >'+key+'/score.txt')
