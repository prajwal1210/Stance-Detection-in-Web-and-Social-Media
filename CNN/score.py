import os
import glob
import sys
import ast
import numpy as np

if len(sys.argv) < 2:
    print "Please enter the path to the data directory"

f = {}
for root, directories, filenames in os.walk(sys.argv[1]):
    print(root)
    for directory in directories:             
        f[os.path.join(root, directory)] = {}
    for filename in filenames:
        f[root][os.path.splitext(filename)[0]] = os.path.join(root,filename)

sem = ['AT','LA','HC','FM','CC']
mphi = ['EC','MMR','HRT','SC','VC']

tot =np.array([[0, 0, 0], [0, 0, 0]])

for key in f.keys():
    cat = key.replace(sys.argv[1],"")
    if cat not in sem:
        continue
    with open(f[key]['score'],'r') as sf:
        lines = sf.readlines()
        arr = np.array(ast.literal_eval(lines[0]))
        tot = tot + arr
        print ("%s -%s"%(cat, lines[1]))

r0 = tot[0][0]/(tot[0][2]+1e-5)
p0 = tot[0][0]/(tot[0][1]+1e-5)
r1 = tot[1][0]/(tot[1][2]+1e-5)
p1 = tot[1][0]/(tot[1][1]+1e-5)
f0 = 2*r0*p0/(r0+p0+1e-5)
f1 = 2*r1*p1/(r1+p1+1e-5)

f_avg = (f0+f1)/2
print f_avg

tot =np.array([[0, 0, 0], [0, 0, 0]])
for key in f.keys():
    cat = key.replace(sys.argv[1],"")
    if key not in mphi:
        continue
    with open(f[key]['score'],'r') as sf:
        lines = sf.readlines()
        arr = np.array(ast.literal_eval(lines[0]))
        tot = tot + arr
        print ("%s -%s"%(cat, lines[1]))

r0 = tot[0][0]/(tot[0][2]+1e-5)
p0 = tot[0][0]/(tot[0][1]+1e-5)
r1 = tot[1][0]/(tot[1][2]+1e-5)
p1 = tot[1][0]/(tot[1][1]+1e-5)
f0 = 2*r0*p0/(r0+p0+1e-5)
f1 = 2*r1*p1/(r1+p1+1e-5)

f_avg = (f0+f1)/2
print f_avg
