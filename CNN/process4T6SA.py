# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import sys
import os

"""directories should be modified according to your directory """
File=[]
File.append(sys.argv[1])
filename = os.path.basename(sys.argv[1])
path = sys.argv[1].replace(filename,"")
Result_File=[path+'T6SA_stance.neg',
			path+'T6SA_stance.pos',
			path+'T6SA_stance.none',
			]

RES=[]
			
def process_str(topic,msg):
	msg=msg.replace("#SemST","")
	msg=msg.replace("\r\n","\n")
	msg=msg.lower()
	msg=msg+"\n"
	return msg
	
if __name__=="__main__":
    for i in range(1):
        File[i]=unicode(File[i],'utf8')
        j=0
        with open(File[i],"rb") as f:
            for line in f:
                print j
                j+=1
                tmp=line.split('\t')
                id=int(tmp[0])
                topic=''.join(tmp[1])
                msg=''.join(tmp[2])
                stance=''.join(tmp[3])
                print stance
                if stance=="AGAINST\n":
                    stance=0
                if stance=="FAVOR\n":
                    stance=1
                if stance=="NONE\n":
                    stance=2
                msg=process_str(topic,msg)
                datum = {"topic":topic,
						"stance":stance,
						"message":msg
						}
                if msg!="not available\n":
                    RES.append(datum)
	
    len_RES=len(RES)
    f_neg=open(unicode(Result_File[0],'utf8'),"wb")
    f_pos=open(unicode(Result_File[1],'utf8'),"wb")
    f_none=open(unicode(Result_File[2],'utf8'),"wb")
    for i in range (len_RES):
    	print RES[i]["message"]
        if RES[i]["stance"]==0:
        	print 0
        	f_neg.write(RES[i]["message"])
        if RES[i]["stance"]==1:
        	print 1
        	f_pos.write(RES[i]["message"])
        if RES[i]["stance"]==2:	
        	print 2
        	f_none.write(RES[i]["message"])

    f.close()
    f_neg.close()
    f_pos.close()
    f_none.close()
		
	
	
	
	
