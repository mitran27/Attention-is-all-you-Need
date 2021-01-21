# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 15:56:35 2021

@author: mitran
"""
import numpy as np
from Encoder import ENCODER
from Components.Attention import Linear,Softmax
from Components.Embedding import Embedding
from tqdm import tqdm


def preprocess_data(data,pos,tag,max_len=100):
   """
   Input :sentence
         part-o-speech
         tag
   Output : Convert part-o-speech to ont encode
           Convert tag to one encode
     """
   pos_sent=data[1]
  
   o_hot_Sent_pos=[]
   for p in pos_sent:
      
     
       o_hot_Sent_pos.append(pos[p])
     
   tag_sent=data[2]
  
   o_hot_Sent_tag=[]
   for t in tag_sent:
     
      o_hot_Sent_tag.append(tag[t])

   return data[0][:max_len],o_hot_Sent_pos[:max_len],o_hot_Sent_tag[:max_len]
       
class Entity_Data_generator():
   def __init__(self,text,pos,tag,voc2id):
       
       self.text=text
       self.pos=pos
       self.tag=tag
       self.Tokenzer=voc2id
       self.max_len=160
   def __len__(self):
       return len(self.text)
       
   def __getitem__(self,item_no):
      
       text=self.text[item_no]
       pos=self.pos[item_no]
       tag=self.tag[item_no]
       if(not(len(text)==len(pos) and len(pos)==len(tag) and len(text)==len(tag))):
           pass
       
       # tokenizing for bert
       input_ids=[]# the id's will pass through the lookup table and the vectors of the word will go through the model
       target_pos=[]
       target_tag=[]
       
       for w,word in enumerate(text):
           # tokeizes word
           inp_W=(self.Tokenzer[word])
           # words nt in vocab will be splitted
           input_ids.append(inp_W) # splitted unknown words are considered as a seperate wprds of the sentence
           target_pos.append(pos[w])# all the spllited word will have same output and other embedding tokens of the word (sent[i]) before split
           target_tag.append(tag[w]) # same as pos
           
           # adding [cls] tokens
          
       input_ids=[self.Tokenzer["<cls>"]]+input_ids+[self.Tokenzer["<sep>"]]
       target_pos=[0]+target_pos+[0]
       target_tag=[0]+target_tag+[0]
      
        
       
       # padding
           
       """    
       pad_len=(self.max_len-len(input_ids))
       
       input_ids=input_ids+ ([0] * pad_len)
       target_pos=target_pos+([0]* pad_len)
       target_tag=target_tag+([0]* pad_len)
       """
       return {
               "input_samp":np.array((input_ids)),              
               "output_pos":np.array((target_pos)),
               "output_tag":np.array((target_tag)),               
                            
               }
       
        
        

from Encoder import ENCODER
import json

file =open("data_file.json","r").read() 
data=json.loads(file)
vocabulary={ i for sent in data for i in sent[0]}
vocabulary.add("<cls>")
vocabulary.add("<sep>")
posset={ i for sent in data for i in sent[1]}
tagset={ i for sent in data for i in sent[2]}

pos2id={p:i+1 for i,p in enumerate(posset)}
tag2id={t:i+1 for i,t in enumerate(tagset)}
voc2id={v:i for i,v in enumerate(vocabulary)}
id2voc={i:v for i,v in enumerate(vocabulary)}
pos2id['None']=0
tag2id['None']=0
posset.add("None")
tagset.add("None")

print(len(pos2id))


id2pos={i+1:p for i,p in enumerate(posset)}
id2tag={i+1:t for i,t in enumerate(tagset)}
id2tag[0]='None'
id2pos[0]='None'





processed_samples_text=[]

processed_samples_pos=[]

processed_samples_tag=[]

# appedning thw word id of the pos id of the tag
for dat in data:
         txt,pos,tag=preprocess_data(dat,pos2id,tag2id,60)
         processed_samples_text.append(txt)
         processed_samples_pos.append(pos)
         processed_samples_tag.append(tag)
         
Data_generator=Entity_Data_generator(processed_samples_text,processed_samples_pos,processed_samples_tag,voc2id)
Data_generator[1]
class ohe():
    def __init__(self,vocab):
        self.v=vocab
    def forward(self,inp):
        x=list(inp.shape)
        
        x.append(self.v)
        z=np.zeros((x))
        for i in range(inp.shape[0]):
            z[i,inp[i]]=1
        return z    
"""        
class mod():
    def __init__(self,inp,out):
        self.a1=Embedding(inp,out)  
        self.a2=Encoder_block(out,4)
    def forward(self,x):
        y=self.a1.forward(x)
        a=np.sum(y.reshape(-1))
        y=self.a2.forward(y,0)
        b=np.sum(y.reshape(-1))
        return y
    def backward(self,err,lr):
        err=self.a2.backward(err,0,lr)
        self.a1.backward(err,lr)
"""        
    
        
ld=256       
hd=6
Model=ENCODER(4,len(vocabulary),ld,hd) 
posnn=Linear(ld,len(posset))
tagnn=Linear(ld,len(tagset))


def sig(x):
    return 1/(1+np.exp(-x))
def der_sig(x):
    return sig(x)*(1-sig(x))

soft=Softmax()
lr=1e-3
n_samp=100
"""

for epc in range(3000):
     finloss=0
     for i in range(n_samp):
        
         data=Data_generator[i]
         inp=data["input_samp"].reshape(1,-1)
         tarpos=data["output_pos"].reshape(1,-1)
         tartag=data["output_tag"].reshape(1,-1)
         
         #print(inp)
         op=Model.forward(inp)
         posop=posnn.forward(op,0)
         tagop=tagnn.forward(op,0)
         
       
         posop=soft(posop)
         tagop=soft(tagop)
       
    
    
         err_pos=posop.copy()
         
         seqlen=err_pos.shape[1]
         for s in range(seqlen):
             err_pos[0,s,tarpos[0,s]]-=1
    
         err_tag=tagop.copy()
         
         for s in range(seqlen):
             err_tag[0,s,tartag[0,s]]-=1
         
         loss_pos=np.sum(err_pos.reshape(-1)**2)
         loss_tag=np.sum(err_tag.reshape(-1)**2)    
        
             
         loss=loss_pos+loss_tag
         
         
         finloss+=loss           
      
         
         der_pos=posnn.backward(err_pos,0,lr)
         der_tag=tagnn.backward(err_tag,0,lr)
         
         
         
         
         
         enc_err=der_pos+der_tag
        
        
         
         Model.backward(enc_err,lr)
        
             
             
         
     print(epc,finloss/n_samp)
     
     Model.zero_grad()
"""

                 
def avg(x):
  return  sum(x)/len(x)           
                     
import datetime
t1 = datetime.datetime.now()

                     
for epc in range(70):
     finloss=[]
     finloss1=[]
     c=0
     t=tqdm(Data_generator,total=n_samp,position=0, leave=True)
     for data in t:
        
         
         inp=data["input_samp"].reshape(1,-1)
         tarpos=data["output_pos"].reshape(1,-1)
         tartag=data["output_tag"].reshape(1,-1)
         
         #print(inp)
         op=Model.forward(inp)
         posop=posnn.forward(op,0)
         tagop=tagnn.forward(op,0)
         
       
         posop=soft(posop)
         tagop=soft(tagop)
       
    
    
         err_pos=posop.copy()
         
         seqlen=err_pos.shape[1]
         for s in range(seqlen):
             err_pos[0,s,tarpos[0,s]]-=1
    
         err_tag=tagop.copy()
         
         for s in range(seqlen):
             err_tag[0,s,tartag[0,s]]-=1
         
         loss_pos=np.sum(err_pos.reshape(-1)**2)#/len(posset)
         loss_tag=np.sum(err_tag.reshape(-1)**2) #
        
             
         loss=loss_pos+loss_tag
         loss1=(loss_pos+loss_tag)/seqlen
         finloss.append(loss)
         finloss1.append(loss1)
         string="epoch  "+str(epc)+" loss: {:.4f}".format(avg(finloss))+"  {:.4f}".format(avg(finloss1)) 
         t.set_description(string)
         t.refresh() 
         
         
         
         der_pos=posnn.backward(err_pos,0,lr)
         der_tag=tagnn.backward(err_tag,0,lr)
         
         
         
         
         
         enc_err=der_pos+der_tag
        
        
         
         Model.backward(enc_err,lr)
         if(c==n_samp):
             break
         c+=1
     
     
     Model.zero_grad()     
     
     
t2 = datetime.datetime.now()
print(t2 - t1)  
data=Data_generator[73]
inp=data["input_samp"].reshape(1,-1)
tarpos=data["output_pos"].reshape(1,-1)
tartag=data["output_tag"].reshape(1,-1)
         
         #print(inp)
op=Model.forward(inp)
posop=posnn.forward(op,0)
tagop=tagnn.forward(op,0)
         
       
posop=soft(posop).argmax(axis=-1)
tagop=soft(tagop).argmax(axis=-1) # softmax convert to probabilities argmax take the entity with highest probability 
print(posop)
print([0]+processed_samples_pos[73]+[0])             