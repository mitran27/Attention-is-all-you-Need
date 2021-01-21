# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 14:27:49 2021

@author: mitran
"""
import numpy as np
from Components.Attention import MultiAttention,Linear
from Components.Normalization import LayerNormalization
from Components.Embedding import Embedding,Positional_encoding


class Encoder_block():
    def __init__(self,dim,heads,mask=False):
        self.Attn=MultiAttention(dim,heads,mask)
        self.Norm1=LayerNormalization()
        self.FFN=Linear(dim,dim)
        self.Norm2=LayerNormalization()
    
    def forward(self,x,ts):
        
        y=self.Attn.forward(x,ts)
        y=self.Norm1.forward(x+y)
        z=self.FFN.forward(y,ts)
        z=self.Norm2.forward(y+z)
        return z
    def backward(self,err,ts,lr):
        
        y=self.Norm2.backward(err,lr)
        z=self.FFN.backward(y,ts,lr)
        
        x=self.Norm1.backward(y+z,lr)
        y=self.Attn.backward(x,ts,lr)
        
        z=x+y
        
        return z
    
   
    
    def zero_grad(self):# after a epoch to remove gradients
        self.Attn.zero_grad()
        self.FFN.zero_grad()
    
class ENCODER():
    def __init__(self,no_blocks,vocab_size,dimension,heads,mask=False):
        
        self.embedding=Embedding(vocab_size,dimension)
        self.posenc=Positional_encoding(dimension)
        self.N=no_blocks
        self.block={}
        for i in range(no_blocks):
             
             self.block[i]=Encoder_block(dimension,heads,mask)
        
    def forward(self,inputs):
        
        
        emb=self.embedding.forward(inputs)
        pos=self.posenc.forward(inputs.shape[-1])
        
        x=emb+pos
        # encoder only one time step
        for i in range(self.N):
             x=self.block[i].forward(x,0)
       
        return x
    def backward(self,error,lr):
        err=error
        for i in range(self.N-1,-1,-1):
             err=self.block[i].backward(err,0,lr)        
        x=self.embedding.backward(err,lr)
       
        return 1
    def zero_grad(self):
        for i in range(self.N):
             self.block[i].zero_grad()    
      
