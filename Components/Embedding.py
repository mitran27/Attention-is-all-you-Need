# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 14:38:16 2021

@author: mitran
"""

import numpy as np

class Embedding():
    def __init__(self,voc_size,embedding_size):
        
        self.emb_mat=(np.random.rand(voc_size, embedding_size) - 0.5)*(1./np.sqrt(embedding_size))
        self.dim=embedding_size
        self.voc_Sze=voc_size
       
    def forward(self,dense):
        
        self.input = dense          
        sha=list(dense.shape)
        sha.append(self.dim)
        self.output=self.emb_mat[dense.reshape(1,-1)].reshape(sha)        
        return self.output
        
        
    def backward(self, output_error, lr):
        
        input_error = np.dot(output_error, self.emb_mat.T)
        o_R=output_error.reshape(-1,self.dim)
        new=self.emb_mat.copy()       
        k=self.input.reshape(-1)        
        for i,ind in enumerate(k.T):
            new[ind]-=(o_R[i]*lr)  
        self.emb_mat=new
        return input_error
    
    
class Positional_encoding():
    def __init__(self,dim):
        self.d=dim
    def get_enc_vec(self,pos,i):
        d=self.d
        # according to th formula if it is 2i(even) 2i is used else 2i+1(odd) 2i is used (-1) is done
        k= i if i%2==0 else i-1
        if(i%2==0):
            return np.sin(pos/(10000**(k/d)))
        else: 
            return np.cos(pos/(10000**(k/d)))
        
    def create_encoding(self,seq_len):
        
        enc_mat=np.zeros((seq_len,self.d))# each word in the sentence will have a d dimensional vector according to its postion
        for pos in range(seq_len):
            for i in range(self.d):
                enc_mat[pos,i]=self.get_enc_vec(pos,i)
        #plt.imshow(enc_mat)
        return enc_mat
    
    def forward(self,seq_len):
        
        enc_mat =self.create_encoding(seq_len)
        
        return enc_mat    