# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 14:28:43 2021

@author: mitran
"""
import numpy as np

class Linear():
    def __init__(self,input_dim,output_dim,B1=0.9,B2=0.99):
        self.Weight=(np.random.rand(input_dim, output_dim) - 0.5)*(2./np.sqrt(input_dim))
        self.Vdw=np.zeros((input_dim,output_dim))
        self.Sdw=np.zeros((input_dim,output_dim))
       
        self.der_W=0 # for accumulating weight for all time steps
        self.input={} # each compnent will store the inputs for each time step external model will not give lik e done for lstm
        self.ind=input_dim
        self.opd=output_dim
        self.B1=B1
        self.B2=B2
        self.eps=0.0001
    def forward(self,data,time_step):# data shape is batchsize,....,hidden layer
        self.input[time_step]=data
        output=np.dot(data,self.Weight)
        return output
    def zero_grad(self):
        
        self.Vdw*=0
   
       
        
    def update(self,lr):
        
        
        
        
        beta1=0.1
        self.Vdw=beta1*self.Vdw+(1-beta1)*self.der_W
                  
        m_correlated = self.Vdw / (1 - beta1)
        self.Weight -=lr * m_correlated 
          
        
        
        
        
        
        
        
        
        
        #self.Weight-=self.der_W*lr
       
        # after update accumulate der weight are zero
        self.der_W=0
        self.input={}
        
        
    def backward(self,err,time_Step,lr):
        # since derivative has to accumulated for all time steps
        ts=time_Step
        self.der_W+=(np.dot((self.input[ts]).reshape(-1,self.ind).T,err.reshape(-1,self.opd)) )
        # accumulating the weight of time steps  after 
        der_inp=np.dot(err,self.Weight.T) # since all time steps using same weight , while backpropagation also they should use the same weights so accumulation of derivatives is done seperately 
        
       
        if(ts==0):
            self.update(lr)
        return der_inp



       
        

     
      
class Softmax():
    def __init__(self,axis=-1):
        self.axis=axis
    def __call__(self,x):
        e_x = np.exp(x - np.max(x,axis=self.axis,keepdims=True)) # max(x) subtracted for numerical stability
        self.sm = e_x / np.sum(e_x,axis=self.axis,keepdims=True)
        return self.sm
        
    def backward(self,err):
               
        SM = self.sm.reshape(-1,1)
        jac = np.diagflat(self.sm) - np.dot(SM, SM.T)
        jac=np.sum(jac,axis=1,keepdims=True).reshape(err.shape)
        
        return jac*err
def Masking(matrix):
    
    look_ahead_mask=np.tril((np.ones(matrix.shape)))
    look_ahead_mask=np.where(look_ahead_mask==1,0,-1e20)
    scores=matrix +look_ahead_mask
   
    return scores
    
class MultiAttention():
    def __init__(self,dim,heads,mask=False):
        
        self._Keyss=Linear(dim,dim*heads) # for self attention Feed_Forward(dim,dim) it has only one vector to store the attention between two words where multi head attention has muliple vectors to store relationship/attention between two words for improved performance
        self._Query=Linear(dim,dim*heads)
        self._Value=Linear(dim,dim*heads)
        self._Unify=Linear(dim*heads,dim)
        self.head=heads
        self.D=dim
        self.sm=Softmax()
        self.mask=mask
        self.param={}
    def zero_grad(self):
        
        self._Keyss.zero_grad()
        self._Query.zero_grad()
        self._Value.zero_grad()
        self._Unify.zero_grad()
        
    def forward(self,input_vec,time_step):# after embedding layer and adding with pos enc the input reaches attention layer
        ts=time_step
        #initially we are passing the input vectors through (key,query ,value) Weights
        queries=self._Query.forward(input_vec,ts) #determines which values to focus
        keys   =self._Keyss.forward(input_vec,ts) #hint to find score for its value pair      
        values =self._Value.forward(input_vec,ts) # extract interesting features  
        #print(self._Keyss.Weight,'\n\n',self._Query.Weight,'\n\n')
        
        
        # batchsize,seqlen,head*dim   ====>>>   batchsize*head,seqlen,dim since the dimensions of sequence will react with other all sequences within same head ,heads wont react with each other
        sha=list(keys.shape)
        sha_dh=sha.copy()     
        self.param[str(ts)+'shape_BSDxh']=sha_dh
        sha[-1]=int(sha[-1]/self.head) # shapes will be same in all time steps
        sha[0]*=self.head
        self.param[str(ts)+'shape_BxhSD']=sha
        
        
        # Weight are reshaped

        keys=keys.reshape(sha)
        queries=queries.reshape(sha)
        values=values.reshape(sha)
        
        # parameters are saved for backpropagation
        
        self.param[str(ts)+'k']=keys   
        self.param[str(ts)+'q']=queries
        self.param[str(ts)+'v']=values
        
        
        # Each Query is iterated with each other key to Get its score with the corresponding word
        
        #print(keys,'\n\n',queries,'\n\n')
       
        W_=np.matmul(queries, np.transpose(keys, (0, 2, 1)))
        
        W_=W_*(1./np.sqrt(self.D))
        if(self.mask):
            print("mask activated")
            W_=Masking(W_)# masking will be done each word will not have acces to future word
            
        
        #print('\n\n',queries,'\n\n',keys,'\n\n',W_,'\n\n')
        W=self.sm(W_) 

        
        self.param[str(ts)+'W']=W
        
        Y=np.matmul(W,values) # the scores are multiplied with the values and they are added
        # concat the heads
        
        Y=Y.reshape(sha_dh)
        op=self._Unify.forward(Y,ts)
        
        
        return op
    
    def backward(self,err,time_step,lr):
        ts=time_step
        con=np.sum(err.reshape(-1))
        dY=self._Unify.backward(err,ts,lr)
        
        dY=dY.reshape(self.param[str(ts)+'shape_BxhSD'])
        
        dval=np.matmul(self.param[str(ts)+'W'],dY)     
        dW =np.matmul(self.param[str(ts)+'v'],np.transpose(dY, (0, 2, 1)))
        
        dW_=self.sm.backward(dW)
        der_Scale=(-0.5*(np.power(self.D,-(3/2))))
        dW_=dW_*der_Scale
        
        zw=np.sum(dW_.reshape(-1))
        dkeys=np.matmul(dW_,self.param[str(ts)+'q'])
        dque=np.matmul(dW_,self.param[str(ts)+'k'])
        sha_=self.param[str(ts)+'shape_BSDxh']
        dkeys=dkeys.reshape(sha_)
        dque=dque.reshape(sha_)
        dval=dval.reshape(sha_)
        
        x=np.sum(dque.reshape(-1))
        y=np.sum(dkeys.reshape(-1))
        z=np.sum(dval.reshape(-1))
      
        der_inp =self._Keyss.backward(dkeys,ts,lr)
        der_inp+=self._Query.backward(dque,ts,lr)
        der_inp+=self._Value.backward(dval,ts,lr)
        #print(x,y,z,zw,con)
        
        return der_inp   














































class EDCAttention():
    def __init__(self,dim,heads):
        
        self._Keyss=Linear(dim,dim*heads) # for self attention Feed_Forward(dim,dim) it has only one vector to store the attention between two words where multi head attention has muliple vectors to store relationship/attention between two words for improved performance
        self._Query=Linear(dim,dim*heads)
        self._Value=Linear(dim,dim*heads)
        self._Unify=Linear(dim*heads,dim)
        self.head=heads
        self.D=dim
        self.sm=Softmax()
        self.param={}
    def zero_grad(self):
        
        self._Keyss.zero_grad()
        self._Query.zero_grad()
        self._Value.zero_grad()
        self._Unify.zero_grad()    
    def forward(self,input_vec,Enc_vec,time_step,mask=None):# after embedding layer and adding with pos enc the input reaches attention layer
        ts=time_step
        #initially we are passing the input vectors through (key,query ,value) Weights
        queries=self._Query.forward(input_vec,ts) #determines which values to focus
        keys   =self._Keyss.forward(Enc_vec,ts) #hint to find score for its value pair      
        values =self._Value.forward(Enc_vec,ts) # extract interesting features  
        #print(self._Keyss.Weight,'\n\n',self._Query.Weight,'\n\n')
        
        
        # batchsize,seqlen,head*dim   ====>>>   batchsize*head,seqlen,dim since the dimensions of sequence will react with other all sequences within same head ,heads wont react with each other
        sha=list(keys.shape)
        sha_dh=sha.copy()     
        self.param[str(ts)+'shape_BSDxh']=sha_dh
        sha[-1]=int(sha[-1]/self.head) # shapes will be same in all time steps
        sha[0]*=self.head
        self.param[str(ts)+'shape_BxhSD']=sha
        
        
        # Weight are reshaped

        keys=keys.reshape(sha)
        queries=queries.reshape(sha)
        values=values.reshape(sha)
        
        # parameters are saved for backpropagation
        
        self.param[str(ts)+'k']=keys   
        self.param[str(ts)+'q']=queries
        self.param[str(ts)+'v']=values
        
        
        # Each Query is iterated with each other key to Get its score with the corresponding word
        
        #print(keys,'\n\n',queries,'\n\n')
       
        W_=np.matmul(queries, np.transpose(keys, (0, 2, 1)))
        
        W_=W_*(1./np.sqrt(self.D))
        if(mask):
            W_=Masking(W_)# masking will be done each word will not have acces to future word
            
        
        #print('\n\n',queries,'\n\n',keys,'\n\n',W_,'\n\n')
        W=self.sm(W_) 

        
        self.param[str(ts)+'W']=W
        
        Y=np.matmul(W,values) # the scores are multiplied with the values and they are added
        # concat the heads
        
        Y=Y.reshape(sha_dh)
        op=self._Unify.forward(Y,ts)
        
        
        return op
    
    def backward(self,err,time_step,lr):
        ts=time_step
        con=np.sum(err.reshape(-1))
        dY=self._Unify.backward(err,ts,lr)
        
        dY=dY.reshape(self.param[str(ts)+'shape_BxhSD'])
        
        dval=np.matmul(self.param[str(ts)+'W'],dY)     
        dW =np.matmul(self.param[str(ts)+'v'],np.transpose(dY, (0, 2, 1)))
        
        dW_=self.sm.backward(dW)
        der_Scale=(-0.5*(np.power(self.D,-(3/2))))
        dW_=dW_*der_Scale
        
        zw=np.sum(dW_.reshape(-1))
        dkeys=np.matmul(dW_,self.param[str(ts)+'q'])
        dque=np.matmul(dW_,self.param[str(ts)+'k'])
        sha_=self.param[str(ts)+'shape_BSDxh']
        dkeys=dkeys.reshape(sha_)
        dque=dque.reshape(sha_)
        dval=dval.reshape(sha_)
        
        x=np.sum(dque.reshape(-1))
        y=np.sum(dkeys.reshape(-1))
        z=np.sum(dval.reshape(-1))
        
        der_enc=self._Query.backward(dque,ts,lr)
        
        der_inp =self._Keyss.backward(dkeys,ts,lr)
        der_inp+=self._Value.backward(dval,ts,lr)
        #print(x,y,z,zw,con)
        
        return der_inp,der_enc   



