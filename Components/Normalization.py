# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 14:45:04 2021

@author: mitran
"""
import numpy as np
"""
class LayerNormalization():
    def __init__(self,axis=-1):
        self.axis=axis
        self.e=0.001 #epsilon
        self.alpha=1
        self.beta=0
        self.momentum=0.99
    def forward(self,data):
        # we have to normalize the data across hidden units  so that standardizes the inputs to a layer helping to converge easily and fastly
        
        self.H=data.shape[-1]# the hidden unit dimension
        self.mean=np.sum(data,axis=-1,keepdims=True)/self.H
        a=data.copy()           
        self.Z=a-self.mean
        
        var=(self.Z)**2
        self.var=np.sum(var,axis=-1,keepdims=True)/self.H            
                
        num=(self.Z)
        den=np.sqrt(self.var+self.e)
        
        self.A=num/(den) # let A be the utput       
        Y=self.alpha*self.A+self.beta
        #print(data.reshape(-1),Y.reshape(-1))
        return Y
        
        
    def backward(self,err,lr):
        
        d_alpha=self.A*err
        d_alpha=np.sum(d_alpha.reshape(-1))
        d_beta=np.sum(err.reshape(-1))
        d_A=err*self.alpha
        
        dA_dx1=1/(np.sqrt(self.var+self.e))
        dA_dmean1=-1/(np.sqrt(self.var+self.e))
        
        dA_dvar=-0.5*self.Z*((self.var+self.e)**(-(3/2)))*1
                
        dvar_dx=(2*self.Z)/self.H
        dvar_dmean=(-2*self.Z)/self.H
        
        dmean_dx=1/self.H
        
        d_var=np.sum(d_A*dA_dvar,axis=-1,keepdims=True)
        d_mean=np.sum(d_A*dA_dmean1,axis=-1,keepdims=True)+np.sum(d_var*dvar_dmean,axis=-1,keepdims=True)
        
        d_x   =d_A*dA_dx1+d_var*dvar_dx+d_mean*dmean_dx
       
        
        self.alpha-=d_alpha*lr
        self.beta-=d_beta*lr
        
        return d_x
       
"""        
class LayerNormalization():
    def __init__(self,axis=-1):
        self.axis=axis
        self.e=0.001 #epsilon
        self.alpha=1
        self.beta=0
        self.momentum=0.99
    def forward(self,x):
        y,cache=forward_batch(x,self.alpha,self.beta,self.e)
        self.C=cache
        return y
    def backward(self,err,lr):
        
        derr,a,b=backward_batch(err,self.C,self.alpha)
        self.alpha-=(a*lr*0.000001)
        self.beta-=(b*lr*0.000001)
        return derr
        
    
def forward_batch(x,a,b,e):
   # Mean of the mini-batch, mu
   mu = np.mean(x, axis=-1,keepdims=True)

   # Variance of the mini-batch, sigma^2
   var = np.var(x, axis=-1,keepdims=True)
   std_inv = 1.0 / np.sqrt(var + e)

   # The normalized input, x_hat
   x_hat = (x - mu) * std_inv
     
   y = a * x_hat + b

   cache = ( std_inv, x_hat)

   return y, cache
        
def backward_batch(dy, backprop_stuff,alpha):
   
   std_inv, x_hat = backprop_stuff
   
   dx_hat = dy * alpha
   dx = std_inv * (dx_hat - np.mean(dx_hat, axis=-1,keepdims=True) - x_hat * np.mean(dx_hat * x_hat, axis=-1,keepdims=True))
   """print(np.sum(abs(std_inv.reshape(-1)))) 
   print(np.sum(abs(((dx_hat - np.mean(dx_hat, axis=-1,keepdims=True) - x_hat)).reshape(-1)))) 
   print(np.sum(abs((np.mean(dx_hat * x_hat, axis=-1,keepdims=True)).reshape(-1)))) 
   print(np.sum(abs(dx.reshape(-1)))) """
   #dx=dx(4/dy.shape[-1])
   dalpha = np.sum((dy * x_hat).reshape(-1))
   dbeta = np.sum(dy.reshape(-1))
    
    
    
   return dx, dalpha, dbeta        
        
      
        
        
        
        