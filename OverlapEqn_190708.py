#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Preamble
import tensorflow as tf
#enable Eager execution
tf.enable_eager_execution()

import numpy as np
import matplotlib.pyplot as plt
import TaylorFlow0PN as tlf


# In[8]:


def cmplxInnerProd(a,b,psd,df):
    "computes complex inner product in the fourier domain IP = 4 deltaf sum((a * conguagte(b))/Weights)"
    a_psd = tf.divide(a,psd)
    intgrl = tf.reduce_sum(tf.multiply(a_psd,b))
    prefactor = 4*df
    return tf.multiply(prefactor,intgrl)

def InnerProd(a,b,psd,df):
    "computes inner product in the fourier domain IP = 4 deltaf RE sum((a * conguagte(b))/Weights)"
    return tf.real(cmplxInnerProd(a,b,psd,df))

def sigsq(a, psd, df):
    "computes sig^2 = (a|a), which can then be used to normalise function "
    norm = tf.constant(4*df)
    inner = InnerProd(a,a,psd,df)
    return tf.multiply(norm,inner)

def sig(a, psd, df):
    "returns the sigma value of the signal"
    return tf.sqrt(sigsq(a, psd, df))

def overlap(a,b,psd,df):
    "Overlap equation"
    
    #norm= tf.reciprocal(tf.multiply(sig(a,psd,df),sig(b,psd,df)))
    norm = tf.divide(4*df,tf.multiply(sig(a,psd,df),sig(b,psd,df)))
    print(norm)
    inner = InnerProd(a,b,psd,df)
    print(inner)
    overlap = tf.multiply(norm,inner)
    print(overlap)

        
    return overlap


# In[9]:


#TaylorFlow(mass1,mass2,frequencies,LuminosityDistance=1.,t_c=-1.0,phi_c=np.pi)
#getFrequencies(mass1,mass2,f_low,df)

mass1 = 1.4
mass2 = 1.4
f_low = 5
df = 0.1

freq = tlf.getFrequencies(mass1,mass2,f_low,df)
signal1 = tlf.TaylorFlow(mass1,mass2,freq)
signal2 = tlf.TaylorFlow(mass1,mass2,freq)

plt.plot(freq,np.real(signal1))
#plt.plot(freq,signal2)


# In[10]:


print(signal1)
print(signal2)

s1max = tf.reduce_max(tf.abs(signal1))
print(s1max)
signal1 = tf.divide(tf.abs(signal1),s1max)
print(signal1)
s2max = tf.reduce_max(tf.abs(signal2))
print(s2max)
signal2 = tf.divide(tf.abs(signal2),s1max)
print(signal2)
psd = np.ones(len(signal1))
print(freq)


# In[11]:


overlap1 = overlap(signal1,signal2,psd,df,normalised='true')


# In[12]:


overlap2 = overlap(signal1,signal2,psd,df,normalised='false')


# In[13]:


cmplxIP = cmplxInnerProd(signal1,signal2,psd,df)
print(cmplxIP)
IP = InnerProd(signal1,signal2,psd,df)
print(IP)
sig1 = sig(signal1,psd,df)
print(sig1)
sig2 = sig(signal2,psd,df)
print(sig2)
sigsq= sigsq(signal1, psd, df)
print(sigsq)


# In[ ]:





# In[ ]:




