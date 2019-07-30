import numpy as np
import scipy as sp 
import matplotlib.pyplot as plt


def ComplexInnerProduct(a,b,delta_f,weights='none'):
    "computes complex inner product in the fourier domain IP = 4RE deltaf sum((a * conguagte(b))/PSD Weights)"
    
    if weights =='none':
        weights = np.ones(len(a))
        
    return 4*delta_f*(np.vdot((a/np.sqrt(weights)),(b/np.sqrt(weights))))

def InnerProduct(a,b,delta_f,weights='none'):
    "returns real component of Inner Product"
    
    return ComplexInnerProduct(a,b,delta_f,weights).real

def sigsq(a,weights,delta_f):
    "returns the square of the normal of a waveform (root of inner product with itself)"
    
    return 4*delta_f*np.real((np.vdot((a/np.sqrt(weights)),(a/np.sqrt(weights)))))

def sig(a,weights,delta_f):
    
    "returns the normal of a waveform (root of inner product with itself)"
                        
    return np.sqrt(sigsq(a,weights,delta_f))

def overlap(h1,h2,weights,delta_f):
    "returns the normalised overlap of the signals"
                        
    return (1/sig(h1,weights,delta_f)/sig(h2,weights,delta_f))*InnerProduct(h1,h2,delta_f,weights)