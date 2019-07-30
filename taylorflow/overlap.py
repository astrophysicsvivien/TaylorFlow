#Preamble
import tensorflow as tf
#enable Eager execution
tf.enable_eager_execution()

def cmplxInnerProd(temp,data,psd,df):
    "computes complex inner product in the fourier domain IP = 4 deltaf sum((a * conguagte(b))/Weights)"
    
    weights = tf.sqrt(psd)
    norm = (4*df)
    a_weight = tf.divide(temp,weights)
    b_conj = tf.conj(data)
    b_weight = tf.divide(tf.cast(b_conj,dtype=tf.complex64),weights)
    a_dot_b = tf.reduce_sum(tf.multiply(a_weight,b_weight))
    
    return tf.multiply(norm,a_dot_b)

def InnerProd(temp,data,psd,df):
    "computes inner product in the fourier domain IP = 4 deltaf RE sum((a * conguagte(b))/Weights)"
    
    return tf.real(cmplxInnerProd(temp,data,psd,df))

def sigsq(temp, psd, df):
    "computes sig^2 = (a|a), which can then be used to normalise function "
    
    weights = tf.sqrt(psd)
    norm = (4*df)
    a_weight = tf.divide(temp,weights)
    b_conj = tf.conj(temp)
    b_weight = tf.divide(tf.cast(b_conj,dtype=tf.complex64),weights)
    a_dot_b = tf.reduce_sum(tf.multiply(a_weight,b_weight))
    
    return tf.real(tf.multiply(norm,a_dot_b))

def sig(temp, psd, df):
    "returns the sigma value of the signal"
    
    return tf.sqrt(sigsq(temp, psd, df))

def tensornorm(a):
    "divides by max value of tensor to normalise between 0 and 1"
    TensorMax = tf.reduce_max(tf.abs(a))
    Tmax_cmplx_recip = tf.complex(tf.reciprocal(TensorMax),0.)
    
    return tf.multiply(a,Tmax_cmplx_recip),TensorMax

def overlap(temp,data,psd,df):
    "Overlap equation"
    
    norm = tf.divide(1,tf.multiply(sig(temp,psd,df),sig(data,psd,df)))
    inner = InnerProd(temp,data,psd,df)
    overlap = tf.multiply(norm,inner)
        
    return overlap
