import tensorflow as tf
import numpy as np

def tensornorm(a):
    """divides by max value of tensor to normalise between 0 and 1"""
    TensorMax = tf.reduce_max(tf.abs(a))
    norm_condition = tf.complex(tf.math.reciprocal(TensorMax),0.)
    
    return tf.multiply(a,norm_condition)

def ComplexInnerProduct(temp,data,psd,df):
    """computes complex inner product in the fourier domain IP = 4 deltaf sum((a * conjugate(b))/Weights)"""
    
    weights = tf.sqrt(psd)
    #print('weights is {}'.format(weights))
    norm = (4*df)
    #print('norm is {}'.format(norm))
    a_weight = tf.math.divide_no_nan(temp,weights)
    #print('a_weight is {}'.format(a_weight))
    b_conj = tf.math.conj(data)
    #print('b_conj is {}'.format(b_conj))
    b_weight = tf.math.divide_no_nan(tf.cast(b_conj,dtype=tf.complex64),weights)
    #print('b_weight is {}'.format(b_weight))
    a_dot_b = tf.reduce_sum(tf.multiply(a_weight,b_weight))
    #print('a_dot_b is {}'.format(a_dot_b))
    
    return tf.multiply(norm,a_dot_b)

def InnerProduct(temp,data,psd,df):
    """computes inner product in the fourier domain IP = 4 deltaf RE sum((a * conguagte(b))/Weights)"""
    
    return tf.math.real(ComplexInnerProduct(temp,data,psd,df))

def matchedfilter(temp,data,psd,df):
    """Computes the overlap between two waveforms"""
    
    prod_a_b = InnerProduct(temp,data,psd,df)
    prod_a_a = InnerProduct(temp,temp,psd,df)
    prod_b_b = InnerProduct(data,data,psd,df)
    
    sigma = tf.sqrt(tf.multiply(prod_a_a,prod_b_b))
    
    return tf.divide(prod_a_b,sigma)

def match(temp,data,psd,df,freq,tc_low = -1.,tc_high = 1.,samples=101):
    """Calculate the max match between two waveforms by maximisiung over coalescence time"""
    
    t_c = tf.cast(tf.linspace(tc_low,tc_high,samples),dtype=tf.float32)
    freq_tc = tf.tensordot(t_c,freq,axes=0)
    freq_tc = tf.cast(freq_tc,dtype=tf.complex64)
    shift_factor = tf.exp(1j*2*np.pi*freq_tc)

    match_max = np.zeros(len(t_c))
    for coa_time in range(len(t_c)):
        waveform_shift = shift_factor[coa_time,:]
        waveform_shifted = tf.multiply(waveform_shift,temp)
        match_max[coa_time] = matchedfilter(waveform_shifted,data,psd,df)
    
    return t_c, match_max

def match_cpu(temp,data,psd,df,freq, tc_low = -1.,tc_high = 1.,samples=201):
    """Calculate the max match between two waveforms by maximisiung over coalescence time, calculated on the cpu"""
    
    with tf.device('/cpu:0'):
        t_c = tf.cast(tf.linspace(tc_low,tc_high,samples),dtype=tf.float32)
        freq_tc = tf.tensordot(t_c,freq,axes=0)
        freq_tc = tf.cast(freq_tc,dtype=tf.complex64)
        shift_factor = tf.exp(1j*2*np.pi*freq_tc)

        match_max = np.zeros(len(t_c))
        for coa_time in range(len(t_c)):
            waveform_shift = shift_factor[coa_time,:]
            waveform_shifted = tf.multiply(waveform_shift,temp)
            match_max[coa_time] = matchedfilter(waveform_shifted,data,psd,df)
    
    return t_c, match_max

def loglikelihood(temp,data,psd,df):
    """Computes the relative log likelihood of two data sets, computed by <data|temp>-1/2<temp|temp> """
    return (InnerProduct(data,temp,psd,df)-0.5*InnerProduct(temp,temp,psd,df))


def SNR(temp,data,psd,df):
    """Compute the SNR at the true value of the parameters """

    return tf.sqrt(2*loglikelihood(temp,data,psd,df))