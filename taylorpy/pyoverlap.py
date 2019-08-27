import numpy as np

def cmplxInnerProd(a, b, weights, df):
    """computes complex inner product in the fourier domain IP = 4 deltaf sum((a * conguagte(b))/Weights)"""
    
    a_ = a/np.sqrt(weights)
    b_ = b/np.sqrt(weights)
    
    prod = 4*df*np.vdot(a_,b_)
    
    return prod

def InnerProd(a, b, weights, df):
    """computes inner product in the fourier domain IP = 4 deltaf RE sum((a * conguagte(b))/Weights)"""
    
    return cmplxInnerProd(a, b, weights, df).real

def matchedfilter(a, b, weights, df):
    """Computes the overlap between two waveforms"""
    
    prod_a_b = InnerProd(a,b,weights,df)
    prod_a_a = InnerProd(a,a,weights,df)
    prod_b_b = InnerProd(b,b,weights,df)
    
    return prod_a_b/np.sqrt(prod_a_a*prod_b_b)

def match(temp,data,psd,df,freq,tc_low = -1.,tc_high = 1.):
    """Calculate the max match between two waveforms by maximisiung over coalescence time"""
    
    t_c = np.linspace(tc_low,tc_high,200)
    norm_condition = 1/max(temp)
    freq_tc = np.outer(t_c,freq1)
    shift_factor = np.exp(1j*2*np.pi*freq_tc)*norm_condition
    
    match_max = np.zeros(len(t_c))
    for coa_time in range(len(t_c)):
        waveform_shift = shift_factor[coa_time,:]
        waveform_shifted = waveform_shift*temp
        match_max[coa_time] = matchedfilter(waveform_shifted,data,psd,df)
    
    return t_c, match_max

def relativelikelihood(a, b, weights, df):
    """Computes the relative log likelihood of two data sets, computed by <data|temp>-1/2<temp|temp> """
    
    return (InnerProd(a,b,weights,df)-0.5*InnerProd(b,b,weights,df))

def SNR(a, b, weights, df):
    """Compute the SNR at the true value of the parameters """
    
    return np.sqrt(2*relativelikelihood(a, b, weights, df))
