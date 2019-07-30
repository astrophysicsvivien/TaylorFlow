#preamble
import numpy as np
import matplotlib.pyplot as plt

def PN_phases(freq,mTot,eta,t_c,phi_c):
    "produce phases at 0PN order"
    #Characteristic velocity (expansion parameter)
    
    piM = mTot*np.pi*4.92549102554e-6 # pi multiplyed by total mass in seconds
    v = ((piM*freq)**(1/3))
    
    # generate phases
    phase =  (2*np.pi*freq*t_c) - phi_c + ((3/128)*(1/eta*(v**(-5))))
    return phase

def PN_amplitude(freq,ChirpMass,LuminosityDistance):
    "produce amplitude at 0PN order"
    
    return (299792458/LuminosityDistance)*(((5/24)**0.5)*(np.pi**(-2/3)))*((ChirpMass*4.92549102554e-6)**(5/6))*(freq**(-7/6))

def taylorf2(mass1,mass2,frequencies=None,LuminosityDistance=1.,t_c=0.,phi_c=0.,f_low=20.,
               df=1./512,f_high=1600.):
    
    #genrate characteristic masses
    M = mass1+mass2
    eta = (mass1*mass2)/(M**2)
    ChirpMass = M*(eta**(3/5)) #Chirp mass is given by M*(eta^3/5)
    
    #define frequencies
    f_ISO = 1/(6.**1.5*np.pi*(M)*4.92549102554e-6) #frequency of innermost stable circular orbit (SI units)
    if frequencies == None:
        f = np.arange(1.,f_high,df)
        freq = f
    else:  
        freq = frequencies 
        
    #Luminosity distance in SI Units
    L_D =  3.086e+22*LuminosityDistance
    
    #Evaluate phases and amplitudes across frequencies
    phase = PN_phases(freq,M,eta,t_c,phi_c)
    amp = PN_amplitude(freq,ChirpMass,L_D)
    
    #set amplitude to zero where the waveform results are unphysical or unwanted
    amp[freq<(f_low)] = 0.
    amp[freq>f_ISO] = 0.
    
    #Produce waveform
    waveform = amp*np.exp(1j*phase)
    
    return freq, waveform