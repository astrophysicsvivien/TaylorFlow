#Preamble
import tensorflow as tf
#enable Eager execution
tf.enable_eager_execution()

import numpy as np
import matplotlib.pyplot as plt

#Start Interactive Session
sess = tf.InteractiveSession()

def PN_phases(freq,mTot,eta,t_c,phi_c,phase_order):
    
    #constants in equation
    piM = tf.multiply(mTot,np.pi) # Total mass times pi
    v = tf.pow(tf.multiply(piM,freq),(1/3)) # characteristic velocity of binary
    etasq = tf.pow(eta,2) # square of the symmetric mass ratio
    etacb= tf.pow(eta,3) # cube of the symmetric mass ratio
    vISCO = 1. / tf.sqrt(6.) # velocity at innermost stable circular orbit
    gamma = 0.577215664901532 #Euler–Mascheroni constant
    
    #v parameters
    v0 = tf.pow(v,-5)
    v2 = tf.pow(v,2)
    v3 = tf.pow(v,3)
    v4 = tf.pow(v,4)
    v5 = tf.pow(v,5)
    v5log = tf.log(tf.divide(v,vISCO))
    v6 = tf.pow(v,6)
    v6log = tf.log(4*v)
    v7 = tf.pow(v,7)
    
    #produce PN coeffiecients
    PN_cnst = tf.add((2*np.pi*t_c*freq),(-phi_c))
    P0 = tf.multiply((3./128),tf.reciprocal(eta))
#     P2 = tf.multiply(tf.add(743/84,((11)*eta)),(5/9))
#     P3 = (-16*np.pi)
#     P4 = tf.multiply(tf.add((3058673/1016064),tf.add(((5429/1008)*eta),(617/144)*etasq)),(10))
#     P5 = tf.multiply(tf.add(38645/756,(-(65/9)*eta)),(np.pi))
#     Pl5 = tf.multiply(tf.multiply(tf.add(38645/756,(-(65/9)*eta)),(np.pi)),3)
#     P6 = tf.add(((11583231236531/4694215680)-(640*np.pi*np.pi/3)-(6848*gamma/21)),\
#          tf.add(((((-15737765635/3048192)+(2255*np.pi*np.pi/12)))*eta),\
#          tf.add(((76055/1728)*etasq),(-127825/1296)*etacb)))
#     Pl6 = -(6848/21)
#     P7 = tf.multiply(tf.add((77096675/254016),tf.add(((378515/1512)*eta),(-74045/756)*etasq)),(np.pi))

    # For setting terms to zero 
    P2 = 0.
    P3 = 0.
    P4 = 0.
    P5 = 0.
    Pl5 = 0.
    P6 = 0.
    Pl6 = 0.
    P7 = 0.
    
    #Produce full PN terms
    PN0 = tf.multiply(P0,v0)
    PN2 = tf.multiply(P2,v2)
    PN3 = tf.multiply(P3,v3)
    PN4 = tf.multiply(P4,v4)
    PN5 = tf.multiply(tf.add(P5,tf.multiply(Pl5,v5log)),v5)
    PN6 = tf.multiply(tf.add(P6,tf.multiply(Pl6,v6log)),v6)
    PN7 = tf.multiply(P7,v7)
    
    #phases = PN_cnst + PN0(1+PN1*v+PN2*v2+PN3*v3+PN4+PN5+PN6+PN7)
    phases = tf.add(PN_cnst,tf.multiply(PN0,(1+PN2+PN3+PN4+PN5+PN6+PN7)))
 
    return phases
                         
def PN_amplitude(freq,ChirpMass,LuminosityDistance):
    #returns the amplitude of the 0PN waveform
    #arguments are chirp mass, luminosity distance and frequencies
                         
    term1 = tf.reciprocal(LuminosityDistance/3e8)# divide by factor of c to convert to units of time
    term2 = tf.multiply(np.sqrt(5/24)*(np.pi**(-2/3)),tf.pow(ChirpMass*4.96e-6,(5/6))) #times by mtsol to get units of time
    term3 = tf.pow(freq,(-7/6))
                         
    return tf.multiply(term1,tf.multiply(term2,term3))
                         
def TaylorFlow(mass1,mass2,frequencies=None,LuminosityDistance=1.,t_c=0,phi_c=0.,f_low=10.,
               df=0.1,f_high=2000.,phase_order=0):
    """
    TaylorFlow Main Function
    
    """
                         
    #Define variable and constants in TensorFlow variables and constants   
    #Masses and frequencies are required inputs
    mass1 = tf.constant(mass1,name="mass1",dtype=tf.float32)
    mass2 = tf.constant(mass2,name="mass2",dtype=tf.float32)
    
    #get total mass,chirp mass and symetric mass ratio for use later in function
    M = tf.add(mass1,mass2)
    eta = tf.divide(tf.multiply(mass1,mass2),tf.square(M))
    #Chirp mass is given by M*(eta^3/5)
    ChirpMass = tf.multiply(M,tf.pow(eta,(3/5)))    
    
    #define frequencies
    f_ISO = 1/(6.**1.5*np.pi*(M)*4.93e-6) #frequency of innermost stable circular orbit (SI units)
    if frequencies == None:
        f = np.arange(1.,f_high,df)
        frequencies = tf.Variable(f,name= "frequencies",dtype= tf.float32)
    else:  
        frequencies = tf.Variable(frequencies,name= "frequencies",dtype= tf.float32)
        
    #other constants                     
    L_D =  tf.constant((3.086e+22*LuminosityDistance),name="LuminosityDistance",dtype=tf.float32) 
    
    #get phases at 0PN order
    phase = PN_phases(frequencies,M,eta,t_c,phi_c,phase_order)
    amp = PN_amplitude(frequencies,ChirpMass,L_D)
    
    #set amplitude to zero where the waveform results are unphysical
    cond1 = frequencies<f_low #below f_low
    ind1 = tf.where(cond1)
    cond2 = frequencies>f_ISO #above fISCO
    ind2 = tf.where(cond2)
    amp = tf.scatter_update(tf.Variable(amp), ind1, tf.constant(0.))
    amp = tf.scatter_update(tf.Variable(amp), ind2, tf.constant(0.))
    
    #Calculte waveform
    iphase = tf.multiply(tf.complex(0.,1.),tf.complex(phase,0.))
    #iphase = tf.add(tf.complex(tf.cos(phase),0.),tf.multiply(tf.complex(0.,1.),tf.complex(tf.sin(phase),0.)))
    waveform = tf.multiply(tf.complex(amp,0.),tf.exp(iphase))
    #waveform = tf.multiply(tf.complex(amp,0.),iphase)
    return frequencies,waveform

def getFrequencies(mass1,mass2,f_low,df):
    "defines a simple set of frequencies to evaluate waveform over"
    f_high = fISCO(mass1,mass2)
    N = int(f_high/df)
    freq = np.linspace(f_low,f_high,N,np.float64)
    return freq
