import tensorflow as tf
#enable Eager execution
tf.enable_eager_execution()

import numpy as np

def PN_phases(freq,mTot,eta,phase_order):
    """
    PN phase orders for TaylorF2
    """
    
    #constants in equation
    piM = tf.multiply(mTot*4.92549102554e-6,np.pi) # Total mass times pi
    v = tf.pow(tf.multiply(piM,freq),(1/3)) # characteristic velocity of binary
    etasq = tf.pow(eta,2) # square of the symmetric mass ratio
    etacb= tf.pow(eta,3) # cube of the symmetric mass ratio
    gamma = 0.577215664901532 #Euler–Mascheroni constant
    
    #v parameters
    v0 = tf.pow(v,-5)
    v2 = tf.pow(v,2)
    v3 = tf.pow(v,3)
    v4 = tf.pow(v,4)
    v5 = tf.pow(v,5)
    v5log = tf.log(v)
    v6 = tf.pow(v,6)
    v6log = tf.log(4*v)
    v7 = tf.pow(v,7)
    
    #produce PN coeffiecients
    P0 = tf.multiply((3./128),tf.reciprocal(eta))
    P2 = tf.multiply(tf.add(743/84,((11)*eta)),(5/9))
    P3 = (-16*np.pi)
    P4 = tf.multiply(tf.add((3058673/1016064),tf.add(((5429/1008)*eta),(617/144)*etasq)),(10))
    P5 = tf.multiply(tf.add(7729/84,(-13*eta)),(np.pi*5/9))
    Pl5 = tf.multiply(tf.add(7729/84,(-13*eta)),(np.pi*5/3))
    P6 = tf.add(((11583231236531/4694215680)-(640*np.pi*np.pi/3)-(6848*gamma/21)),\
         tf.add(((((-15737765635/3048192)+(2255*np.pi*np.pi/12)))*eta),\
         tf.add(((76055/1728)*etasq),(-127825/1296)*etacb)))
    Pl6 = -(6848/21)
    P7 = tf.multiply(tf.add((77096675/254016),tf.add(((378515/1512)*eta),(-74045/756)*etasq)),(np.pi))

    
    #Produce full PN terms
    PN0 = tf.multiply(P0,v0)
    PN2 = tf.multiply(P2,v2)
    PN3 = tf.multiply(P3,v3)
    PN4 = tf.multiply(P4,v4)
    PN5 = tf.multiply(tf.add(P5,tf.multiply(Pl5,v5log)),v5)
    PN6 = tf.multiply(tf.add(P6,tf.multiply(Pl6,v6log)),v6)
    PN7 = tf.multiply(P7,v7)
    
    #phases = PN_cnst + PN0(1+PN1*v+PN2*v2+PN3*v3+PN4+PN5+PN6+PN7)
    if phase_order == 7:
        phases = tf.multiply(PN0,(1+PN2+PN3+PN4+PN5+PN6+PN7))
    elif phase_order == 6:
        phases = tf.multiply(PN0,(1+PN2+PN3+PN4+PN5+PN6))
    elif phase_order == 5:
        phases = tf.multiply(PN0,(1+PN2+PN3+PN4+PN5))
    elif phase_order == 4:
        phases = tf.multiply(PN0,(1+PN2+PN3+PN4)) 
    elif phase_order == 3:
        phases = tf.multiply(PN0,(1+PN2+PN3))
    elif phase_order == 2:
        phases = tf.multiply(PN0,(1+PN2))
    elif phase_order == 1:
        phases = PN0
    else:
        phases = PN0
        

    return phases
                         
def PN_amplitude(freq,ChirpMass,LuminosityDistance):
    """
    Amplitude at 0PN order
    
    """
    #returns the amplitude of the 0PN waveform
    #arguments are chirp mass, luminosity distance and frequencies
                         
    term1 = tf.reciprocal(LuminosityDistance/299792458)# divide by factor of c to convert to units of time
    term2 = tf.multiply(np.sqrt(5/24)*(np.pi**(-2/3)),tf.pow(ChirpMass*4.92549102554e-6,(5/6))) #times by mtsol to get units of time
    term3 = tf.pow(freq,(-7/6))
                         
    return tf.multiply(term1,tf.multiply(term2,term3))
                         
def getwaveform(mass1,mass2,frequencies=None,LuminosityDistance=1.,f_low=10.,
               df=1./512,f_high=1600.,phase_order=7):
    """
    TaylorFlow Main Function
    
    """
                         
    #Define variable and constants in TensorFlow variables and constants   
    #Masses and frequencies are required inputs
    mass1 = tf.constant(mass1,name="mass1",dtype=tf.float32)
    mass2 = tf.constant(mass2,name="mass2",dtype=tf.float32)
    
    #get total mass,chirp mass and symetric mass ratio for use later in function
    M = tf.add(mass1,mass2)
    eta = tf.divide(tf.multiply(mass1,mass2),tf.pow(M,2))
    ChirpMass = tf.multiply(M,tf.pow(eta,(3/5))) #Chirp mass is given by M*(eta^3/5)   
    
    #define fISCO
    f_ISO = 1/(6.**1.5*np.pi*(M)*4.92549102554e-6) #frequency of innermost stable circular orbit (SI units)
    
    #define frequencies
    if frequencies is None:
        f = np.arange(1.,f_high,df)
        frequencies = tf.Variable(f,name= "frequencies",dtype= tf.float32)
    else: 
        frequencies[0] = 1
        frequencies = tf.Variable(frequencies,name= "frequencies",dtype= tf.float32)
        

    #other constants                     
    L_D =  tf.constant((3.086e+22*LuminosityDistance),name="LuminosityDistance",dtype=tf.float32) 
    
    #get phases at 0PN order
    phase = PN_phases(frequencies, M, eta, phase_order)
    amp = PN_amplitude(frequencies, ChirpMass, L_D)
    
    #set amplitude to zero where the waveform results are unphysical/unwanted
    cond1 = frequencies<f_low #below f_low
    ind1 = tf.where(cond1)
    cond2 = frequencies>f_ISO #above fISCO
    ind2 = tf.where(cond2)
    amp = tf.scatter_update(tf.Variable(amp), ind1, tf.constant(0.))
    amp = tf.scatter_update(tf.Variable(amp), ind2, tf.constant(0.))
    
    #Calculte waveform
    iphase = tf.multiply(tf.complex(0.,1.),tf.complex(phase-(np.pi/4)+np.pi,0.))
    waveform = tf.multiply(tf.complex(amp,0.),tf.exp(iphase))
     
    return frequencies, waveform
