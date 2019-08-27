def pyPN_phases(freq,mTot,eta,phase_order):
    "produce phases at 0PN order"
    
    #constants in equation
    mtsun = 4.925491025543575903411922162094833998e-6
    piM = mTot*np.pi*mtsun # pi multiplyed by total mass in seconds
    v = (piM*freq)**(1/3) #Characteristic Velocity of Binary
    etasq = eta*eta # square of the symmetric mass ratio
    etacb= eta*eta*eta # cube of the symmetric mass ratio
    gamma = 0.577215664901532 #Euler–Mascheroni constant
    theta = -11831./9240.
    lambdaa = -1987./3080.0
    
    
    #Generate V terms
    V0 = v**(-5)
    V2 = v**2
    V3 = v*V2
    V4 = v*V3
    V5 = v*V4
    Vlog5 = np.log(v)
    V6 = v*V5
    Vlog6 = np.log(4*v)
    V7 = v*V6
    
    # generate phase coefficients
    P0 = (3/128)*(1/eta)
    P2 = (743/84+(11*eta))*(5/9)
    P3 = (-16*np.pi)
    P4 = 10*((3058673/1016064)+((5429/1008)*eta)+((617/144)*etasq))
    P5 = 5.0/9.0 * (7729.0/84.0 - 13.0 * eta) * np.pi 
    Pl5 = 5.0/3.0 * (7729.0/84.0 - 13.0 * eta) * np.pi 
    P6 = (11583.231236531/4.694215680 - 640.0/3.0 * np.pi * np.pi- \
            6848.0/21.0*gamma) + \
            eta * (-15335.597827/3.048192 + 2255./12. * np.pi * \
            np.pi - 1760./3.*theta +12320./9.*lambdaa) + \
            etasq * 76055.0/1728.0 - \
            etacb*  127825.0/1296.0
    Pl6 = -6848.0/21.0
    P7 = np.pi*((77096675/254016)+((378515/1512)*eta)+((-74045/756)*etasq))
    
    #Generetate PN terms
    
    PN0 = P0*V0
    PN2 = P2*V2
    PN3 = P3*V3
    PN4 = P4*V4
    PN5 = (P5+Pl5*Vlog5)*V5
    PN6 = (P6+Pl6*Vlog6)*V6
    PN7 = P7*V7
    
    #phases = PN_cnst + PN0*(1+PN2+PN3+PN4+PN5+PN6+PN7)
    if phase_order == 7:
        phases = PN0*(1+PN2+PN3+PN4+PN5+PN6+PN7)
    elif phase_order == 6:
        phases = PN0*(1+PN2+PN3+PN4+PN5+PN6)
    elif phase_order == 5:
        phases = PN0*(1+PN2+PN3+PN4+PN5)
    elif phase_order == 4:
        phases = PN0*(1+PN2+PN3+PN4) 
    elif phase_order == 3:
        phases = PN0*(1+PN2+PN3)
    elif phase_order == 2:
        phases = PN0*(1+PN2)
    elif phase_order == 1:
        phases = PN0
    else:
        phases = PN0
    
    return phases

def pyPN_amplitude(freq,ChirpMass,LuminosityDistance):
    "produce amplitude at 0PN order"
    
    return (299792458/LuminosityDistance)*(((5/24)**0.5)*(np.pi**(-2/3)))*((ChirpMass*4.92549102554e-6)**(5/6))*(freq**(-7/6))

def pytaylorf2(mass1,mass2,frequencies=None,LuminosityDistance=1.,f_low=10.,
               df=1./512, f_high=1600., phase_order=0):
    
    #genrate characteristic masses
    M = mass1+mass2
    eta = (mass1*mass2)/(M**2)
    ChirpMass = M*(eta**(3/5)) #Chirp mass is given by M*(eta^3/5)
    
    #define frequencies
    f_ISO = 1/(6.**1.5*np.pi*(M)*4.92549102554e-6) #frequency of innermost stable circular orbit (SI units)
    if frequencies is None:
        f = np.arange(f_low,f_ISO,df)
        freq = f
    else: 
        frequencies[0] = 1
        freq = frequencies 
        
    #Luminosity distance in SI Units
    L_D =  3.086e+22*LuminosityDistance
    
    #Evaluate phases and amplitudes across frequencies
    phase = pyPN_phases(freq,M,eta,phase_order)
    amp = pyPN_amplitude(freq,ChirpMass,L_D)
    
    #set amplitude to zero where the waveform results are unphysical or unwanted
    np.array(amp)[np.array(np.argwhere(freq<f_low))] = 0.
    np.array(amp)[np.array(np.argwhere(freq>f_ISO))] = 0.
    
    #Produce waveform
    waveform = amp*np.exp(1j*(phase+(3/4)*np.pi))
    
    return freq, waveform
