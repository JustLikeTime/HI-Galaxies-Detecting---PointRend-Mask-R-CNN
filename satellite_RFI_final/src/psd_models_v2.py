'''
2020/11/15
Contains the Power Spectrum Density Models for the GNSS satellites.
Informationg comes primarly from <Springer Handbook of GNSS; P.J.G Teunissen pg 100
'''

import numpy as np

def BPSK(f, n_c, f0=1.023):
    '''
    Binary Phase Switch Keying PSD pg 107
    n_c - chip rate
    f0 - reference frequency
    '''
    t_c = 1/(n_c*f0)
    a = np.sqrt(t_c) * np.sinc(f * t_c)
    return a**2

#--------------------------------------------------------------------------------------#

def BOC(f, n_c, n_s, f0=1.023):
    '''
    Binary Offset Carrier found in Springer pg110
    f - frequency
    n_s - sub-carrier rate
    n_c - chip rate
    f0 - reference frequency
    '''
    
    dfrac = f[1] -f[0]
    
    f1 = f - 0.5*dfrac
    f2 = f + 0.5*dfrac
    
    def calc_even(f, n_s=n_s, n_c=n_c, f0=f0):
        '''Even calculation for the data Eq 4.97'''
        a = np.sqrt(n_c * f0)
        b = np.sin(np.pi * f / n_c / f0) / np.pi / f
        c = np.tan(np.pi * f / 2 / n_s / f0)
        return (a*b*c)**2
    
    def calc_odd(f, n_s=n_s, n_c=n_c, f0=f0):
        '''Odd calculation for the data Eq 4.98'''
        a = np.sqrt(n_c * f0)
        b = np.cos(np.pi * f / n_c / f0) / np.pi / f
        c = np.tan(np.pi * f / 2 / n_s / f0)

        return (a*b*c)**2
    
    # Check, find the source
    if n_s>=n_c:
    
        #Checking if odd or even
        N = 2.*n_s/n_c
#         print N
        if N%2==0:
            E1 = calc_even(f=f1)
            E2 = calc_even(f=f2)
#            print 'Even'

            psd = 0.5*(E1+E2)

        else:
            O1 = calc_odd(f=f1)
            O2 = calc_odd(f=f2)
#            print 'Odd'

            psd = 0.5*(O1+O2)
    else:
#         print 'n_c > n_s: can not be'
        psd = np.zeros(len(f))
        
    return psd
        
#--------------------------------------------------------------------------------------#            

def BOCc(f, n_c, n_s, f0=1.023):
    '''
    Cosine Binary Offset Carrier found in Springer pg110
    f - frequency
    n_s - sub-carrier rate
    n_c - chip rate
    f0 - reference frequency
    '''
    
    dfrac = f[1] -f[0]
    
    f1 = f - 0.5*dfrac
    f2 = f + 0.5*dfrac
    
    def calc_even(f, n_s=n_s, n_c=n_c, f0=f0):
        '''Even calculation for the data Eq 4.99'''
        a = np.sqrt(n_c * f0)
        b = np.sin(np.pi * f / n_c / f0) / np.pi /f
        c = (1 - np.cos(np.pi * f / 2/ n_s / f0)) / np.cos(np.pi * f / 2 / n_s / f0)
        
        return (a*b*c)**2
    
    def calc_odd(f, n_s=n_s, n_c=n_c, f0=f0):
        '''Odd calculation for the data Eq 4.100'''
        a = np.sqrt(n_c * f0)
        b = np.cos(np.pi * f / n_c / f0) / np.pi / f
        c = (1 - np.cos(np.pi * f / 2 / n_s / f0)) / np.cos(np.pi * f / 2 / n_s / f0)

        return (a*b*c)**2
    
    # Check, find the source
    if n_s>=n_c:
    
        # Checking if odd or even
        N = 2.*n_s/n_c
#         print N
        if N%2==0:
            E1 = calc_even(f=f1)
            E2 = calc_even(f=f2)
#             print 'Even'

            psd = 0.5*(E1+E2)

        else:
            O1 = calc_odd(f=f1)
            O2 = calc_odd(f=f2)
#             print 'Odd'

            psd = 0.5*(O1+O2)
    else:
#         print 'n_c > n_s: can not be'
        psd = np.zeros(len(f))
        
    return psd
        
#--------------------------------------------------------------------------------------#

# Missing AltBOC - Still require new version

def altBOC(f, n_c, n_s, f0):
    '''
    Binary Offset Carrier (BOC). The altBOC name refers to use a QPSK multiplexing signal.
    The chip size of the BPSK signal (Tc) is larger and an integer value of the
    carrier signal chap size (Ts). Tc > Ts
    https://gssc.esa.int/navipedia/index.php/AltBOC_Modulation.
    n > m
    (Tc > Ts)
    '''


    phi = 2 * n_s / n_c 
    alp = 4*n_c*f0 / (np.pi*f)**2
    div = np.cos(np.pi*f/2/n_s/f0)**2
    beta = np.cos(np.pi*f/2/n_s/f0)**2 - np.cos(np.pi*f/2/n_s/f0) - 2*np.cos(np.pi*f/2/n_s/f0)*np.cos(np.pi*f/4/n_s/f0) + 2

    
    if np.mod(phi, 2) == 0: # Phi-even spectrum
        num = np.sin(np.pi*f/n_c/f0)**2

    else:
        num = np.cos(np.pi*f/n_c/f0)**2
        
    return alp * (num / div) * beta



#--------------------------------------------------------------------------------------#    

# Missing MBOC - Still require new version

def MBOC(f, m, f0=1.023):
    '''
    https://en.wikipedia.org/wiki/Multiplexed_binary_offset_carrier
    https://gssc.esa.int/navipedia/index.php/MBOC_Modulation
    '''

    fc = m * f0

    A = fc / 11 / np.pi**2 / f**2
    B = np.sin(np.pi*f/fc)**2
    C = 10*np.tan(np.pi*f/2/fc)**2
    D = np.tan(np.pi*f/12/fc)**2

    return A*B*(C+D)

#--------------------------------------------------------------------------------------#

# TMBOC

def TMBOC(f, n_c, n_s, f0, ratio):
    '''
    Time Multiplexed BOC, same as CBOC?
    https://gssc.esa.int/navipedia/index.php/Time-Multiplexed_BOC_(TMBOC)
    See BOC for information guide.
    Ratio - the ratio in the power values
    '''
    
    diff_ratio = 1. - ratio
    
    t1 = diff_ratio * BOC(f=f, n_c=n_c, n_s=1, f0=f0)
    t2 = ratio * BOC(f=f, n_c=1, n_s=n_s, f0=f0)
    
    return t1+t2
    
#--------------------------------------------------------------------------------------#    

# TMBOC

def CBOC(f, n_c, n_s, f0, ratio):
    '''
    Composite BOC
    See BOC for information guide.
    Ratio - the ratio in the power values
    https://www.dlr.de/kn/Portaldata/27/Resources/dokumente/06_projekte/Loh_PLANS2010.pdf  Eq.3
    
    '''
    
    diff_ratio = 1. - ratio
    
    t1 = np.sqrt(diff_ratio) * BOC(f=f, n_c=n_c, n_s=1, f0=f0)
    t2 = np.sqrt(ratio) * BOC(f=f, n_c=1, n_s=n_s, f0=f0)
    
    return t1+t2   

#--------------------------------------------------------------------------------------#    
