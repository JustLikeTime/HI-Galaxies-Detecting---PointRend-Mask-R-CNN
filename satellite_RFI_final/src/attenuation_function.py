import scipy.signal as ss
import numpy as np

def tophat_rect(f, fi, band, level, values):
    '''
    Rectangular tophat function, https://en.wikipedia.org/wiki/Rectangular_function,
    f-frequency range;
    fi-central frequency;
    band-size of the bandwidth;
    level-value of the OOB region
    values-function values
    '''

    fx = f<fi+band
    fy = f>fi-band
    idx = np.where(fx*fy==True)[0]
    
    return np.array([x if i in idx else x*level for i,x in enumerate(values)])
