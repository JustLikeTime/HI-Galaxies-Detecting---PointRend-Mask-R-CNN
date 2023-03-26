import numpy as np
from astropy.io import fits
from scipy.interpolate import interp1d, interp2d
from astropy import constants as const

# from scipy.interpolate import RegularGridInterpolator
# from scipy.interpolate import interp1d
# #from scipy import constants as const
# from scipy.integrate import dblquad
# from scipy import special as spf
# from multiprocessing import Pool
# from functools import partial
# from astropy.cosmology import Planck15 as plk15
# from astropy.cosmology import z_at_value
# from astropy import units
# from astropy import constants as const
# from scipy.signal import convolve
# from scipy import linalg as la
# import warnings
# from scipy.spatial.transform.rotation import Rotation
# from scipy.sparse.linalg import splu
# from scipy.sparse import csc_matrix


def _Khans_beam_model(phi=None):

    beam_path = '/idia/users/ycli/meerkat_beam/'
    beam_file = 'primary_beam_mh_184channels_10deg_re.fits'

    with fits.open(beam_path + beam_file) as hdul:
        beam_data = hdul[0].data
        
    _theta = np.linspace(-5, 5, 256, dtype='float32')
    _freq  = np.arange(800, 1720, 5, dtype='float32')
    
    if phi is None:
        beam_profile = beam_data[:, 0, 0, ...] ** 2 + beam_data[:, 1, 1, ...] ** 2
        beam_profile /= 2.

        mid_idx = beam_profile.shape[1] / 2
        beam_HH = beam_profile[:, :, mid_idx]
        beam_VV = beam_profile[:, mid_idx, :]
        beam_mean = 0.5 * (beam_HH + beam_VV)
    elif phi == 'HH':
        beam_profile = beam_data[:, 0, 0, ...] ** 2
        mid_idx = beam_profile.shape[1] / 2
        beam_HH = beam_profile[:, :, mid_idx]
        #beam_HH = beam_profile[:, mid_idx, :]
        beam_mean = beam_HH
    elif phi == 'VV':
        beam_profile = beam_data[:, 1, 1, ...] ** 2
        mid_idx = beam_profile.shape[1] / 2
        beam_VV = beam_profile[:, mid_idx, :]
        beam_mean = beam_VV

    beam_mean = beam_mean.astype('float32')

    return beam_mean, _freq, _theta

    #if axis == 1:
    #    beam_intf  = interp1d(_theta, beam_mean, axis=1,bounds_error=False, fill_value=0)
    #    return beam_intf, _freq

    #elif axis == 0:
    #    beam_intf  = interp1d(_freq, beam_mean, axis=0,bounds_error=False, fill_value=0)
    #    return beam_intf, _theta

def Khans_beam_model(freq=None, theta=None):

    beam_mean, _freq, _theta = _Khans_beam_model()

    beam_func = interp2d(_theta, _freq, beam_mean)

    kwargs = { 'bounds_error' :False, 'fill_value': 0 }

    if freq is not None:
        _beam_func = interp1d(_theta, beam_func(_theta,  freq).astype('float32'), axis=1, **kwargs)
        return lambda x: _beam_func(x)
    elif theta is not None:
        _beam_func = interp1d(_freq,  beam_func(theta,  _freq).astype('float32'), axis=0, **kwargs)
        return lambda x: _beam_func(x)
    else:
        return beam_func






def get_OmegaA_from_Khans_beam(freq, threshold=0.):

    beamf, _freq = Khans_beam_model()
    d_theta = 0.01
    theta = np.arange(0, 5, d_theta)
    p = beamf(theta)
    p[p<threshold] = 0.
    theta *= np.pi / 180.
    d_theta *= np.pi / 180.
    omega_a = 2. * np.pi * np.sum(p * np.sin(theta)[None, :] * d_theta, axis=1)
    omega_a_f = interp1d(_freq, omega_a)

    return omega_a_f(freq)


def get_fwhm_from_Khans_beam(freq):

    beam, _t = Khans_beam_model(axis=0)
    _freq = np.arange(800, 1720, 10)
    width = lambda x, f: interp1d(beam(f)[128:], _t[128:])(x)
    _fwhm = []
    for _f in _freq:
        _fwhm.append(width(0.5, _f))
    _fwhm = np.array(_fwhm)
    
    return interp1d(_freq, _fwhm)(freq) * 2.

def get_factor_from_Khans_beam(freq, angle=0.5, phi=None):

    beam_intf, _freq = Khans_beam_model(phi=phi)
    
    factor = interp1d(_freq, beam_intf(angle))
    
    return factor(freq)



def Cosine_beam_model(freq):

    speed_of_light=299792458   # speed of light in m/s
    lamb=(speed_of_light/freq/1.0e6) # wavelength in m
    dish_diameter=13.5 # diameter of telescope dish in meters (m)
    fwhm=1.16*np.degrees(lamb/dish_diameter) # FWHM in degrees

    fwhm  = fwhm[:, None, None]

    A=1.189
    B=4
    return lambda x: (np.cos( A*np.pi*x[None, ...]/fwhm)/(1-B*(A*x[None, ...]/fwhm)**2))**2


def Cosine_beam_model_1D(freq):

    speed_of_light=299792458   # speed of light in m/s
    lamb=(speed_of_light/freq/1.0e6) # wavelength in m
    dish_diameter=13.5 # diameter of telescope dish in meters (m)
    fwhm=1.16*np.degrees(lamb/dish_diameter) # FWHM in degrees

    A=1.189
    B=4
    return lambda x: (np.cos( A*np.pi*x/fwhm)/(1-B*(A*x/fwhm)**2))**2

def FAST_HPBW(freq):
    '''
    freq in MHz
    Half Power BandWidth need to be updated using data from
    Jiang, Peng, et al. "The fundamental performance of FAST with 19-beam receiver at L band." Research in Astronomy and Astrophysics 20.5 (2020): 064.
    '''
    HPBW = np.array([1.22*const.c.value*1.e-6/freq/300.*180./np.pi*60]*19)
    return HPBW

def FAST_beam(freq):

    HPBW = 1.22*const.c.value*1.e-6/freq/300.*180./np.pi

    HPBW  = HPBW[:, None, None]

    sigma = HPBW/(2*np.sqrt(2*np.log(2)))

    return lambda x: np.exp(-x**2/2.0/sigma**2)