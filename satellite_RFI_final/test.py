#%%
import sys
src_loc1 = 'e:\\satellite_RFI_new'
src_loc2 = 'e:\\satellite_RFI_new\\src'
if src_loc1 in sys.path:
    pass
else:
    sys.path.append(src_loc1)
if src_loc2 in sys.path:
    pass
else:
    sys.path.append(src_loc2)
print(sys.path)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import pickle
import time
import pytz
from datetime import datetime
from tqdm.notebook import tqdm

import astropy.constants as cc
import astropy.units as u
from astropy.time import Time

import beam_model as bm
import gnss_models_v4 as gm
import check_satellite as cs
from satellite_sims_v2 import sims as ss



#%%
time_during=5120
y=2023
m=2
d=7
h=0
minute=0
s=0

t_adv=1.00663296
freq_resolution= 0.00762939453125
freq_start=1330
freq_end=freq_start + 70*freq_resolution

sats="all"

print('start @ ' + time.asctime(time.localtime(time.time())) +'...', "\n")

freq_res = freq_resolution
freq_range = np.arange(freq_start, freq_end, freq_res)
time_inds_range = np.arange(0, time_during, t_adv)


obs_time_1 = datetime(y, m, d, h, minute, s)

print("Time start at: ", obs_time_1, "\n")

time2 = Time(obs_time_1) + time_inds_range*u.second

print("Time end at: " , time2[-1], "\n")
print("Time series shape is ", time2.shape , "\n")

#%%
sats_type = ['geo', 'glo-ops', 'gps-ops', 'beidou'] 
msc = cs.FASTsite_Satellite_Catalogue(sats_type=sats_type, reload=False)
obs_time = time2[0]
obs_time.format = 'fits'
msc.obs_time = obs_time.value
msc.obs_time_list = (time2.unix - time2[0].unix) * u.second


print(msc.obs_time_list)
# %%
t_adv=1.00663296
freq_resolution= 0.00762939453125
freq_start=1330
freq_end=freq_start + 700*freq_resolution

#fname = '1675728000_' + '%d' + '-' +'%f' % (freq_start, freq_end)

fname = '1675728000_%d-%d' % (freq_start,freq_end)

print(fname)

# %%

time_during=5120
t_adv=1.00663296
time_inds_range = np.arange(0, time_during*t_adv, t_adv)


print(time_inds_range.shape)


# %%


freq=np.array([1400,1420])
speed_of_light=299792458   # speed of light in m/s
lamb=(speed_of_light/freq/1.0e6) # wavelength in m
dish_diameter=13.5 # diameter of telescope dish in meters (m)
fwhm=1.16*np.degrees(lamb/dish_diameter) # FWHM in degrees

fwhm  = fwhm[:, None, None]

print(fwhm)

# %%
a = np.array([1])
a = a[:, None, None]

A=4
B=1.2

print(a)

print(a.shape)

def calcu(x):
    y = (np.cos( A*np.pi*x[None, ...]/fwhm)/(1-B*(A*x[None, ...]/fwhm)**2))**2
    return y

def calcu1(x):
    freq = np.array([1420,1350])
    print(freq)
    HPBW = 1.22*300000000*1.e-6/freq/300.*180./np.pi
    print(HPBW)
    sigma = HPBW/(2*np.sqrt(2*np.log(2)))
    sigma = sigma[:, None, None]
    print(sigma)
    y=np.exp(-x**2/2.0/sigma**2)
    return y

x = np.array([[0,0.5],[0.1,0.3]])

calcu1(x)

# %%

def Cosine_beam_model0(freq):

    speed_of_light=299792458   # speed of light in m/s
    lamb=(speed_of_light/freq/1.0e6) # wavelength in m
    dish_diameter=13.5 # diameter of telescope dish in meters (m)
    fwhm=1.16*np.degrees(lamb/dish_diameter) # FWHM in degrees

    fwhm  = fwhm[:, None, None]

    A=1.189
    B=4
    return lambda x: (np.cos( A*np.pi*x[None, ...]/fwhm)/(1-B*(A*x[None, ...]/fwhm)**2))**2

z = np.array([1420,1420])

beam_func = Cosine_beam_model0(freq=z)

x = np.array([[20,30],[10,30]])

y = beam_func(x)

print(y)

# %%
freq = 1420
speed_of_light=299792458   # speed of light in m/s
lamb=(speed_of_light/freq/1.0e6) # wavelength in m
dish_diameter=13.5 # diameter of telescope dish in meters (m)
fwhm=1.16*np.degrees(lamb/dish_diameter) # FWHM in degrees

HPBW = 1.22*299792458*1.e-6/freq/300.*180./np.pi

print(fwhm)
print(HPBW)
# %%
freq_resolution= 0.00762939453125
freq_channel=5000
freq_start=1330
freq_end = None

freq_res = freq_resolution

if freq_end is None:
    if freq_channel is None:
        raise ValueError('No freq information!')
    else:
        freq_end = freq_start + freq_channel*freq_res

freq_band = np.arange(freq_start, freq_end, freq_res)

print("frequency band is: ", freq_band, '\n')
print("frequency band's shape is: ", freq_band.shape, '\n')
# %%

freq_resolution= 0.00762939453125
freq_channel=5000
freq_start=1330
freq_end = None

freq_res = freq_resolution

k = freq_channel*freq_res
print(k)

freq_band = np.arange(freq_start, freq_end, freq_res)

print(freq_band)

# %%

time_during=5120
y=1997
m=2
d=7
h=0
minute=0
s=0


obs_time = datetime(y, m, d, h, minute, s)
fname = int((obs_time - datetime(1970, 1, 1)).total_seconds())

print(fname)

# %%

y=2023
m=2
d=7
h=0
minute=0
s=0

freq_start=1330
freq_end=None
freq_channel=[0,64]
time_during=[0,5120]

t_adv=1.00663296
freq_resolution= 0.00762939453125

sats="all"
beam='FAST'
data_loc='e:/satellite_RFI_new/'
sats_type = ['geo', 'glo-ops', 'gps-ops', 'beidou'] 

time_inds_range = np.arange(time_during[0]*t_adv, time_during[-1]*t_adv, t_adv)
time_mark_0 = datetime(y, m, d, h, minute, s)
time_obs_band = Time(time_mark_0) + time_inds_range*u.second
time_obs_start_unix = time_obs_band[0].unix

msc = cs.FASTsite_Satellite_Catalogue(sats_type=sats_type, reload=False)
time_obs_start = time_obs_band[0]
time_obs_start.format = 'fits'
msc.obs_time = time_obs_start.value
msc.obs_time_list = (time_obs_band.unix - time_obs_band[0].unix) * u.second

print(time_obs_start)
print(msc.obs_time)
print(msc.obs_time_list)

time_list_num=np.array(msc.obs_time_list)

print(time_list_num)

# %%

def FAST_beam_model(freq):

    speed_of_light=299792458
    HPBW = 1.22*speed_of_light*1.e-6/freq/300.*180./np.pi

    HPBW  = HPBW[:, None, None]

    sigma = HPBW/(2*np.sqrt(2*np.log(2)))

    return lambda x: np.exp(-x**2/2.0/sigma**2)


def Cosine_beam_model(freq):

    speed_of_light=299792458   # speed of light in m/s
    lamb=(speed_of_light/freq/1.0e6) # wavelength in m
    dish_diameter=13.5 # diameter of telescope dish in meters (m)
    fwhm=1.16*np.degrees(lamb/dish_diameter) # FWHM in degrees

    fwhm  = fwhm[:, None, None]

    A=1.189
    B=4
    return lambda x: (np.cos( A*np.pi*x[None, ...]/fwhm)/(1-B*(A*x[None, ...]/fwhm)**2))**2

freq_resolution= 0.00762939453125
freq_start=1330
freq_end = 1360

freq_band = np.arange(freq_start, freq_end, freq_resolution)

Gauss_beam = FAST_beam_model(freq_band)
Cosine_beam = Cosine_beam_model(freq=freq_band)

obs_time_1 = datetime(2023, 2, 7, 0, 0, 0)
time2 = Time(obs_time_1) + np.arange(0, 100, 1) * u.second
print(time2.shape)
pointings = np.array([[0, 0],]*time2.shape[0])
print(pointings.shape)

k1 = Gauss_beam(pointings)
k2 = Cosine_beam(pointings)

print(k1.shape)
print(k1)

print(k2.shape)
print(k2)

# %%
