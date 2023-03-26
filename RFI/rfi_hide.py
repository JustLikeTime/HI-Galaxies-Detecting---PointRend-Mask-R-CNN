#%%

import numpy as np
from numpy import random
from numpy import concatenate, ceil, pi
from scipy.signal import fftconvolve
from datetime import timedelta
import matplotlib.pyplot as plt


class MyParams:
    def __init__(self, white_noise_scale, rfiamplitude, rfifrac, rfideltat, rfideltaf, rfiexponent, rfienhance, rfiday, rfidamping):
        self.white_noise_scale = white_noise_scale
        self.rfiamplitude = rfiamplitude
        self.rfifrac = rfifrac
        self.rfideltat = rfideltat
        self.rfideltaf = rfideltaf
        self.rfiexponent = rfiexponent
        self.rfienhance = rfienhance
        self.rfiday = rfiday
        self.rfidamping = rfidamping

def add_rfi_to_data(tod_vx, frequencies, strategy_coords, strategy_start, params):
    time = get_time(strategy_coords, strategy_start)
    
    rfi = getRFI(params.white_noise_scale, params.rfiamplitude,
                 params.rfifrac, params.rfideltat,
                 params.rfideltaf, params.rfiexponent,
                 params.rfienhance, frequencies, time, params.rfiday, params.rfidamping)
    
    tod_vx += rfi
    tod_vx_rfi = rfi
    
    return tod_vx, tod_vx_rfi


def generate_rfi(frequencies, time, params):
    rfi = getRFI(params.white_noise_scale, params.rfiamplitude,
                 params.rfifrac, params.rfideltat,
                 params.rfideltaf, params.rfiexponent,
                 params.rfienhance, frequencies, time, params.rfiday, params.rfidamping)
    
    return rfi


def get_time(strategy_coords, strategy_start):
    time = []
    for coord in strategy_coords:
        t = strategy_start + timedelta(seconds=coord.time)
        time.append(t.hour + t.minute / 60. + t.second / 3600.)
    return np.asarray(time)



# 在这里包含原始代码中的getRFI、calcRFI、getDayNightMask和其他相关函数


def getRFI(background, amplitude, fraction, deltat, deltaf, exponent, enhance,
           frequencies, time, rfiday, damping):
    """
    Get time-frequency plane of RFI.
     
    :param background: background level of data per channel
    :param amplitude: maximal amplitude of RFI per channel
    :param fraction: fraction of RFI dominated pixels per channel
    :param deltat: time scale of rfi decay (in units of pixels)
    :param deltaf: frequency scale of rfi decay (in units of pixels)
    :param exponent: exponent of rfi model (either 1 or 2)
    :param enhance: enhancement factor relative to fraction
    :param frequencies: frequencies of tod in MHz
    :param time: time of day in hours of tod
    :param rfiday: tuple of start and end of RFI day
    :param damping: damping factor for RFI fraction during the RFI night
    :returns RFI: time-frequency plane of RFI 
    """
    assert rfiday[1] >= rfiday[0], "Beginning of RFI day is after it ends."
    r = 1 - (rfiday[1] - rfiday[0]) / 24.
    nf = frequencies.shape[0]
    if (r == 0.0) | (r == 1.0):
        RFI = calcRFI(background, amplitude, fraction,
                      deltat, deltaf, exponent, enhance,
                      nf, time.shape[0])
    else:
        day_night_mask = getDayNightMask(rfiday, time)
        # Get fractions of day and night
        fday = np.minimum(1, fraction * (1 - damping * r)/(1 - r))
        fnight = (fraction - fday * (1 - r)) / r
        nday = day_night_mask.sum()
        nnight = time.shape[0] - nday
        RFI = np.zeros((nf, time.shape[0]))
        if nnight > 0:
            RFI[:,~day_night_mask] = calcRFI(background, amplitude, fnight,
                                             deltat, deltaf, exponent, enhance,
                                             nf, nnight)
        if nday > 0:
            RFI[:,day_night_mask] = calcRFI(background, amplitude, fday,
                                            deltat, deltaf, exponent, enhance,
                                            nf, nday)
    return RFI


def calcRFI(background, amplitude, fraction, deltat, deltaf, exponent, enhance,
           nf, nt):
    """
    Get time-frequency plane of RFI.
     
    :param background: background level of data per channel
    :param amplitude: maximal amplitude of RFI per channel
    :param fraction: fraction of RFI dominated pixels per channel
    :param deltat: time scale of rfi decay (in units of pixels)
    :param deltaf: frequency scale of rfi decay (in units of pixels)
    :param exponent: exponent of rfi model (either 1 or 2)
    :param enhance: enhancement factor relative to fraction
    :param nf: number of frequency channels
    :param nt: number of time steps
    :returns RFI: time-frequency plane of RFI 
    """
    lgb = np.log(background)
    lgA = np.log(amplitude)
    d = lgA - lgb
    # choose size of kernel such that the rfi is roughly an order of magnitude
    # below the background even for the strongest RFI
    Nk = int(ceil(np.amax(d))) + 3
    t = np.arange(nt)
    if exponent == 1:
        n = d * d * (2. * deltaf * deltat / 3.0)
    elif exponent == 2:
        n = d * (deltaf * deltat * pi *.5)
    else:
        raise ValueError('Exponent must be 1 or 2, not %d'%exponent)
    neff = fraction * enhance * nt / n
    N = np.minimum(random.poisson(neff, nf), nt)
    RFI = np.zeros((nf,nt))
    dt = int(ceil(.5 * deltat))
    # the negative indices really are a hack right now
    neginds = []
    for i in range(nf):
#         trfi = choice(t, N[i], replace = False)
        trfi = random.permutation(t)[:N[i]]
#         trfi = randint(0,nt,N[i])
        r = random.rand(N[i])
        tA = np.exp(r * d[i] + lgb[i])
        r = np.where(random.rand(N[i]) > .5, 1, -1)
        sinds = []
        for j in range(dt):
            fac = (-1)**j * (j + 1) * dt
            sinds.append(((trfi + fac * r) % nt))
        neginds.append(concatenate(sinds))
        RFI[i,trfi] = tA
    k = kernel(deltaf, deltat, nf, nt, Nk, exponent)
    RFI = fftconvolve(RFI, k, mode = 'same')
#     neginds = np.unique(concatenate(neginds))
#     RFI[:,neginds] *= -1
    df = int(ceil(deltaf))
    for i, idxs in enumerate(neginds):
        mif = np.maximum(0, i-df)
        maf = np.minimum(nf, i+df)
        RFI[mif:maf,idxs] *= -1
    return RFI

def getDayNightMask(rfiday, time):
    return (rfiday[0] < time) & (time < rfiday[1])

def logmodel(x, dx, exponent):
    """
    Model for the log of the RFI profile:
     * -abs(x)/dx for exponent 1
     * -(x/dx)^2 for exponent 2

    :param x: grid on which to evaluate the profile
    :param dx: width of exponential
    :param exponent: exponent of (x/dx), either 1 or 2
    :returns logmodel: log of RFI profile
    """
    if exponent == 1:
        return -np.absolute(x)/dx
    elif exponent == 2:
        return -(x * x) / (dx * dx)
    else:
        raise ValueError('Exponent must be 1 or 2, not %d'%exponent)
    
def kernel(deltaf, deltat, nf, nt, N, exponent):
    """
    Convolution kernel for FFT convolution
    
    :param deltaf: spread of RFI model in frequency
    :param deltat: spread of RFI model in time
    :param nf: number of frequencies
    :param nt: number of time steps
    :param N: size of kernel relative to deltaf, deltat
    :param exponent: exponent of RFI model (see logmodel)
    :returns kernel: convolution kernel
    """
    fmax, tmax = np.minimum([N * deltaf, N * deltat], [(nf-1)/2,(nt-1)/2])
    f = np.arange(2*fmax+1) - fmax
    t = np.arange(2*tmax+1) - tmax
    return np.outer(np.exp(logmodel(f, deltaf, exponent)), np.exp(logmodel(t, deltat, exponent)))




#%%

t_rev = 1.00663293


f_rev = 0.02
fqs = np.arange(1345,1420,f_rev)
t_rev = 1.00663293
# lsts = np.linspace(0, 124544*t_rev/3600, 11264)
lsts = np.linspace(0, 11264*t_rev/3600, 11264)


#%%


amplitude = np.zeros(615)+10
amp_list = [100, 221, 421, 600]
for ii in amp_list:
    amplitude[ii] = 10

white_noise = np.zeros(615) + 0.2


#%%

# 设置RFI参数
params = MyParams(
    white_noise_scale= white_noise, #np.array([1, 1, 1, 1]),
    rfiamplitude=amplitude, #np.array([0.1, 0.2, 0.3, 0.4]),  # 这是一个示例数组，您可以根据需要自定义
    rfifrac=0.05,
    rfideltat=5.0,
    rfideltaf=10.0,
    rfiexponent=1,
    rfienhance=1,
    rfiday=(0.0, 24.0),
    rfidamping=0
)


rfi = generate_rfi(fqs, lsts, params)

rfi = np.absolute(rfi)


# %%

print(rfi.shape)

hb = plt.imshow((rfi[0:1280, 0:1280]),
                aspect='auto')

cbar = plt.colorbar(hb)
cbar.set_label(r'T [K]', rotation=270, labelpad=20, y=0.45)


# %%
