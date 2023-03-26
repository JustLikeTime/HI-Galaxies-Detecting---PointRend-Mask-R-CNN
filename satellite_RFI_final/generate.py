#%%
import sys
src_loc1 = 'e:\\satellite_RFI_new'
src_loc2 = src_loc1 + '\\src'
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

def satelite_timestream(frequency, pointing_position, msc, beam):
    '''
    Produces the angular seperation plots for the freqeuncy given and pointing
    Uses some global variabes like 
    Frequency - ini, end, channel
    生成给定频率和指向的角度分离图
    使用一些全局变量，比如
    Frequency - ini, end, channel'''
    
    ## Code from satellite_timestream
    freq = frequency  
#     beam_func = beam_model.Khans_beam_model(freq=freq)
    if beam == 'FAST':
        beam_func = bm.FAST_beam(freq=freq)       # using FAST beam model
    else:
        beam_func = bm.Cosine_beam_model(freq)       # using cosine beam model       # using cosine beam model

    stype, sname, stemperature = [], [], []

    for _sat_info in msc.itersats_temperature(pointings=pointing_position, beam_func=beam_func):
        _time_list, _sat_type, _sat_names, _temperature = _sat_info
        print('Sat. Name includes %s ...'%_sat_names[0])# includes the name of each sat
        print('N_freq x N_time x N_sats = %d x %d x %d '%_temperature.shape)
        print
        np.array(stype.append(_sat_type)), sname.append(_sat_names), stemperature.append(np.sum(_temperature[:,:,:], axis=2))

        del _temperature 
    return stype, sname, stemperature


#%%

def generate(y=2023, m=2, d=7, h=0, minute=0, s=0, 
             freq_start=1330, 
             freq_end=None, 
             freq_channel=[0,512], 
             time_during=[0,5120], 
             t_adv=1.00663296, 
             freq_resolution= 0.00762939453125, 
             sats="all", 
             beam='FAST',
             data_loc='E:/satellite_RFI_new/'):

    print('start @ ' + time.asctime(time.localtime(time.time())) +'...', "\n", "\n")


    #按 |频率起点、频率channel个数、频率分辨率| 或者 |频率起点、频率终点、频率分辨率| 生成观测频段
    freq_res = freq_resolution
    if freq_end is None:
        if freq_channel is None:
            raise ValueError('No freq information!')
        else:
            freq_start = freq_start + freq_res*freq_channel[0]
            freq_end = freq_start + freq_res*freq_channel[1]
    freq_band = np.arange(freq_start, freq_end, freq_res)

    print("frequency band start at: ", freq_band[0])
    print("frequency band end at: ", freq_band[-1])
    print("frequency band's shape is: ", freq_band.shape, '\n')


    #按 |时间起点、时序个数、时间分辨率| 生成观测时列
    time_inds_range = np.arange(time_during[0]*t_adv, time_during[-1]*t_adv, t_adv)
    time_mark_0 = datetime(y, m, d, h, minute, s)
    time_obs_band = Time(time_mark_0) + time_inds_range*u.second
    time_obs_start_unix = time_obs_band[0].unix

    print("Time start at: ", time_obs_band[0], ', also ', time_obs_start_unix)
    print("Time end at: ", time_obs_band[-1])
    print("Time series shape is  ", time_obs_band.shape)


    #生成望远镜指向，暂定不动
    #pointing =  np.array([[nd_s0_coords[0][i], nd_s0_coords[1][i]] for i in range(len(nd_s0_coords[0]))])

    pointings = np.array([[0, 0],]*time_obs_band.shape[0])
    print("Pointings'shape is  ", pointings.shape, "\n")


    #确定卫星种类
    if sats == "all":
        #sats_type = ['geo', 'gps-ops', 'glo-ops', 'galileo', 'beidou'] 
        sats_type = ['geo', 'glo-ops', 'gps-ops', 'beidou'] 
    else:
        sats_type = sats


    #调用生成卫星位置信息的类
    msc = cs.FASTsite_Satellite_Catalogue(sats_type=sats_type, reload=False)
    time_obs_start = time_obs_band[0]
    time_obs_start.format = 'fits'
    msc.obs_time = time_obs_start.value
    msc.obs_time_list = (time_obs_band.unix - time_obs_band[0].unix) * u.second
    msc.get_sate_coords()
    #msc.check_altaz()
    time_list_num=np.array(msc.obs_time_list)

    print(' ')
    print("msc.obs_time_list shape is ", time_list_num.shape)
    print("msc.obs_time_list start at ", time_list_num[0,0])
    print("end at ", time_list_num[0,-1], "\n")

    satellite_angle = msc.check_angular_separation(pointings=pointings, max_angle=180, beam_func=None, ymin=1e-10, ymax=90, axes=None)


    #选择Gaussian beam或者cosine beam
    if beam == 'FAST':
        beam_func = bm.FAST_beam(freq=freq_band)       # using FAST beam model
    else:
        beam_func = bm.Cosine_beam_model(freq=freq_band)       # using cosine beam model


    #根据所用的beam生成给定频率和指向的角距图
    sat_type, sat_names, stemperature = satelite_timestream(frequency=freq_band, pointing_position=pointings, msc=msc, beam=beam)

    print("satellite type is : ", sat_type)
    print("satellite angel is : ", stemperature)


    #存储数据
    sat_pos = {
    'sat_name': sat_type,
    'angular': stemperature 
    }

    data_save = data_loc + 'Angular_Position/'
    fname = '%d_time%d-%d_freq%d-%d' % (time_obs_start_unix, time_during[0], time_during[-1], freq_start, freq_end)

    pickle.dump(sat_pos, open(data_save+fname+"_satellite_angular_position.p", "wb"))

    print(' ')
    print('data saved at:  ' + data_save + fname + "_satellite_angular_position.p", "\n")


    print('end @ ' + time.asctime(time.localtime(time.time())) +'#')


#%%

y=2023
m=2
d=7
h=0
minute=0
s=0

freq_start=1330
freq_end=None
freq_channel=[0,1024]
time_during=[0,512]

t_adv=1.00663296
freq_resolution= 0.00762939453125

sats="all"
beam='FAST'
data_loc='E:/satellite_RFI_new/'


#%%

generate(y=y, m=m, d=d, h=h, minute=minute, s=s,
         freq_start=freq_start, 
         freq_end=freq_end, 
         freq_channel=freq_channel, 
         time_during=time_during, 
         t_adv=t_adv, 
         freq_resolution= freq_resolution, 
         sats=sats, 
         beam=beam,
         data_loc=data_loc)


#%%

s_now_time = time.time()


sal_1 = ss(data_loc = data_loc)

sal_1.start(y=y, m=m, d=d, h=h, minute=minute, s=s,
         freq_start=freq_start, 
         freq_end=freq_end, 
         freq_channel=freq_channel, 
         time_during=time_during, 

         t_adv=t_adv, 
         freq_resolution= freq_resolution, 

         file_bias_choice=[30,1,10,200,1500,2,0], 
         add_sub=[1, 1], 
         band_lvl=[25, 0.001])


print (time.time() - s_now_time)

#%%

sal_1.plotting(individual=None, logger=-2, axis_limit=[1100, 1400, 10, 250], tod_limit=None, save_file=None, file_type='png')


# %%
