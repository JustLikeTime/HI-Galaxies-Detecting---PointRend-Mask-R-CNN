import pandas as pd
import numpy as np
import scipy as sp
import astropy.constants as cc
import psd_models_v2 as psd
from fractions import Fraction
import attenuation_function as af
#-------------------------------------------------------------_#

def floaty(x):
    '''
    In case the ratio becomes a pickle to deal with
    '''
    try:
        a = float(x)
    except ValueError:
        a = float(Fraction(x))

    return a

#-------------------------------------------------------------_#

def gnss_satellites(name_gnss, frequency_gnss, excel_sat_info, band_lvl, excel_loc=None):
    '''Returns the Spectral Energy Density of the GNSS and the Data file that we used as an input
    name_gnss - Satellite name
    frequency_gnss - Frequency list of satellites
    excel_sat_info - The satellite excel cataloguen name in the s3 Notebook folder
    band_lvl- the bandwidth and level of the drop
    
    '''
    # Distances to the satellite constellations, taken from Springer Handbook pg 1234 

    if name_gnss=='gps-ops':
        name = 'GPS'
        r = 20180 * 1000 # Distance in metres

    elif name_gnss=='glo-ops':
        name = 'GLO'
        r = 19130 * 1000 # Distance in meters

    elif name_gnss=='galileo':
        name = 'GAL'
        r = 23220 * 1000 # Distance in metres

    elif name_gnss=='beidou':
        name = 'BDS'
        r = 21530 * 1000 # Distance in meters

    elif name_gnss=='irnss':
        name = 'IRNSS'
        r = 35790 * 1000  # Distance in meters

    elif name_gnss=='qzs':
        name = 'QZS'
        r = 35790 * 1000  # Distance in meters

    elif name_gnss=='sbas':
        name = 'SBAS'
        r = 35790 * 1000  # Distance in meters

    else:
        print ('Oops, make the sure name is written as GPS or Galileo or GLONASS')
        return -1

    if excel_loc==None:
        data = pd.read_csv(excel_sat_info, header=0, engine='python')   # Excel data file with all the GNSS and models Table 2
    else:
        data = pd.read_csv(excel_loc+excel_sat_info, header=0, engine='python')   # Excel data file with all the GNSS and models Table 2


    # Re-ordering by frequency
#         data = data.sort_values(by = 'Frequency[MHz]')
#-----# Looking at all frequency below 1500 MHz  超过1500MHz的去掉了
    data = data[data['Frequency[MHz]']< 1500]
#------# Changing the rate values
#         data.loc[data['Rate(MHz)'] != 1.023, 'Rate(MHz)'] = 1.023



    # Extracting smaller data table
    data_sub = data[data[data.columns[0]].str.contains(name)]

    # Making the Spectral Energy density list
    sed = []

    # Looping through all the sub-data index
    for i in data_sub.index:
        if data_sub['P_t (dBW)'][i]==0 or data_sub['G_t (dBi)'][i]==0:
            power = 0
        else:
            power = 10**(data_sub['P_t (dBW)'][i]/10) * 10**(data_sub['G_t (dBi)'][i]/10) / (4*np.pi*r**2) 

        if 'BPSK(' in data_sub['Modulation'][i]:
            s = data_sub['Modulation'][i]
            T_c = float(s[s.find("(")+1:s.find(")")])   # Getting the chip interval value
            model = psd.BPSK(f=frequency_gnss - data_sub['Frequency[MHz]'][i], n_c=T_c,
                             f0=data_sub['Rate(MHz)'][i] / T_c)

#                 print T_c, data_sub['Rate(MHz)'][i] / T_c

        elif 'BOC(' in data_sub['Modulation'][i]:
            try:
                s = data_sub['Modulation'][i]
                # Getting the values between parenthesis and converting to float
                T_s, T_c = [float(x) for x in s[s.find("(")+1:s.find(")")].split(',')]
                model = psd.BOC(f=frequency_gnss - data_sub['Frequency[MHz]'][i], n_c=T_c,
                                 n_s=T_s, f0=data_sub['Rate(MHz)'][i] / T_c)
            # Bad method of by-passing TMBOC values

#                     print T_s, T_c, data_sub['Rate(MHz)'][i] / T_c

            except ValueError as UnboundLocalError:
                pass


        elif 'BOCcos(' in data_sub['Modulation'][i]:
            s = data_sub['Modulation'][i]
            # Getting the values between parenthesis and converting tp float
            T_s, T_c = [float(x) for x in s[s.find("(")+1:s.find(")")].split(',')]
            model = psd.BOCc(f=frequency_gnss - data_sub['Frequency[MHz]'][i], n_c=T_c,
                             n_s=T_s, f0=data_sub['Rate(MHz)'][i] / T_c)

#                 print T_s, T_c, data_sub['Rate(MHz)'][i] / T_c

        elif 'AltBOC(' in data_sub['Modulation'][i]:
            # Going to also catch TD-AltBOC
            s = data_sub['Modulation'][i]
            # Getting the values between parenthesis and converting to float
            T_s, T_c = [float(x) for x in s[s.find("(")+1:s.find(")")].split(',')]
            model = psd.altBOC(f=frequency_gnss - data_sub['Frequency[MHz]'][i], m=T_s,
                               n=T_c, f0=data_sub['Rate(MHz)'][i] / T_c)

#                 print T_s, T_c, data_sub['Rate(MHz)'][i] / T_c

        elif 'TMBOC(' in data_sub['Modulation'][i]:
            s = data_sub['Modulation'][i]
            # Getting the values between parenthesis and converting to float
            T_s, T_c, rt = [floaty(x) for x in s[s.find("(")+1:s.find(")")].split(',')]
            model = psd.TMBOC(f=frequency_gnss - data_sub['Frequency[MHz]'][i], n_c=T_c,
                             n_s=T_s, f0=data_sub['Rate(MHz)'][i] / T_c, ratio=rt)

#                 print T_s, T_c, data_sub['Rate(MHz)'][i] / T_c

        elif 'CBOC(' in data_sub['Modulation'][i]:
            s = data_sub['Modulation'][i]
            # Getting the values between parenthesis and converting to float
            T_s, T_c, rt = [floaty(x) for x in s[s.find("(")+1:s.find(")")].split(',')]
            model = psd.CBOC(f=frequency_gnss - data_sub['Frequency[MHz]'][i], n_c=T_c,
                             n_s=T_s, f0=data_sub['Rate(MHz)'][i] / T_c, ratio=rt)

#                 print T_s, T_c, data_sub['Rate(MHz)'][i] / T_c

        # Checking for NaN values in model and swapping with zeros.
        model = np.array([0 if np.isnan(x) else x for x in model])
        
#         Adding top-hat functions
        model2 = af.tophat_rect(f=frequency_gnss, fi=data_sub['Frequency[MHz]'][i], 
                       band=band_lvl[0], level=band_lvl[1], values=model)


        sed.append(power * model2 * 1e26 / 1e6)  # In Jansky

    sed = np.array(sed)


    # Returning the SED values and the data of the input for the user
    return sed

 
    
    
    
## Function that returns the TOD values

def TOD_sats(name_tod, fname, frequency_tod, beam_model, band_lvl, excel_sat, excel_cat_loc):
    '''
    Returns the sat_temp in units of mK
    name - a str;
    frequency - range ； frequencies in MHz;
    beam_model - coming from Yi-Chaos angular seperation code, if 2d then should be transposed
    excel_sat - the satellite catalogue name
    excel_cat_loc - the location of the mask, if !None, you set the location, else location is the same
    '''
    
    sats_model = gnss_satellites(name_gnss=name_tod, frequency_gnss=frequency_tod, band_lvl=band_lvl, excel_sat_info=excel_sat, excel_loc=excel_cat_loc)   # Calling another fucntion
    sats_model_t = np.sum(sats_model, axis=0)       # Adding all the satellites togther
    
    # Multiplied with the gains  考虑增益
    sats_model_tc = sats_model_t * cc.c.value**2 / cc.k_B.value / 4 / np.pi / (frequency_tod*1e6)**2 / 1e26  # Multiplying the constants, look at the page to see the units
    
    sats_model_tc = sats_model_tc * 1e4  # The gain being multiplied as well, see Harper
    
    # Cheking if the beam model is 1d or 2d
    if beam_model.ndim == 1:
        temp_sats = sats_model_tc[:, None] * beam_model[None, :]     # Multiplying the beam and the satellite data
    
    elif beam_model.ndim == 2:
        temp_sats = beam_model * sats_model_tc[:, None]     # Multiplying the beam and the satellite data

    temp_sats = temp_sats  ## Units in Kelvin                
                              
    return temp_sats, sats_model_tc, sats_model_t, frequency_tod    # Kelvin