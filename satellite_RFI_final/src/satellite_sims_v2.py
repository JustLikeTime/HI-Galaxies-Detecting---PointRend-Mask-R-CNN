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



"""
x1 = ss(file_name='1551055211', sats_only=None, plots_loc='../../../Plots/',
        sat_catalogue_name='table3B_satellite_v3.csv', sat_catalogue_loc='Satellite_Catalogue/')

x1.excecute(obs_time_start=4000, obs_time_end=4500, obs_frequency_start=800, obs_frequency_end=1400, 
            file_bias_choice=[30,1,10,200,1500,2,0], add_sub=[1, 1], band_lvl=[25, 0.001])

def generate(y=2023, m=2, d=7, h=0, minute=0, s=0, 
             freq_start=1330, 
             freq_end=None, 
             freq_channel=[0,512], 
             time_during=[0,5120], 
             t_adv=1.00663296, 
             freq_resolution= 0.00762939453125, 
             sats="all", 
             beam='FAST',
             data_loc='e:/satellite_RFI_new/'):

    data_save = data_loc + 'Data_Processing/'
    fname = '%d_time%d-%d_freq%d-%d' % (time_obs_start_unix, time_during[0], time_during[-1], freq_start, freq_end)
"""

class sims:

    def __init__(self, 
                 file_name='1675728000',
                 sats_only=1,
                 data_loc='E:/satellite_RFI_new/',
                 s1_data_loc='Plot_test/',
                 Angel_data_loc='Angular_Position/', #s2_data_loc='Notebooks/s2_GNSS_position/', 
                 plots_loc='Plots/',
                 sat_catalogue_name='table3B_satellite_v3.csv',
                 sat_catalogue_loc='Satellite_Catalogue/',
                 bias_choice_loc='Satellite_Catalogue/'):

        self.sats_only = sats_only
        self.file_name = file_name

        self.data_loc = data_loc
        self.s1_data_loc = self.data_loc + s1_data_loc
        self.Angel_data_loc = self.data_loc + Angel_data_loc
        self.plots_loc = self.data_loc + plots_loc
        self.sat_catalogue = sat_catalogue_name
        self.sat_catalogue_loc = self.data_loc + sat_catalogue_loc
        self.bias_choice_loc = self.data_loc + bias_choice_loc


    def start(self, y=2023, m=2, d=7, h=0, minute=0, s=0,
                 freq_start=1330, 
                 freq_end=None, 
                 freq_channel=[0,512], 
                 time_during=[0,5120], 

                 t_adv=1.00663296, 
                 freq_resolution= 0.00762939453125, 

                 file_bias_choice=None, 
                 add_sub=[None, None],
                 band_lvl=[None, None]):
        

        # Sets the frequency band
        self.freq_res = freq_resolution
        if freq_end is None:
            if freq_channel is None:
                raise ValueError('No freq information!')
            else:
                self.freq_start = freq_start + self.freq_res*freq_channel[0]
                self.freq_end = freq_start + self.freq_res*freq_channel[1]
        self.frequency = np.arange(self.freq_start, self.freq_end, self.freq_res)
        self.frequency_band = self.frequency
        
        print("frequency band start at: ", self.frequency_band[0])
        print("frequency band end at: ", self.frequency_band[-1])
        print("frequency band's shape is: ", self.frequency_band.shape, '\n')


        #Sets the time
        self.time_during = time_during
        self.time_mark_0 = datetime(y, m, d, h, minute, s)
        self.time_inds_range = np.arange(time_during[0]*t_adv, time_during[-1]*t_adv, t_adv)
        self.time_obs_band = Time(self.time_mark_0) + self.time_inds_range*u.second
        self.time_obs_start_unix = self.time_obs_band[0].unix

        self.fname = int((self.time_mark_0 - datetime(1970, 1, 1)).total_seconds())

        self.nd_s0 = self.time_obs_band.unix
        self.nd_s0_pos = None
        self.nd_s = None
        self.nd_s0_coords = None
        self.timestamps = None

        print("Time start at: ", self.nd_s0[0], ', also ', self.time_obs_start_unix)
        print("Time end at: ", self.nd_s0[-1])
        print("Time series shape is  ", self.nd_s0.shape)


        # Satellite positioning
        self.satellite_type, self.satellite_angle = self.get_satellite_angle_seperation() 

        print("satellite type is : ", self.satellite_type)
        print("satellite angel is : ", self.satellite_angle)


        # Bandwith and level of difference for attenuation
        self.band_lvl = band_lvl


        # Satellite TOD
        self.satellite_TOD, self.satellite_SED = self.get_gnss_simulaton()
        
        print('satellite TOD is : ', self.satellite_TOD)


        # BG Noise: subtract the observational data; add to the simulations
        self.add_BG, self.sub_BG = add_sub

        # Slice idx in the frequency and the time
        #self.time_idx, self.frequency_idx = self.get_slice_idx(start_time=obs_time_start,
        #                                                       end_time=obs_time_end,
        #                                                       start_frequency=obs_frequency_start,
        #                                                       end_frequency=obs_frequency_end)


        self.time_idx, self.frequency_idx = self.get_slice_idx(start_time=None,#obs_time_start1,
                                                               end_time=None,#obs_time_end,
                                                               start_frequency=None,#obs_frequency_start,
                                                               end_frequency=None)#obs_frequency_end


        # Satellite simulation slice
        self.simulation_slice, self.simulation_TOD_slice, self.bias_choice, self.satellite_TOD_slice = self.get_simulation_slice(
            file_bias_choice=file_bias_choice)


    # 计算TOD的

    def get_satellite_angle_seperation(self):
        '''
        Obtain the angular seperation results for the various satellites
        This takes 9sec to read in
        '''

        try:
            
            fname1 = '%d_time%d-%d_freq%d-%d' % (self.time_obs_start_unix, 
                                                 self.time_during[0], self.time_during[-1], 
                                                 self.freq_start, self.freq_end)
            
            data = pickle.load(open(self.Angel_data_loc + fname1 + "_satellite_angular_position.p", "rb"))

            print("load angular seperation data at : ", self.Angel_data_loc + fname1 + "_satellite_angular_position.p")

            Satellite_type = data["sat_name"]  # Contains the names of the constellations
            Satellite_angle = data["angular"]  # Contains the angular seperation maps

            return Satellite_type, Satellite_angle

        except Exception as e:
            print(fname1 + '-Satellite angular seperation not found :(')


    def get_gnss_simulaton(self):
        '''
        Get the TOD maps of the satellites and our data.
        For all the different types of satellites
        '''

        satellite_TOD = np.array([gm.TOD_sats(name_tod=satellite_name,
                                              fname=self.file_name,
                                              frequency_tod=self.frequency_band,
                                              beam_model=self.satellite_angle[i], band_lvl=self.band_lvl,
                                              excel_sat=self.sat_catalogue,
                                              excel_cat_loc=self.sat_catalogue_loc)[0] for i, satellite_name in
                                  enumerate(self.satellite_type)])

        satellite_SED = np.array([gm.TOD_sats(name_tod=satellite_name,
                                              fname=self.file_name,
                                              frequency_tod=self.frequency_band,
                                              beam_model=self.satellite_angle[i], band_lvl=self.band_lvl,
                                              excel_sat=self.sat_catalogue,
                                              excel_cat_loc=self.sat_catalogue_loc)[1] for i, satellite_name in
                                  enumerate(self.satellite_type)])

        return satellite_TOD, satellite_SED


    #下面是切片的

    def get_slice_idx(self, start_time=None, end_time=None, start_frequency=None, end_frequency=None):
        '''
        A function that provides the idx that you wish to slice from
        start_time - the beginning of the scan period - 774 seconds
        end_time - the end of the scan period - 6474 seconds
        start_frequency - beginning of the freqeuncy band usually 981 MHz
        end_frequency - end of the frequency band usually 1499.9
        '''

        # Slicing in the time domain:
        if start_time == None:
            st_pos = 0
        else:
            st_pos = start_time

        if end_time == None:
            et_pos = -1
        else:
            et_pos = end_time

        print('Time between: ' + str(self.nd_s0[st_pos]) + ' and ' + str(self.nd_s0[et_pos]) + ' in seconds\n')
        

        # Slicing in the frequency domain:
        if start_frequency == None:
            sf_pos = 0
        else:
            sf_pos = start_frequency

        if end_frequency == None:
            ef_pos = -1
        else:
            ef_pos = end_frequency
    
        print('Frequency between: ' + str(self.frequency_band[sf_pos]) + ' and ' + str(
                self.frequency_band[ef_pos]) + ' in MHz\n')


        return (st_pos, et_pos), (sf_pos, ef_pos)


    def get_simulation_slice(self, file_bias_choice=None):
        '''
        Slicing the simualted satellite data with the index values obtained from the 'get_data_sliced'

        '''
        #       This is needs to spliced with the above slice
        satellite_TOD_slice = self.satellite_TOD[:, self.frequency_idx[0]:self.frequency_idx[1],
                              self.time_idx[0]:self.time_idx[1]]

        if type(file_bias_choice) == str:
            bias_choice = np.loadtxt(fname=self.bias_choice_loc + (file_bias_choice) + '.txt', delimiter=' ')

        elif type(file_bias_choice) == list:
            print('Bias choice is follows:' + ', '.join(self.satellite_type) + ', noise')
            bias_choice = file_bias_choice

        else:
            print('Enter the ' + str(len(self.satellite_type) + 1) + ' bias choices for the following: ')
            print(', '.join(self.satellite_type) + ', noise')
            bias_choices_input = input('Enter elements of a list separated by space ')
            bias_choice = [int(i) for i in bias_choices_input.split()]

        gnss_bias_model = np.nansum([satellite_TOD_slice[i] * bias_choice[i] for i in range(len(satellite_TOD_slice))],
                                    0)  # + bias_choice[-1] Don't require this amplitude

        # Threshold ---------------------------------------
        threshold_k = 400  # K
        #         gnss_bias_model_m = np.ma.masked_where(gnss_bias_model >=threshold_k, gnss_bias_model)     # Old method of masking the values,
        # NOTE have to change the the variable name to have 'xxx_m'
        gnss_bias_model[gnss_bias_model >= threshold_k] = threshold_k  # Adding a new threshold method
        #         satellite_TOD_slice[satellite_TOD_slice >= threshold_k] = threshold_k                        # Setting the threshold before bias choice
        # ----------------------------------------------

        if self.add_BG == None:
            gnss_bias_model_m = gnss_bias_model
        else:
            gnss_bias_model_m = gnss_bias_model# + self.calibration_noise_slice

        gnss_bias_model_frequency = self._average_over_frequency_(gnss_bias_model_m)

        return gnss_bias_model_frequency, gnss_bias_model_m, bias_choice, satellite_TOD_slice


    #下面是画图的

    def plotting(self, individual=None, logger=None, axis_limit=None,
                 tod_limit=None, save_file=None, file_type=None):
        """
        Plotting various plots: =====
        1. The 1D Simulation model vs the Observational data 
        2. The Time-Ordered-Data for the obsevational data
        3. The Time-Ordered-Data for the simualtion data

        Parameters:
        individual - If "None" will plot the combined model vs observation data. If "not None" will show the indivdiual satellite componants
        logger - If "None" plots will be in linear scale. If "not None" plots will be in log scale
        axis_limit - If "None" will be the whole limit. If "not None" plots will limited [x1, x2, y1, y2]
        tod_limit - The vmin and vmax for both TOD maps
        sats_only - If you want to show the satellites alone
        save_file - If "not None" file will be saved for all plots. Plots name will include both time and freqeuncy positions.
        suffix - If "not None" plot name will contain user input suffix


        """

        self.file_type = file_type

        self.slice_plot_frequency = self._get_slice_plot_(ALL=individual, save_file=save_file,
                                                          log_scale=logger, limit=axis_limit)

        #self.get_slice_plot_diff = self._get_slice_plot_diff_(ALL=individual, save_file=save_file,
        #                                                      log_scale=logger, limit=axis_limit)

        self.sat_sim_map = self._get_TOD_sim_maps_(log_values=logger, vlimits=tod_limit, save_file=save_file)


    def _get_slice_plot_(self, ALL=None, save_file=None, log_scale=None, limit=None):
        '''
        Function for plotting the Simulation outputs
        '''

        plt.figure(figsize=(14, 4))
        plt.title(self.file_name + ': Time-[' + str(np.round(self.nd_s0[self.time_idx[0]], 2)) + '-' + str(
            np.round(self.nd_s0[self.time_idx[1]], 2)) + '] seconds')

        plt.plot(self.frequency_band[self.frequency_idx[0]:self.frequency_idx[1]], self.simulation_slice, color='red',
                 label='Model')

        if self.sats_only == None:
            observation = self._average_over_frequency_(self.calibration_data_slice)
            plt.plot(self.frequency_band[self.frequency_idx[0]:self.frequency_idx[1]], observation, '-', color='black',
                     label='Data')

        if self.add_BG == None:
            bg_noise = 0
        else:
            bg_noise = 0
            #bg_noise = self._average_over_frequency_(self.calibration_noise_slice)

        plt.xlabel('Frequency [MHz]')
        plt.ylabel('Temperature [K]')
        if ALL != None:
            for i in range(len(self.satellite_type)):
                plt.plot(self.frequency_band[self.frequency_idx[0]:self.frequency_idx[1]],
                         self._average_over_frequency_(self.satellite_TOD_slice[i]) * self.bias_choice[i] +
                         self.bias_choice[-1] + bg_noise,
                         label=self.satellite_type[i] + '  x' + str(self.bias_choice[i]))
            plt.ylim(bottom=1e-2)

        if log_scale == None:
            plt.yscale('log')
            plt.ylabel(r'log$_{10}$(Temperature [K])')

        if limit != None:
            x1, x2, y1, y2 = limit
            plt.xlim(x1, x2)
            plt.ylim(y1, y2)

        plt.legend()
        plt.tight_layout()
        if save_file != None:
            plt.savefig(self.plots_loc + self.file_name + '_' + str(np.round(self.nd_s0[self.time_idx[0]], 2))
                        + '_' + str(np.round(self.nd_s0[self.time_idx[1]], 2)) + '.' + self.file_type)

            # Saving the data to file
            pickle.dump(observation, open(
                self.s1_data_loc + self.file_name + '_observation_' + str(np.round(self.nd_s0[self.time_idx[0]], 2)) +
                '_' + str(np.round(self.nd_s0[self.time_idx[1]], 2)) + '_tod.p', 'wb'))

        else:
            plt.show()


    def _get_TOD_sim_maps_(self, log_values=None, vlimits=None, save_file=None):
        '''
        Obtiaing the TOD maps for the different values for the SIMULATION DATA
        log_values - 
        '''

        extent = [self.nd_s0[self.time_idx[0]], self.nd_s0[self.time_idx[1]],
                  self.frequency_band[self.frequency_idx[1]], self.frequency_band[self.frequency_idx[0]]]

        plt.figure()
        #plt.title(
            #self.file_name + '-Simulation Data: Time-[' + str(np.round(self.nd_s0[self.time_idx[0]], 2)) + '-' + str(
            #    np.round(self.nd_s0[self.time_idx[1]], 2)) + '] seconds')
        plt.title(' ')
        plt.xlabel('Time [s]')
        plt.ylabel('Frequency [MHz]')

        data_slice = self.simulation_TOD_slice

        if log_values == None:
            if vlimits == None:
                hb = plt.imshow(np.log10(data_slice), extent=extent, aspect='auto')
            else:
                hb = plt.imshow(np.log10(data_slice), extent=extent, aspect='auto', vmin=vlimits[0], vmax=vlimits[1])

            cbar = plt.colorbar(hb)
            cbar.set_label(r'log$_{10}$(T) [K]', rotation=270, labelpad=20, y=0.45)

        else:
            if vlimits == None:
                hb = plt.imshow((data_slice), extent=extent, aspect='auto')
            else:
                hb = plt.imshow((data_slice), extent=extent, aspect='auto', vmin=vlimits[0], vmax=vlimits[1])

            cbar = plt.colorbar(hb)
            cbar.set_label(r'T [K]', rotation=270, labelpad=20, y=0.45)

        plt.tight_layout()
        if save_file != None:
            plt.savefig(
                self.plots_loc + self.file_name + '_sim_data_' + str(np.round(self.nd_s0[self.time_idx[0]], 2)) +
                '_' + str(np.round(self.nd_s0[self.time_idx[1]], 2)) + '.' + self.file_type)

            # Saving the file
            pickle.dump(data_slice, open(
                self.s1_data_loc + self.file_name + '_sim_data_' + str(np.round(self.nd_s0[self.time_idx[0]], 2)) +
                '_' + str(np.round(self.nd_s0[self.time_idx[1]], 2)) + '_tod.p', 'wb'))
        else:
            plt.show()



    #获取信息的

    def get_katdal_info(self, s1_data_loc):
        '''
        Obtain KATDAL information regarding the data set such as the frequency and the noise diodes in scanning/no diode fired
        '''

        try:
            fname = self.file_name
            data = pickle.load(open(self.s1_data_loc + fname + '_katdal_info.p', 'rb'))

            return data

        except Exception as e:
            print(fname + '-Katdal Information not found :(')

    def get_frequency_information(self):
        '''
        Function for the frequency start and end postion
        !!! Should add some extra stuff here regarding the printing of the freqeuncy bands.
        For not fixed
        '''
        f_start_idx = self.freq_start
        f_end_idx = self.freq_end
        # f_band = self.frequency[f_start_idx:f_start_idx+f_end_idx]
        f_band = self.frequency[f_start_idx:f_end_idx]

        return f_band



    def _average_over_time_(self, x):
        '''
        Function to return the averaged time response
        from a 2d shape, time should be in the first axis
        '''
        return np.mean(x, axis=0)

    def _average_over_frequency_(self, x):
        '''
        Function to return the averaged frequency response
        from a 2d shape
        '''
        return np.mean(x, axis=1)
