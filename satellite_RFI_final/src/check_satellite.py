import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import time
from functools import partial

from astropy import units as u
from astropy.coordinates import AltAz
from astropy.time import Time, TimezoneInfo
from skyfield.api import Topos, load
from datetime import datetime, timedelta

import copy
from tqdm.notebook import tqdm
import multiprocessing as mp
from pathos.multiprocessing import ProcessingPool as Pool

import beam_model as bm



_c_list = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
          "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f"]

_gps_tles = [ 'gps-ops', 'glo-ops', 'beidou', 'galileo', 'sbas', ] #'active', 
_comms_tles = [] #['geo',    'iridium', 'iridium-NEXT',]

def get_sat_tles(sattype='geo', reload=False, source_url=None):
    '''
    Collect the satellite TLEs that fall in the L-Band.
    Parameters
    ----------
    geostationary: boolean
        Include geostationary staellites or not.
    Returns
    -------
    sats: dictionary
        Satellite TLEs ready for consumption by Skyfield.
    '''

    if source_url is None:
        source_url = 'https://www.celestrak.com/NORAD/elements/'
    print('load sat catalogue from %s %s.txt'%(source_url, sattype))
    #gps_tles = ['gps-ops.txt', 'glo-ops.txt', 'beidou.txt', 'galileo.txt', 'sbas.txt', ]
    #comms_tles = ['iridium.txt', 'iridium-NEXT.txt', 'geo.txt']
    #if geostationary:
    #    LbandSats = gps_tles + comms_tles
    #else:
    #    LbandSats = gps_tles + comms_tles[:-1]
    LbandSats = [sattype + '.txt', ]
    sats = {}
    for i in range(len(LbandSats)):
        _sats = load.tle(source_url+LbandSats[i], reload=reload)
        for xx in _sats.keys():
            if not isinstance(xx, int):
                sats[xx] = _sats[xx]

    return sats

def __get_sat_coord__(key, obs_time, obs_location, sats):

    coords = []
    #for key in keys:
    satellite = sats[key]
    topcentric = (satellite - obs_location).at(obs_time)
    alt, az, distance = topcentric.altaz()
    coords.append([az.radians, alt.radians])
    return [coords, key]

def get_sat_coods(sats, obs_time_list, obs_location):
    #for obs_time in obs_time_list:
    #    obs_time = obs_time.datetime.timetuple()
    #    obs_time = load.timescale().utc(*obs_time[:6])
    obs_time = obs_time_list[0]
    obs_time_range = [x.unix - obs_time.unix for x in obs_time_list]

    obs_time = obs_time.datetime.timetuple()
    obs_time = load.timescale(builtin=True).utc(*obs_time[:5] + (obs_time_range,))
    coords = []
    name = []

    # python3 has some issue with pool
    # 
    #pool =  Pool(16)
    #r = pool.map(partial(__get_sat_coord__, sats=sats, 
    #        obs_time=obs_time, obs_location=obs_location), sats.keys())
    #pool.close()
    #pool.join()

    r = [__get_sat_coord__(key, obs_time, obs_location, sats) for key in sats.keys()]

    for _r in r:
        coords.append(_r[0][0])
        name.append(_r[1])


    name = np.array(name)
    coords = np.array(coords)
    coords = np.rollaxis(coords, -1, 0)
    return coords, name

def unix_convert(tlist, sel):
    """
    Convert the UTC time to unix time 
    tlist - 
    """
    ot = tlist.utc[:, sel]     # Selecting the YYYY-MM-DD and hh-mm-ss values from the list per point
    ot2 = ot.astype(int)
    dt = datetime(ot2[0], ot2[1], ot2[2], ot2[3], ot2[4], ot2[5])
    
    return ((dt - datetime(1970, 1, 1)).total_seconds())

def unix_time(time_list, choice):
    """
    Obtain the difference in unix time from a list of utc times
    time_list - the time list should be in UTC format of shape [6, n]
    choice - n value
    
    """
    t0 = unix_convert(tlist=time_list, sel=0) 
    
    time_line = []
    for i in choice:
        time_line.append(unix_convert(tlist=time_list, sel=i) - t0)
        
    return np.array(time_line)

def remove_sats_below_horizen(sats, st_time, ed_time, location):

    #t1 = time.time()

    time_range = ed_time.unix - st_time.unix
    #time_range = st_time + np.linspace(0, time_range, 10) * u.second
    time_range = np.linspace(0, time_range, 50)
    obs_time = st_time.datetime.timetuple()
    obs_time = load.timescale(builtin=True).utc(*obs_time[:5] + (time_range, ))

    # making a place to store the indivdual masks of the satellites alt information
    tlist = []
    
    for ii, key in enumerate(list(sats.keys())):
        satellite = sats[key]
        topcentric = (satellite - location).at(obs_time)
        alt, az, distance = topcentric.altaz()
        
        if np.all(alt.radians < 0):
            del sats[key]
        else:
            alt_mask = (ma.masked_array(alt.degrees, 
                                        mask= alt.degrees<0))
            tlist.append(unix_time(time_list=obs_time,
                              choice=np.where(alt_mask.mask==True)[0]))


    #print("remove sat use %f s"%(time.time() - t1))
    return sats, tlist

class Satellite_Catalogue(object):


    def __init__(self, source_url=None, sats_type=None, reload=False):

        self.sats_type = sats_type
        if self.sats_type is None:
            self.sats_type =  _comms_tles + _gps_tles
        sats = []
        for sats_type in self.sats_type:
            sats.append(get_sat_tles(sats_type, reload=reload, source_url=source_url))
        self.sats = sats

        self._obs_time = None
        self._obs_time_list = np.zeros(1) * u.second
        self._obs_location = None

    @property
    def obs_time(self):
        return self._obs_time

    @obs_time.setter
    def obs_time(self, _obs_time):
        '''
        _obs_time UTC str
        '''
        if isinstance(_obs_time, str):
            self._obs_time = [Time(_obs_time, scale='utc'), ]
        elif isinstance(_obs_time, list):
            self._obs_time = [Time(_o, scale='utc') for _o in _obs_time]

    @property
    def obs_time_list(self):
        return self._obs_time_list

    @obs_time_list.setter
    def obs_time_list(self, _time_list):
        '''
        _time_list in astrou.units, ndarray or list of ndarray.
        if it is ndarray, same obs_time_list applied to all obs
        '''
        if self.obs_time is None:
            raise ValueError('Set obs_time firstly')

        if isinstance(_time_list, np.ndarray):
            self._obs_time_list = [_time_list, ] * len(self.obs_time)
        elif isinstance(_time_list, list):
            if len(_time_list) != len(self.obs_time):
                raise ValueError('time_list num does not match obs num')
            self._obs_time_list = _time_list

    @property
    def obs_location(self):
        return self._obs_location

    @obs_location.setter
    def obs_location(self, _location):
        '''
        _location = [Lat, Lon] in u.deg
        '''
        Lat, Lon = _location
        if np.sign(Lat.value) == -1:
            _Lat = "%f S"%abs(Lat.value)
        else:
            _Lat = "%f N"%abs(Lat.value)

        if np.sign(Lon.value) == -1:
            _Lon = "%f W"%abs(Lon.value)
        else:
            _Lon = "%f E"%abs(Lon.value)
        self._obs_location = Topos(_Lat, _Lon)

    def get_sate_coords(self):

        if self.obs_time is None:
            msg = 'Need to set obs_time'
            raise ValueError(msg)

        if self.obs_location is None:
            msg = 'Need to set obs_location'
            raise ValueError(msg)

        #obs_time_list = self.obs_time + self.obs_time_list
        coord_list_total = []
        name_list_total = []
        for oo, obs_start_time in enumerate(self.obs_time):
            obs_time_list = obs_start_time + self.obs_time_list[oo]
            coord_list = []
            name_list  = []

            print("Time range %s - %s"%(obs_time_list[0].utc, obs_time_list[-1].utc))
            for ii, sat_type in enumerate(self.sats_type):
                t0 = time.time()
                sats = copy.copy(self.sats[ii])
                #if sat_type == 'geo':
                #    sats = remove_sats_below_horizen(sats, obs_time_list[0],
                #            obs_time_list[-1], self.obs_location)
                coord_list_sats, name_list_sats\
                        = get_sat_coods(sats, obs_time_list, self.obs_location)
                coord_list.append(coord_list_sats)
                name_list.append(name_list_sats)
                print("Satellite %12s has %4d satellites"%(sat_type,
                                                         len(sats.keys())),
                      " [use %6.2f s]"%(time.time() - t0))
            coord_list_total.append(coord_list)
            name_list_total.append(name_list)

        self.coord_list = coord_list_total
        self.name_list  = name_list_total

    def get_sate_coords_brandon(self):

        if self.obs_time is None:
            msg = 'Need to set obs_time'
            raise ValueError(msg)

        if self.obs_location is None:
            msg = 'Need to set obs_location'
            raise ValueError(msg)
    
        #obs_time_list = self.obs_time + self.obs_time_list
        coord_list_total = []
        name_list_total = []
        for oo, obs_start_time in enumerate(self.obs_time):
            print(obs_start_time)
            obs_time_list = obs_start_time + self.obs_time_list[oo]
            coord_list = []
            name_list  = []
            delete_list = []
            print( "Time range %s - %s"%(obs_time_list[0].utc, obs_time_list[-1].utc))
            for ii, sat_type in enumerate(self.sats_type):
                t0 = time.time()
                sats = copy.copy(self.sats[ii])
                #if sat_type == 'geo':
                # Edit - removes satellites thats are below the horizon
                sats, tlist_ = remove_sats_below_horizen(sats, obs_time_list[0], 
                                                 obs_time_list[-1], self.obs_location)
                print("Satellite %12s has %4d satellites"%(sat_type, 
                                                         len(sats.keys()))),
                if len(tlist_)==0:
                    delete_list.append(ii)
                    print('[Removing this constellation from sats_type]')
                
                else:
                    coord_list_sats, name_list_sats\
                            = get_sat_coods(sats, obs_time_list, self.obs_location)

                    coord_list_sats_ma = []   # To store the masked values
                    for jj in range(len(tlist_)):
                        """
                        Searching for the times that correspond to tlist_. 
                        If the time is 0 sec then take the last value in 
                        tlist_ and mask everything less than that. 
                        Opposite holds !0 then the first t_list_
                        and everything greater will be masked
                        """
                        try:
                            if int(tlist_[jj][0])==0:
                                mask_tlist = self.obs_time_list[0].value<=tlist_[jj][-1]  
                            else:
                                mask_tlist = self.obs_time_list[0].value>=tlist_[jj][0]
                        except IndexError:
                            mask_tlist = np.zeros((len(self.obs_time_list[0].value)), dtype=bool)


                        coord_list_sats_ma.append(ma.masked_array(coord_list_sats[:, jj, :], 
                                                 mask = np.column_stack((mask_tlist, mask_tlist))))

                    coord_list.append(coord_list_sats_ma)
                    name_list.append(name_list_sats)
                    print(" [use %6.2f s]"%(time.time() - t0))

                coord_list_total.append(coord_list)
                name_list_total.append(name_list)
        
        
        """
        Deleting any satellite constellation that do not show from the list
        """
        self.sats_type = [element for (i,element) in enumerate(self.sats_type) if i not in delete_list]
        
        
        """
        This section is for the satellite coords to change their shape for the check angular
        section of the code. Will comment tomorrow
        """
        coord_sats_total = []
        for jj in range(len(coord_list_total[0])):

            cons_sats = ma.masked_array(coord_list_total[0][jj])

            constellation_list = []
            try:
                for i in range((cons_sats.shape[1])):
                    constellation_list.append(cons_sats[:, i, :])
            
            except IndexError:
                continue 

            coord_sats_total.append(ma.masked_array(constellation_list))   

        self.coord_list = coord_sats_total
        self.name_list  = name_list_total

        
        
    def itersats_temperature(self, pointings, beam_func=None):

        '''
        pointings : ndarray N x 2 (Az, Alt) in deg

        yield: separation angle if beam_func is None; or return the 
                beam convolved temperature, assuming the sats temperature
                is 1 K.
        '''
        if isinstance(pointings, np.ndarray):
            'uisng same pointing for all obs'
            pointings = [pointings, ] * len(self.coord_list)
        elif isinstance(pointings, list):
            if len(pointings) != len(self.coord_list):
                raise ValueError('pointing num does not match obs num')

        for oo, obs_start_time in enumerate(self.obs_time):

            coord_list = self.coord_list[oo]

            obs_time_list = obs_start_time.unix\
                    + self.obs_time_list[oo].to(u.second).value
            obs_time_list = Time(obs_time_list, format='unix')
            obs_time_list.format = 'fits'

            pointing_Az  = pointings[oo][:, 0][:, None] * np.pi / 180.
            pointing_Alt = pointings[oo][:, 1][:, None] * np.pi / 180.

            angle_separation_list = []
            for ii in tqdm(range(len(self.sats_type))):

                coords = coord_list[ii]
#                 coords = np.array(coords)
                coords_Az  = coords[:, :, 0]
                coords_Alt = coords[:, :, 1]

                sats_name  = self.name_list[oo][ii]

                _angle = np.sin(pointing_Alt) * np.sin(coords_Alt)\
                       + np.cos(pointing_Alt) * np.cos(coords_Alt)\
                       * np.cos(pointing_Az - coords_Az)
                _angle = np.arccos(_angle) * 180. / np.pi
                if beam_func is not None:
                    _angle = beam_func(_angle)

                yield obs_time_list, self.sats_type[ii], sats_name, _angle

                
                
    def get_angular_separation(self, pointings, beam_func=None):
        '''
        pointings : ndarray N x 2 (Az, Alt) in deg

        return: separation angle if beam_func is None; or return the 
                beam convolved temperature, assuming the sats temperature
                is 1 K.
        '''

        if isinstance(pointings, np.ndarray):
            'uisng same pointing for all obs'
            pointings = [pointings, ] * len(self.coord_list)
        elif isinstance(pointings, list):
            if len(pointings) != len(self.coord_list):
                raise ValueError('pointing num does not match obs num')

        self.angle_separation_list = []
        for oo, coord_list in enumerate(self.coord_list):

            pointing_Az  = pointings[oo][:, 0][:, None] * np.pi / 180.
            pointing_Alt = pointings[oo][:, 1][:, None] * np.pi / 180.

            angle_separation_list = []
            for ii in range(len(self.sats_type)):

                coords = coord_list[ii]
#                 coords = ma.masked_array(coords)  # Already a masked array (can check)
                coords_Az  = coords[:, :, 0]
                coords_Alt = coords[:, :, 1]

                _angle = np.sin(pointing_Alt) * np.sin(coords_Alt)\
                       + np.cos(pointing_Alt) * np.cos(coords_Alt)\
                       * np.cos(pointing_Az - coords_Az)
                _angle = np.arccos(_angle) * 180. / np.pi
                if beam_func is not None:
                    _angle = beam_func(_angle)
                angle_separation_list.append(_angle)

            self.angle_separation_list.append(angle_separation_list)

    def check_angular_separation(self, pointings, max_angle=10, beam_func=None,
            ymin=None, ymax=None, axes=None, plot=True):
        
        
        #self.get_angular_separation(pointings, beam_func=beam_func)
        self.get_angular_separation(pointings, beam_func=None)
        for oo, obs_start_time in enumerate(self.obs_time):
            if plot:
                 if axes is None:
                     fig = plt.figure(figsize=(10, 3))
                     ax  = fig.add_axes([0.05, 0.1, 0.95, 0.85])
                 else:
                     figi = axes[0]
                     ax = axes[1][oo]
                     #fig = ax.get_figure()

            obs_time_list = obs_start_time.unix\
                    + self.obs_time_list[oo].to(u.second).value
            if plot:
                 x_axis = [ datetime.utcfromtimestamp(s) for s in obs_time_list]
                 x_axis = mdates.date2num(x_axis)
            _angle_sum_total = np.zeros(len(obs_time_list))
            
            # Edit - list to save satellite angles with respect to the pointing 
            satellite_angle = []
            for ii in range(len(self.sats_type)):

                _angle = self.angle_separation_list[oo][ii]
                _angle = np.ma.masked_greater(_angle, max_angle)
                #Edit - saving the angle which the satellites make with the pointing
                satellite_angle.append(_angle)
                if beam_func is not None:
                    _angle = beam_func(_angle)
                print(_angle)
                if plot:
                     ax.plot(x_axis, _angle, '.-', color=_c_list[ii], ms=1, lw=0.1)
                     ax.plot(x_axis, _angle, '-', color=_c_list[ii], lw=0.1)


                # Change all the np.sum to ma.sum where needed
                _angle_sum = ma.sum(_angle, axis=1)
                _angle_sum_total += _angle_sum
                if plot:
                     ax.plot(x_axis, _angle_sum, '.-', color=_c_list[ii], ms=1, lw=1,
                             zorder=1000)
            if plot:
                ax.plot(x_axis, _angle_sum_total, '-', color='k', lw=1,
                        zorder=2000)

                date_format = mdates.DateFormatter('%H:%M')
                ax.xaxis.set_major_formatter(date_format)
                ax.set_xlim(xmin=x_axis[0], xmax=x_axis[-1])
                ax.set_xlabel(obs_start_time.fits)
                ax.set_ylim(ymin=ymin, ymax=ymax)
                if beam_func is not None:
                    ax.semilogy()
                ax.minorticks_on()
                ax.tick_params(length=4, width=1, direction='in')
                ax.tick_params(which='minor', length=2, width=1, direction='in')
                if axes is None:
                    plt.show()


        return satellite_angle

    def check_altaz(self, figaxes=None):
        #coord_list = self.coord_list

        _n  = len(self.coord_list)
        if figaxes is None:
            fig = plt.figure(figsize=(6 * _n + 2 , 6))
            gs = gridspec.GridSpec(1, _n, right=0.95, wspace=0.1, figure=fig)
        else:
            fig, axes = figaxes

        for oo, coord_list in enumerate(self.coord_list):
            #name_list = self.name_list[oo]
            #fig = plt.figure(figsize=(6, 6))
            #ax  = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection='polar',
            #                    theta_offset=0.5 * np.pi, theta_direction=-1)
            if figaxes is None:
                ax = fig.add_subplot(gs[0,oo], projection='polar',
                        theta_offset=0.5 * np.pi, theta_direction=-1)
            else:
                ax = axes[oo]
            
            legend_list = []
            for ii in range(len(self.sats_type)):

                
                coords = coord_list[ii]
                coords = np.array(coords)

                #names  = name_list[ii]

                for jj in range(coords.shape[1]):
                    good = coords[:, jj,1] > 0
                    ax.plot(coords[good,jj,0], 90.- coords[good,jj,1] * 180./np.pi,
                            '.-', color=_c_list[ii], mfc=_c_list[ii], mec='none',
                            lw=0.1, ms=2)
                #ax.scatter(coords[:,0], 90.- coords[:,1] * 180./np.pi, s=10,
                #      marker='s', facecolor='none', edgecolor=_c_list[ii])

                legend_list.append(mpatches.Patch(color=_c_list[ii], 
                                                  label='%s'%self.sats_type[ii]))

            ax.set_rlim(0, 90)
            ax.set_title(self.obs_time[oo])
            if oo == 0:
                ax.legend(handles=legend_list, frameon=False, markerfirst=False, loc=1,
                    bbox_to_anchor=(1, 0.1, 0.15, 1))
        if figaxes is None:
            plt.show()

    def check_pointing(self, altaz, figaxes=None, pad=2., obs_idx=0):
        '''
        rising_altaz = [Az, Alt]
        setting_altaz = [Az, Alt]
        '''
        #_az0, _alt0 = altaz
        _az_min, _alt_min = np.min(altaz, axis=0)
        _az_max, _alt_max = np.max(altaz, axis=0)
        if _az_min < 0: _az_min += 360.
        if _az_max < 0: _az_max += 360.

        if figaxes is None:
            fig2 = plt.figure(figsize=(12, 12))
            gs = gridspec.GridSpec(1, 1, right=0.95, wspace=0.05, figure=fig2)
            ax_r = fig2.add_subplot(gs[0,0])
        else:
            fig2, ax_r = figaxes

        coord_list_1 = self.coord_list[obs_idx]
        name_list_1  = self.name_list[obs_idx]

        _sats_list1 = []

        legend_list = []
        for ii in range(len(self.sats_type)):
            coords_1 = coord_list_1[ii]
            coords_1 = np.array(coords_1)

            names_1 = name_list_1[ii]

            for jj in range(coords_1.shape[1]):
                good  = coords_1[:, jj, 1] > 0
                az  = coords_1[good, jj, 0] * 180./np.pi
                alt = coords_1[good, jj, 1] * 180./np.pi
                #az[az>180.] -= 360.
                ax_r.plot(az, alt, '.-', color=_c_list[ii], mfc=_c_list[ii],
                    mec='none', lw=0.1, ) #label=names_1[jj])

                good  = coords_1[:, jj, 0]* 180./np.pi > _az_min - pad
                good *= coords_1[:, jj, 0]* 180./np.pi < _az_max + pad
                good *= coords_1[:, jj, 1]* 180./np.pi > _alt_min - pad - 0.5
                good *= coords_1[:, jj, 1]* 180./np.pi < _alt_max + pad + 0.5
                if np.any(good):
                    az  = np.mean(coords_1[good, jj, 0] * 180./np.pi)
                    alt = np.mean(coords_1[good, jj, 1] * 180./np.pi)
                    _sats_list1.append([az, alt, r'%s'%names_1[jj], _c_list[ii]])

            legend_list.append(mpatches.Patch(color=_c_list[ii], 
                label='%s'%self.sats_type[ii]))
        #shift = 0.015
        shift = 0.04
        _sats_list1.sort( key=lambda x: x[1])
        for ss in range(len(_sats_list1)):
            _sats = _sats_list1[ss]
            ax_r.annotate(_sats[2].replace('&', r'\&'), (_sats[0], _sats[1]), 
                    textcoords = 'axes fraction',
                    xytext = (0.9, 0.02 + ss * shift),
                    arrowprops=dict(edgecolor=_sats[3],
                        arrowstyle='->', 
                        connectionstyle="arc,angleA=180,armA=100,rad=20"),
                    horizontalalignment='left', 
                    verticalalignment='bottom',
                    size = 5,
                    color=_sats[3])

        ax_r.legend(handles=legend_list, frameon=False, markerfirst=False, loc=2)

        ax_r.set_title(self.obs_time[obs_idx])

        xx0 = np.linspace(_az_min, _az_max, 100)
        yy0 = np.ones_like(xx0) * (_alt_min + _alt_max) * 0.5
        ax_r.fill_between(xx0, yy0+0.5, yy0-0.5, facecolor='none', edgecolor='r')

        xx0 = np.linspace(_az_min-5, _az_max+5, 100)
        yy0 = np.ones_like(xx0) * (_alt_min + _alt_max) * 0.5
        ax_r.fill_between(xx0, yy0+5, yy0-5, facecolor='none', edgecolor='k', ls='--')

        ax_r.set_xlim( _az_min - 15.,  _az_max + 15.)
        ax_r.set_ylabel(r'${\rm Alt}\, [^{\circ}]$')
        ax_r.set_xlabel(r'${\rm AZ}\, [^{\circ}]$')
        ax_r.tick_params(which='both', direction='in')

        _alt = (_alt_min + _alt_max) * 0.5
        ax_r.set_ylim(_alt - 8., _alt + 8.)

        #ax_r.plot([_alt + 5, _alt + 5, _alt -5, _alt - 5, _alt + 5],
        #          [_az_min - 5, _az_max + 5, _az_max + 5, _az_min - 5, _az_min - 5], 'k--')
        if figaxes is None:
            plt.show()
            plt.close()


class MeerKATsite_Satellite_Catalogue(Satellite_Catalogue):

    #def __init__(self, reload=False, source_url=None):
    def __init__(self, *args, **kwargs):

        super(MeerKATsite_Satellite_Catalogue, self).__init__(*args, **kwargs)

        meerKAT_Lon =  (21. + 26./60. + 38.00/3600.) * u.deg #
        meerKAT_Lat = -(30. + 42./60. + 47.41/3600.) * u.deg #
        self.obs_location = [meerKAT_Lat, meerKAT_Lon]

    def get_angular_separation(self, pointings, beam_func=None):

        super(MeerKATsite_Satellite_Catalogue, self).get_angular_separation(
                pointings, beam_func=beam_func)

    def check_angular_separation(self, pointings, max_angle=100, beam_func=None,
            ymin=None, ymax=None, axes=None):
        
        return super(MeerKATsite_Satellite_Catalogue, self).check_angular_separation(
            pointings, max_angle=max_angle,
            beam_func=beam_func, ymin=ymin, ymax=ymax, axes=axes)

class FASTsite_Satellite_Catalogue(Satellite_Catalogue):

    #def __init__(self, reload=False, source_url=None):
    def __init__(self, *args, **kwargs):

        super(FASTsite_Satellite_Catalogue, self).__init__(*args, **kwargs)

        FAST_Lon = (106. + 51./60. + 24.0/3600.) * u.deg
        FAST_Lat = (25. + 39./60. + 10.6/3600.) * u.deg
        self.obs_location = [FAST_Lat, FAST_Lon]

    def get_angular_separation(self, pointings, beam_func=None):

        super(FASTsite_Satellite_Catalogue, self).get_angular_separation(
                pointings, beam_func=beam_func)

    def check_angular_separation(self, pointings, max_angle=100, beam_func=None,
            ymin=None, ymax=None, axes=None):
        
        return super(FASTsite_Satellite_Catalogue, self).check_angular_separation(
            pointings, max_angle=max_angle,
            beam_func=beam_func, ymin=ymin, ymax=ymax, axes=axes)
    





    #def check_pointing(self, rising_altaz, setting_altaz, az_range=18., pad=2.):
    #    '''
    #    rising_altaz = [Az, Alt]
    #    setting_altaz = [Az, Alt]
    #    '''
    #    _az0, _alt0 = rising_altaz
    #    if _az0 < 0: _az0 += 360.

    #    _az1, _alt1 = setting_altaz
    #    if _az1 < 0: _az1 += 360.
    #        

    #    fig2 = plt.figure(figsize=(12, 12))
    #    gs = gridspec.GridSpec(1, 2, right=0.95, wspace=0.05, figure=fig2)
    #    ax_r = fig2.add_subplot(gs[0,0])
    #    ax_s = fig2.add_subplot(gs[0,1])

    #    coord_list_1 = self.coord_list[0]
    #    coord_list_2 = self.coord_list[1]
    #    name_list_1  = self.name_list[0]
    #    name_list_2  = self.name_list[1]

    #    _sats_list1 = []
    #    _sats_list2 = []

    #    legend_list = []
    #    for ii in range(len(self.sats_type)):
    #        coords_1 = coord_list_1[ii]
    #        coords_1 = np.array(coords_1)

    #        coords_2 = coord_list_2[ii]
    #        coords_2 = np.array(coords_2)

    #        names_1 = name_list_1[ii]
    #        names_2 = name_list_2[ii]

    #        for jj in range(coords_1.shape[1]):
    #            good  = coords_1[:, jj, 1] > 0
    #            az  = coords_1[good, jj, 0] * 180./np.pi
    #            alt = coords_1[good, jj, 1] * 180./np.pi
    #            #az[az>180.] -= 360.
    #            ax_r.plot(az, alt, '.-', color=_c_list[ii], mfc=_c_list[ii],
    #                mec='none', lw=0.1, ) #label=names_1[jj])

    #            good  = coords_1[:, jj, 0]* 180./np.pi > _az0 - pad
    #            good *= coords_1[:, jj, 0]* 180./np.pi < _az0 + az_range + pad
    #            good *= coords_1[:, jj, 1]* 180./np.pi > _alt0 - pad - 0.5
    #            good *= coords_1[:, jj, 1]* 180./np.pi < _alt0 + pad + 0.5
    #            if np.any(good):
    #                az  = np.mean(coords_1[good, jj, 0] * 180./np.pi)
    #                alt = np.mean(coords_1[good, jj, 1] * 180./np.pi)# + shift1
    #                _sats_list1.append([az, alt, r'%s'%names_1[jj], _c_list[ii]])

    #        for jj in range(coords_2.shape[1]):
    #            good = coords_2[:, jj,1] > 0
    #            az  = coords_2[good, jj, 0] * 180./np.pi
    #            alt = coords_2[good, jj, 1] * 180./np.pi
    #            ax_s.plot(az, alt, '.-', color=_c_list[ii], mfc=_c_list[ii],
    #                    mec='none', lw=0.1)

    #            good  = coords_2[:, jj, 0]* 180./np.pi > _az1 - pad
    #            good *= coords_2[:, jj, 0]* 180./np.pi < _az1 + az_range + pad
    #            good *= coords_2[:, jj, 1]* 180./np.pi > _alt1 - pad - 0.5
    #            good *= coords_2[:, jj, 1]* 180./np.pi < _alt1 + pad + 0.5
    #            if np.any(good):
    #                az  = np.mean(coords_2[good, jj, 0] * 180./np.pi)
    #                alt = np.mean(coords_2[good, jj, 1] * 180./np.pi) 
    #                _sats_list2.append([az, alt, r'%s'%names_2[jj], _c_list[ii]])

    #        legend_list.append(mpatches.Patch(color=_c_list[ii], 
    #            label='%s'%self.sats_type[ii]))
    #    shift = 0.015
    #    _sats_list1.sort( key=lambda x: x[1])
    #    for ss in range(len(_sats_list1)):
    #        _sats = _sats_list1[ss]
    #        ax_r.annotate(_sats[2].replace('&', r'\&'), (_sats[0], _sats[1]), 
    #                textcoords = 'axes fraction',
    #                xytext = (0.8, 0.02 + ss * shift),
    #                arrowprops=dict(edgecolor=_sats[3],
    #                    arrowstyle='->', 
    #                    connectionstyle="arc,angleA=180,armA=100,rad=20"),
    #                horizontalalignment='left', 
    #                verticalalignment='bottom',
    #                size = 8,
    #                color=_sats[3])

    #    _sats_list2.sort( key=lambda x: x[1])
    #    for ss in range(len(_sats_list2)):
    #        _sats = _sats_list2[ss]
    #        ax_s.annotate(_sats[2].replace('&', r'\&'), (_sats[0], _sats[1]), 
    #                textcoords = 'axes fraction',
    #                xytext = (0.8, 0.02 + ss * shift),
    #                arrowprops=dict(edgecolor=_sats[3],
    #                    arrowstyle='->', 
    #                    connectionstyle="arc,angleA=180,armA=100,rad=20"),
    #                horizontalalignment='left', 
    #                verticalalignment='bottom',
    #                size = 8,
    #                color=_sats[3])

    #    ax_s.legend(handles=legend_list, frameon=False, markerfirst=False,
    #            loc=1, )

    #    ax_r.set_title(self.obs_time[0])
    #    ax_s.set_title(self.obs_time[1])

    #    ax_r.legend()
    #    
    #    xx0 = np.linspace(_az0, _az0 + az_range, 100)
    #    yy0 = np.ones_like(xx0) * _alt0
    #    ax_r.fill_between(xx0, yy0+0.5, yy0-0.5, facecolor='none', edgecolor='r')

    #    ax_r.set_xlim( _az0 - 10.,  _az0 + az_range + 10.)
    #    #ax_r.set_yticklabels([])
    #    ax_r.set_ylabel(r'${\rm Alt}\, [^{\circ}]$')
    #    ax_r.set_xlabel(r'${\rm AZ}\, [^{\circ}]$')
    #    ax_r.tick_params(which='both', direction='in')
    #    
    #    xx1 = np.linspace(_az1, _az1 + az_range, 100)
    #    yy1 = np.ones_like(xx1) * _alt1
    #    ax_s.fill_between(xx1, yy1+0.5, yy1-0.5, facecolor='none', edgecolor='b')

    #    ax_s.set_xlim( _az1 - 10.,  _az1 + az_range + 10.)
    #    ax_s.set_xlabel(r'${\rm AZ}\, [^{\circ}]$')
    #    #ax_s.set_ylabel(r'${\rm Alt}\, [^{\circ}]$')
    #    ax_s.set_yticklabels([])
    #    ax_s.tick_params(which='both', direction='in')
    #    
    #    _alt = (_alt0 + _alt1) * 0.5
    #    ax_r.set_ylim(_alt - 5., _alt + 5.)
    #    ax_s.set_ylim(_alt - 5., _alt + 5.)
    #    plt.show()
