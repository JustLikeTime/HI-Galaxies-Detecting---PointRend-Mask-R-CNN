"""Models of radio frequency interference."""
import astropy.units as u
import numpy as np
import warnings
from pathlib import Path


class RfiStation:
    """Generate RFI based on a particular "station".

    Parameters
    ----------
    f0 : float
        Frequency that the station transmits (any units are fine).
    duty_cycle : float, optional
        With ``timescale``, controls how long the station is seen as "on". In
        particular, ``duty_cycle`` specifies which parts of the station's cycle are
        considered "on". Can be considered roughly a percentage of on time.
    strength : float, optional
        Mean magnitude of the transmission.
    std : float, optional
        Standard deviation of the random RFI magnitude.
    timescale : float, optional
        Controls the length of a transmision "cycle". Low points in the sin-wave cycle
        are considered "off" and high points are considered "on" (just how high is
        controlled by ``duty_cycle``). This is the wavelength (in seconds) of that
        cycle.

    Notes
    -----
    This creates RFI with random magnitude in each time bin based on a normal
    distribution, with custom strength and variability. RFI is assumed to exist in one
    frequency channel, with some spillage into an adjacent channel, proportional to the
    distance to that channel from the station's frequency. It is not assumed to be
    always on, but turns on for some amount of time at regular intervals.
    """

    def __init__(
        self,
        f0: float = 0.1220703125,
        duty_cycle: float = 0.5,
        strength: float = 100.0,
        std: float = 10.0,
        timescale: float = 100.0,
    ):
        self.f0 = f0
        self.duty_cycle = duty_cycle
        self.strength = strength
        self.std = std
        self.timescale = timescale

    def __call__(self, lsts, freqs, num_channels=20):
        """Compute the RFI for this station.

        Parameters
        ----------
        lsts : array-like
            LSTs at which to generate the RFI.
        freqs : array-like of float
            Frequencies in units of ``f0``.


        Returns
        -------
        array-like
            2D array of RFI magnitudes as a function of LST and frequency.
        """
        # initialize an array for storing the rfi
        rfi = np.zeros((lsts.size, freqs.size), dtype=complex)

        # get the mean channel width
        channel_width = np.mean(np.diff(freqs))

        # find out if the station is in the observing band
        try:
            ch1 = np.argwhere(np.abs(freqs - self.f0) < channel_width)[0, 0]
        except IndexError:
            # station is not observed
            return rfi

        # find out whether to use the channel above or below... why?
        # I would think that the only time we care about neighboring
        # channels is when the station bandwidth causes the signal to
        # spill over into neighboring channels
        
        # Find out if the station is in the observing band and get the indices
        # of the surrounding channels.
        channels = []
        for i in range(-num_channels // 2 + 1, num_channels // 2 + 1):
            try:
                channel = np.argwhere(np.abs(freqs - self.f0 + i * channel_width) < channel_width)[0, 0]
                channels.append(channel)
            except IndexError:
                # Station is not observed
                continue

        #ch2 = ch1 + 1 if self.f0 > freqs[ch1] else ch1 - 1

        # generate some random phases
        phs1, phs2 = np.random.uniform(0, 2 * np.pi, size=2)

        # find out when the station is broadcasting
        is_on = 0.999 * np.cos(lsts * u.sday.to("s") / self.timescale + phs1)
        is_on = is_on > (1 - 2 * self.duty_cycle)

        # generate a signal and filter it according to when it's on
        signal = np.random.normal(self.strength, self.std, lsts.size)
        signal = np.where(is_on, signal, 0) * np.exp(1j * phs2)

        # now add the signal to the rfi array
        # for ch in (ch1, ch2):
        #     # note: this assumes that the signal is completely contained
        #     # within the two channels ch1 and ch2; for very fine freq
        #     # resolution, this will usually not be the case
        #     df = np.abs(freqs[ch] - self.f0)
        #     taper = (1 - df / channel_width).clip(0, 1)
        #     rfi[:, ch] += signal * taper

        # Now add the signal to the RFI array
        for ch in channels:
            # Note: this assumes that the signal is completely contained
            # within the surrounding channels; for very fine freq
            # resolution, this will usually not be the case
            df = np.abs(freqs[ch] - self.f0)
            # sigma = channel_width * num_channels // 2
            # taper = np.exp(-0.5 * (df / sigma) ** 2)
            taper = (1 - df / (channel_width * num_channels / 2)).clip(0, 1)
            rfi[:, ch] += signal * taper

        return rfi


class Stations:
    """A collection of RFI stations.

    Generates RFI from all given stations.

    Parameters
    ----------
    stations : list of :class:`RfiStation`
        The list of stations that produce RFI.
    """

    _alias = ("rfi_stations",)
    return_type = "per_baseline"

    def __init__(self, stations=None):
        self.stations = stations

    def __call__(self,  lsts, freqs, num_channels=20):
        """Generate the RFI from all stations.

        Parameters
        ----------
        lsts : array-like
            LSTs at which to generate the RFI.
        freqs : array-like of float
            Frequencies in units of ``f0`` for each station.

        Returns
        -------
        array-like of float
            2D array of RFI magnitudes as a function of LST and frequency.

        Raises
        ------
        TypeError
            If input stations are not of the correct type.
        """
        #stations = (self.stations,)

        # initialize an array to store the rfi in
        rfi = np.zeros((lsts.size, freqs.size), dtype=complex)

        if self.stations is None:
            warnings.warn(
                "You did not specify any stations to simulate.",
                stacklevel=2,
            )
            return rfi
        # elif isinstance(stations, (str, Path)):
        #     # assume that it's a path to a npy file
        #     stations = np.load(self.stations)
        else:

            for station in self.stations:
                if not isinstance(station, RfiStation):
                    if len(station) != 5:
                        raise ValueError(
                            "Stations are specified by 5-tuples. Please "
                            "check the format of your stations."
                        )

                    # make an RfiStation if it isn't one
                    station = RfiStation(*station)

                # add the effect
                rfi += station(lsts, freqs, num_channels)

        return rfi


class Impulse:
    """Generate RFI impulses (short time, broad frequency).

    Parameters
    ----------
    impulse_chance : float, optional
        The probability in any given LST that an impulse RFI will occur.
    impulse_strength : float, optional
        Strength of the impulse. This will not be randomized, though a phase
        offset as a function of frequency will be applied, and will be random
        for each impulse.
    """

    _alias = ("rfi_impulse",)
    return_type = "per_baseline"

    def __init__(self, impulse_chance=0.001, impulse_strength=2.0):
        
        self.chance=impulse_chance
        self.strength=impulse_strength

    def __call__(self, lsts, freqs):
        """Generate the RFI.

        Parameters
        ----------
        lsts : array-like
            LSTs at which to generate the RFI.
        freqs : array-like of float
            Frequencies in arbitrary units.

        Returns
        -------
        array-like of float
            2D array of RFI magnitudes as a function of LST and frequency.
        """

        # initialize the rfi array
        rfi = np.zeros((lsts.size, freqs.size), dtype=complex)

        # find times when an impulse occurs
        impulses = np.where(np.random.uniform(size=lsts.size) <= self.chance)[0]

        # only do something if there are impulses
        if impulses.size > 0:
            # randomly generate some delays for each impulse
            dlys = np.random.uniform(-300, 300, impulses.size)  # ns

            # generate the signals
            signals = self.strength * np.asarray(
                [np.exp(2j * np.pi * dly * freqs) for dly in dlys]
            )

            rfi[impulses] += signals

        return rfi


class Scatter:
    """Generate random RFI scattered around the waterfall.

    Parameters
    ----------
    scatter_chance : float, optional
        Probability that any LST/freq bin will be occupied by RFI.
    scatter_strength : float, optional
        Mean strength of RFI in any bin (each bin will receive its own
        random strength).
    scatter_std : float, optional
        Standard deviation of the RFI strength.
    """

    _alias = ("rfi_scatter",)
    return_type = "per_baseline"

    def __init__(self, scatter_chance=0.0001, scatter_strength=1.0, scatter_std=3.0):

        self.chance = scatter_chance
        self.strength = scatter_strength
        self.std = scatter_std

    def __call__(self, lsts, freqs):
        """Generate the RFI.

        Parameters
        ----------
        lsts : array-like
            LSTs at which to generate the RFI.
        freqs : array-like of float
            Frequencies in arbitrary units.

        Returns
        -------
        array-like of float
            2D array of RFI magnitudes as a function of LST and frequency.
        """

        # make an empty rfi array
        rfi = np.zeros((lsts.size, freqs.size), dtype=complex)

        # find out where to put the rfi
        rfis = np.where(np.random.uniform(size=rfi.size) <= self.chance)[0]

        # simulate the rfi; one random amplitude, all random phases
        signal = np.random.normal(self.strength, self.std) * np.exp(
            2j * np.pi * np.random.uniform(size=rfis.size)
        )

        # add the signal to the rfi
        rfi.flat[rfis] += signal

        return rfi

