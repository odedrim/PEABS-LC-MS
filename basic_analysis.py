from __future__ import print_function
import os
import sys
import time
import warnings
import argparse
import collections
import datetime
import struct
import math
import numpy
import scipy
from matplotlib import pyplot
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
try:
    import lmfit
    SVM = lmfit.models.SkewedVoigtModel()
except (ImportError, ModuleNotFoundError) as e:
    lmfit = None
    print("Warning: Will produce less precise results as lmfit is not installed.", file=sys.stderr)
# MassLynx API
if sys.version_info.major >= 3 and sys.version_info.minor >= 8:
    # on these versions, the path to the dll needs to be added explicitly
    os.add_dll_directory(os.path.dirname(__file__))
try:
    import MassLynxRawInfoReader as mlrir
    import MassLynxRawScanReader as mlrsr
except (ImportError, ModuleNotFoundError) as e:
    cwd = os.getcwd()
    os.chdir(os.path.dirname(__file__))
    sys.path.append(os.path.dirname(__file__))
    import MassLynxRawInfoReader as mlrir
    import MassLynxRawScanReader as mlrsr
    os.chdir(cwd)
# mine
import peak_picking


#################
# GENERAL STUFF #
#################

TIME_FMT = '%d-%b-%Y %H:%M:%S'

class ProgressBar(object):
    def __init__(self, total, width=80, name='Progress Bar'):
        self._name = name
        self.total = float(total)
        self.cur_real = 0.
        self.cur = 0
        self.width = width
        self.__fin = False
        self.__started = time.time()
        print(name)
        print('v'*width)

    def progress(self, value):
        if self.__fin or value <= self.cur_real:
            return
        self.cur_real = value
        relative = value / self.total
        num_hashtags = int(relative * self.width)
        if num_hashtags > self.cur:
            to_draw = num_hashtags - self.cur
            self.cur = num_hashtags
            sys.stdout.write('#'*to_draw)
            sys.stdout.flush()
        if relative >= 1.:
            self.finish()

    def progress_one(self):
        self.progress(self.cur_real + 1)

    def progress_some(self, value):
        self.progress(self.cur_real + value)

    def finish(self):
        if not self.__fin:
            print()
            print('^'*self.width)
            print("%s ran for %.1f Seconds" % (self._name, time.time() - self.__started))
            self.__fin = True


##################
# LC-MS ANALYSIS #
##################

def magic_protein_peak_number_estimation(mass):
    """
    This "magic" function estimates with pretty high precision the number
    of isotope peaks arising from a given protein or peptide.
    It is based on:
    (1) the empirical observation that carbon atoms account for roughly 53%
        of a protein mass, and thefore #C = ~0.53 / 12 * mass
    (2) the abundance of 13C atoms being ~1.11%
    (3) the assumption that peaks of less than 0.1% of the total ions are
        not really observable or interesting
    (4) a weird function that I fit to data without much explanation :)

    @param mass: The mass of the peptide/protein in Da
    @type mass: number

    @return: Estimated number of detectable isotope peaks
    @rtype: int
    """
    return int(round(15.2 + 5.55e-4 * (mass - 18400 * numpy.exp(-mass / 33300))))

def predict_peak_widths(mass, z, num_mzs, base_peak_width):
    """
    Based on the mass, we can predict how many isotopic peaks would be
    detectable; based on this number and the charge, as well as the base
    peak width of the instrument, we can also predict the width of the
    observable peak/peak-set.
    This is very useful when one wants to integrate all the ion intensities
    corresponding to a single orignal mass.

    @param mass: The mass of the peptide/protein in Da
    @type mass: number
    @param z: The charge of the (first, i.e. highest z = lowest m/z) peak
    @type z: int
    @param num_mzs: Total number of peaks
    @type num_mzs: int
    @param base_peak_width: The base peak width expected, to decide the
                            margins around each m/z value to look at
    @type base_peak_width: float

    @return: expected width of each of the num_mzs peaks
    @rtype: list of float
    """
    exp_num_peaks = magic_protein_peak_number_estimation(mass)
    if z >= 0.3 / base_peak_width:
        # peak overlap is expected
        return [(exp_num_peaks + 1) / (z - i)
                 + base_peak_width
                for i in range(num_mzs)]
    else:
        # peak overlap is NOT expected, just use base peak width
        return [base_peak_width] * num_mzs

def naive_predict_peaks(mass, z_max, nz, base_peak_width = 0.15):
    """
    A naive prediction of the locations and widths of M/z peaks for a given
    macromolecule (or at least protein) mass in an ESI experiment.
    Simply calculates (M + z*H) / z for a given range of z values,
    and uses other functions to predict the widths.
    On Xevo LC-MS, the default base peak width of 0.15 Da gives pretty
    incredible predictions of observed peaks.

    @param mass: The mass of the macromolecule (M) in daltons
    @type mass: number
    @param z_max: Maximal z value to predict for
    @type z_max: int
    @param nz: Number of z values to predict for
    @type nz: int
    @param base_peak_width: The expected width of a single pure ion with
                            charge state +1 on spectrum (default: 0.15)
    @type base_peak_width: float

    @return: predicted peaks as (left limit, right limit)
    @rtype: numpy.ndarray
    """
    # calculate (m + z*H) / z = m / z + 1.01
    ions = mass / numpy.arange(z_max, z_max - nz, -1) + 1.01
    # use predict_peak_widths to predict peak widths
    halfwidths = [x / 2 for x in predict_peak_widths(mass, z_max, nz, base_peak_width)]
    # return predicted peaks as (left limit, right limit) tuples
    return numpy.stack((ions - halfwidths, ions + halfwidths)).T

def integrate_chromatograms(rt_arr, ic_arr):
    """
    To integrate data from a chromatogram, it is insufficient to simply
    sum ion count (TIC/XIC) values, as that assumes uniformity in the
    time-sampling, which many mass spec instruments do not guarantee.
    This has a significant impact when looking at XICs with very narrow
    peaks, as a skipped recording within the peak would have a large
    impact on the integral if calculated naively.

    This function takes the time element into account and returns a more
    precise approximation of the underlying integral.

    @param rt_arr: An array of retention times, size N
    @type rt_arr: numpy.ndarray
    @param ic_arr: (An) array(s) of ion counts (TIC/XIC), size N or M x N
    @type ic_arr: numpy.ndarray
    """
    # calculate the sampling rate (derivative of retention time)
    sampling = rt_arr[1:] - rt_arr[:-1]
    # use the mean for the last value, as we can't calculate it directly
    sampling = numpy.concatenate((sampling, (sampling.mean(),)))
    # calculate the mean between each two consecutive points,
    #  using the last point's own value twice to get back its original
    #  value because as above, we don't have the next one to use
    means = (ic_arr.T + numpy.concatenate((ic_arr.T[1:], ic_arr.T[-1:]))) / 2
    # now we have all the values to calculate a more precise integral
    return (means * sampling).sum(0)


class MassSpec(object):
    def __init__(self, acq_date, label, msfunc_params, progress_bar=True, debug=False):
        """
        @param acq_date: Date spectrum was acquired
        @type acq_date: datetime.datetime
        @param label: Label for this spectrum
        @type label: str
        @param msfunc_params: the LC-MS functions (e.g. MS, DAD) for which we have data, and any other associated information (e.g. [{'func': 'MS', 'ion_mode': 'ES+', total_scans: 239}, {'func': 'DAD', 'ion_mode': 'EI+', 'total_scans': 24001}]); minimally has to contain the 'func' and 'total_scans' keys for each function
        @type msfunc_params: iter of dict
        @param progress_bar: Whether or not to show a progress bars for
                             time-consuming processes (default: True)
        @type progress_bar: bool
        @param debug: Turn on debug printing (default: False)
        @type debug: float
        """
        self.acq_date = acq_date
        self.label = label
        self._msfunc_params = tuple(msfunc_params)
        num_funcs = len(self._msfunc_params)
        self._pb = progress_bar
        self._debug = debug
        # some things to avoid re-analysis of the same data
        self.__rts = [None] * num_funcs
        self.__tics = [None] * num_funcs
        self.__allsum = [None] * num_funcs
        self.__abs = {}

    def _get_rt(self, msfunc, i):
        """
        Get the retention time (RT) for the i's scan of the MS function msfunc.

        *** Implementation is dependent on the file type and engine used and
            should be overridden by each daughter class. ***

        @param msfunc: MS function to use (default: 0)
        @type msfunc: int
        @param i: Scan number
        @type i: int

        @return: Retention time
        @rtype: number
        """
        raise NotImplementedError("This function should be overridden in classes inheriting from MassSpec!")

    def _get_scan(self, msfunc, i):
        """
        Get scan data - m/z values and their corresponding inensities - for the
        i's scan of the MS function msfunc.

        *** Implementation is dependent on the file type and engine used and
            should be overridden by each daughter class. ***

        @param msfunc: MS function to use (default: 0)
        @type msfunc: int
        @param i: Scan number
        @type i: int

        @return: m/z list and corresponding intensity list
        @rtype: tuple (tuple, tuple)
        """
        raise NotImplementedError("This function should be overridden in classes inheriting from MassSpec!")

    def _get_x_range(self, msfunc):
        """
        Get the range of values on the x axis (usually m/z or wavelength) for
        the MS function msfunc.

        *** Implementation is dependent on the file type and engine used and
            should be overridden by each daughter class. ***

        @param msfunc: MS function to use (default: 0)
        @type msfunc: int

        @return: x value range (minimum, maximum)
        @rtype: tuple (float, float)
        """
        raise NotImplementedError("This function should be overridden in classes inheriting from MassSpec!")

    def _get_chromatogram(self, msfunc=0, mzranges=None, fltr=None):
        """
        Retrieve the data from the file and marginalise over m/z to receive
        a chromatogram; filters can be used to extract ions (XIC) or no
        filters for a TIC.

        @param msfunc: MS function to use (default: 0)
        @type msfunc: int
        @param mzranges: ranges of m/z values ((min1,max1),(min2,max2),...) to accept, all other m/z values will be rejected and not summed over
        @type mzranges: numpy.ndarray
        @param fltr: Filter function to apply to (m/z, intensity) pairs
        @type fltr: function

        @return: total or extracted ion chromatogram as (RTs, T/XICs)
        @rtype: numpy.ndarray, list
        """
        get_tic = mzranges is None and fltr is None
        if not get_tic or self.__rts[msfunc] is None or self.__tics[msfunc] is None:
            # get from file
            rts = []
            tics = []
            times = []
            n = self._msfunc_params[msfunc]['total_scans']
            if self._pb:
                pb = ProgressBar(n, name='Gathering chromatogram data')
            for i in range(n):
                rts.append(self._get_rt(msfunc, i))
                mzs, intens = self._get_scan(msfunc, i)
                intens = numpy.array(intens)
                mzs = numpy.array(mzs)
                where = None
                if get_tic:
                    tics.append(intens.sum())
                else:
                    if mzranges is not None:
                        # this looks scary, but it just calculates a boolean array that represents all the places where the m/z is within one of the ranges
                        # thanks to Thomas Lohr for this line of code
                        where = ((mzs.reshape(-1, 1) >= mzranges[:,0]) &
                                 (mzs.reshape(-1, 1) <= mzranges[:,1])).any(axis=1)
                    if fltr is not None:
                        vfltr = numpy.frompyfunc(fltr, 2, 1)
                        if where is not None:
                            where &= vfltr(mzs, rts)
                        else:
                            where = vfltr(mzs, rts)
                    # sum over the respective intensities
                    tics.append(intens.dot(where))
                if self._pb:
                    pb.progress_one()
            self.__rts[msfunc] = numpy.array(rts)
            tics = numpy.array(tics)
            if get_tic:
                self.__tics[msfunc] = tics
        # either return just-calculated values or load from memory
        if get_tic:
            tics = self.__tics[msfunc]
        return self.__rts[msfunc], tics.copy()

    def plot_chromatogram(self, msfunc=0, normalise=False, mzranges=None, fltr=None, fltr_label=None, plot_params={}, ax=None, show=True):
        rts, tics = self._get_chromatogram(msfunc, mzranges=mzranges, fltr=fltr)
        if normalise:
            tics /= tics.max()
        func_type = self._msfunc_params[msfunc]['func']
        if func_type == 'DAD':
            data_label = 'Absorbance'
        else:
            if fltr is None and mzranges is None:
                data_label = 'TIC'
            else:
                data_label = 'XIC' + (' (%s)' % fltr_label if fltr_label is not None else '')
        if ax is None:
            fig, ax = pyplot.subplots()
        plt = ax.plot(rts, tics, label=data_label, **plot_params)
        ax.set_xlabel('RT (min)')
        ax.set_ylabel(('Normalised ' if normalise else '') + data_label)
        if show:
            pyplot.legend()
            pyplot.show()
            pyplot.close()
        else:
            return ax, plt, (rts, tics)

    def plot_absorbance(self, wavelength=280, normalise=False, show=True):
        # identify correct function
        msfunc = None
        for i, func_params in enumerate(self._msfunc_params):
            if func_params['func'] == 'DAD':
                msfunc = i
        if msfunc is None:
            raise ValueError("No UV/Vis spectral data channel found!")
        # find closest measured wavelength (assumes all scans have same X axis)
        xvals = numpy.array(self._get_scan(msfunc, 0)[0])
        dvector = abs(xvals - wavelength)
        idx = dvector.argmin()
        if dvector[idx] > 2.:
            warnings.warn("Showing absorbance at closest wavelength to %d nm, which is %d nm" % (wavelength, xvals[idx]))
            wavelength = xvals[idx]
        # get absorbance values
        if self.__rts[msfunc] is None:
            # get from file
            rts = []
            total_scans = self._msfunc_params[msfunc]['total_scans']
            for i in range(total_scans):
                rts.append(self._get_rt(msfunc, i))
            self.__rts[msfunc] = numpy.array(rts)
        # or load from memory
        rts = self.__rts[msfunc]
        if idx not in self.__abs:
            abss = []
            total_scans = self._msfunc_params[msfunc]['total_scans']
            for i in range(total_scans):
                abss.append(self._get_scan(msfunc, i)[1][idx])
            self.__abs[idx] = abss
        else:
            # load from memory
            abss = self.__abs[idx]
        if normalise:
            abss = numpy.array(abss)
            abss -= abss.min()
            abss /= abss.max()
        pyplot.plot(rts, abss, label='%d nm' % wavelength)
        pyplot.xlabel('RT (min)')
        pyplot.ylabel(('Normalised ' if normalise else '') + 'A%d' % wavelength)
        if show:
            pyplot.legend()
            pyplot.show()
            pyplot.close()
        return wavelength

    def fetch_raw_data(self, msfunc=0, rtrange=None, mzrange=None, num_bins=0):
        """
        Fetch raw, or minimally processed, 3-dimensional data from the
        raw file for further processing by other procedures.

        @param msfunc: MS function to use (default: 0)
        @type msfunc: int
        @param rtrange: Range of retention times to limit to (optional)
        @type rtrange: tuple (float, float) or None
        @param mzrange: Range of m/z values to accept, all other m/z values will be rejected and not summed over (optional)
        @type mzrange: tuple (float, float) or None
        @param num_bins: Optional argument allowing reduction in the
                         memory required to store the data by binning
                         close-by m/z values together; this specifies
                         the number of bins to collect the data into
        @type num_bins: int

        @return: RTs (size N), m/zs (size M), and TICs or XICs (N x M),
                 each corresponding to one time point and comprising M
                 values for the M m/z values/ranges
        @rtype: tuple (list, list, numpy.ndarray)
        """
        rts = []
        tics = []
        total_scans = self._msfunc_params[msfunc]['total_scans']
        # get mass range and create bins
        if mzrange is None:
            rl, rh = self._get_x_range(msfunc)
        else:
            rl, rh = mzrange
        # calculate bin identities
        if num_bins:
            bin_width = (rh - rl) / float(num_bins)
            mzbins = numpy.arange(rl, rh, bin_width)
        else:
            mzbins = {}
            mz_counter = 0
        if self._pb:
            pb = ProgressBar(total_scans, name='Gathering raw data')
        for i in range(total_scans):
            rt = self._get_rt(msfunc, i)
            if rtrange is None or rtrange[0] <= rt <= rtrange[1]:
                # we are in the correct RT range
                rts.append(rt)
                mz, intens = self._get_scan(msfunc, i)
                bins = numpy.zeros(num_bins or (len(mzbins) + len(mz)))
                for mzi, intensi in zip(mz, intens):
                    # filter by mzrange
                    if rl <= mzi <= rh:
                        if num_bins:
                            # arrange data into bins
                            idx = int((mzi - rl) / bin_width)
                        else:
                            # use m/z values directly rather than binning
                            idx = mzbins.get(mzi, None)
                            if idx is None:
                                # new m/z value
                                mzbins[mzi] = idx = mz_counter
                                mz_counter += 1
                        bins[idx] += intensi
                # remove tail data
                if not num_bins:
                    bins = bins[:mz_counter]
                # done with this set
                tics.append(bins)
            if self._pb:
                pb.progress_one()
        # some annoying consolidation we have to do when no bins are used
        if not num_bins:
            mzbins = numpy.array(tuple(mzbins.keys()))
            # only now we have the final number of m/z values, so we
            #  have to pad the earlier value lists with zeros
            for i in range(len(tics)):
                tics[i] = numpy.pad(tics[i], (0, mz_counter - len(tics[i])))
            # TODO: consider ordering by m/z
        return rts, mzbins, numpy.array(tics)

    def plot_3d(self, msfunc=0, num_bins=1000, rtrange=None, mzrange=None, flatten=True, cmap=None, ax=None, show=True):
        # get RTs and TICs from file
        rts, mzs, tics = self.fetch_raw_data(msfunc, rtrange, mzrange, num_bins)
        # create ranges
        xx = numpy.array(rts)
        zz = numpy.array(tics).T
        if mzrange is not None:
            i0 = mzs.searchsorted(mzrange[0])
            ie = mzs.searchsorted(mzrange[1])
            mzs = mzs[i0:ie]
            zz = zz[i0:ie]
        # make ranges 2d
        xx, yy = numpy.meshgrid(xx, mzs)
        # plot
        if flatten:
            if ax is None:
                fig, ax = pyplot.subplots()
            ##pyplot.pcolor(xx, yy, zz, cmap=cmap, linewidth=0, antialiased=False)
            retval = ax.contour(xx, yy, zz, cmap=cmap, antialiased=False)
        else:
            fig = pyplot.figure()
            ax = fig.add_subplot(111, projection='3d')
            retval = ax.plot_surface(xx, yy, zz, cmap=cmap, rstride=1, cstride=1, linewidth=0, antialiased=False)
        if show:
            pyplot.show()
            pyplot.close()
        else:
            return retval

    def _sum_scans(self, msfunc, scan_b, scan_e, baseline=0.):
        # gather scans to sum up
        if scan_e <= scan_b:
            return
        to_sum = (self._get_scan(msfunc, i) for i in range(scan_b, scan_e+1))
        # calculate sum with minimal set of x values (code: Thomas Lohr)
        ## TODO: make more memory-efficient, currently has to load the
        ##       entire 3d spectrum to memory for this..!
        data = [numpy.array((x,y)).T for x,y in to_sum]
        allx = numpy.unique(numpy.concatenate([d[:,0] for d in data]))
        ally = numpy.zeros_like(allx)
        # XXX THIS DOES NOT TAKE INTO ACCOUNT THE RT AXIS, SO IF XXX
        # XXX THERE IS NON-UNIFORM SAMPLING (AND THERE IS!), IT  XXX
        # XXX WILL UNDERESTIMATE ALL SUMS!                       XXX
        for dat in data:
            _, ix, jx = numpy.intersect1d(allx, dat[:,0], return_indices=True)
            ally[ix] += (dat[jx,1] - baseline)
        return allx, ally

    def sum_scans(self, beg=None, end=None, msfunc=0, baseline=0., export_path=None):
        # load sum from memory if available
        sumall = beg is None and end is None
        if sumall and self.__allsum[msfunc] is not None:
            return self.__allsum[msfunc]
        # else, look for RTs
        if self.__rts[msfunc] is None:
            # get from file
            rts = []
            total_scans = self._msfunc_params[msfunc]['total_scans']
            for i in range(total_scans):
                rts.append(self._get_rt(msfunc, i))
            rts = numpy.array(rts)
            self.__rts[msfunc] = numpy.array(rts)
        # or load from memory
        rts = self.__rts[msfunc]
        # find closest point to beginning RT
        if beg is None:
            begi = 0
        else:
            d_beg_vec = abs(rts - beg)
            begi = d_beg_vec.argmin()
        # find closest point to end RT
        if end is None:
            endi = len(rts) - 1
        else:
            d_end_vec = abs(rts - end)
            endi = d_end_vec.argmin()
        # and we have indexes!
        retval = self._sum_scans(msfunc, begi, endi, baseline)
        if retval is None:
            raise ValueError("No scans found in RT range %.2f-%.2f for function %s" % (beg, end, self._msfunc_params[msfunc]['func']))
        if export_path is not None:
            w = open(export_path, 'w')
            for i in range(len(retval[0])):
                w.write('%f %f\n' % (retval[0][i], retval[1][i]))
            w.close()
        if sumall:
            self.__allsum[msfunc] = retval
        return retval

    def pick_peaks(self, msfunc=0):
        rts, tics = self._get_chromatogram(msfunc)
        last_scan_idx = self._msfunc_params[msfunc]['total_scans'] - 1
        partition_pts = peak_picking.pick_peaks(tics) + [last_scan_idx]
        partitions = [(partition_pts[i-1] if i > 0 else 0, partition_pts[i]) for i in range(len(partition_pts))]
        peaks = [(rts[partition[0]], rts[partition[1]]) +
                 self._sum_scans(msfunc, *partition) for partition in partitions]
        return peaks

    def export_peaks(self, outpath, plot_peaks=False, msfunc=0, show=True):
        peaks = self.pick_peaks(msfunc)
        self.plot_chromatogram(msfunc, normalise=True, show=False)
        fig = pyplot.gcf()
        w = open(outpath + '-peaks.csv', 'w')
        w.write("Start,End,TIC\n")
        for i in range(len(peaks)):
            peak = peaks[i]
            pyplot.plot((peak[0],peak[1]), (-0.08+0.01*(i%2), -0.08+0.01*(i%2)))
            pyplot.text((peak[1]+peak[0])/2, -0.06+0.01*(i%2), str(i+1), ha='center')
            with open(outpath + '-peak%02d-%.02f_%.02f.txt' % (i+1, peak[0], peak[1]), 'w') as f:
                for j in range(len(peak[2])):
                    f.write('%f %f\n' % (peak[2][j], peak[3][j]))
            w.write("%f,%f,%f\n" % (peak[0], peak[1], peak[3].sum()))
        w.close()
        pyplot.savefig(outpath + '.png')
        if plot_peaks:
            fig = pyplot.figure(figsize=(16,12))
            width = int(math.ceil(len(peaks) ** 0.5))
            height = int(math.ceil(float(len(peaks)) / width))
            axarr = fig.subplots(height, width)
            if height == 1:
                axarr = [axarr]
            for i in range(height):
                for j in range(width):
                    n = i * width + j
                    if n < len(peaks):
                        peak = peaks[n]
                        axarr[i][j].plot(peak[2], peak[3])
                        axarr[i][j].set_title('Peak %d (%.1f-%.1f)' % (n+1, peak[0], peak[1]))
            fig.savefig(outpath + '.peaks.png')
        if show:
            fig.canvas.set_window_title(self.label)
            pyplot.show()
        pyplot.close()

    def get_xic(self, mzranges, mzmargins=0.05, msfunc=0, normalise=False, plot=False, label=None, ax=None, show=True):
        # cover many different input types for mzranges
        if not hasattr(mzranges, '__iter__'):
            mzranges = numpy.array((mzranges-mzmargins, mzranges+mzmargins)).reshape(1,2)
        elif len(mzranges) == 2 and not hasattr(mzranges[0], '__iter__'):
            mzranges = numpy.array(mzranges).reshape(1,2)
        else:
            mzranges = numpy.array(mzranges)
        # make a filter label
        if label is None:
            label = '; '.join("%.2f - %.2f" % tuple(rng) for rng in mzranges)
        # fetch the XIC
        if plot:
            ax, line, data = self.plot_chromatogram(msfunc=msfunc, normalise=normalise, mzranges=mzranges, fltr_label=label, ax=ax, show=False)
            if show:
                pyplot.legend()
                pyplot.show()
                pyplot.close()
            return ax, line[0], data
        else:
            rts, tics = self._get_chromatogram(msfunc=msfunc, mzranges=mzranges)
            if normalise:
                tics /= tics.max()
            return None, None, (rts, tics)

    def plot_xic(self, mzranges, mzmargins=0.05, msfunc=0, normalise=False, label=None, ax=None, show=True):
        # ugly hack
        ax, line, data = self.get_xic(mzranges, mzmargins, msfunc, normalise, True, label, ax, show)
        if not show:
            return ax, line, data

    def plot_multi_mxics(self, mzs, mzmargins=0.05, normalise=False, show=True):
        pb = ProgressBar(len(mzs) + 1, name='Plotting XICs')
        ax = self.plot_chromatogram(normalise=normalise, show=False)[0]
        pb.progress_one()
        for mass in mzs:
            self.plot_xic((mass-mzmargins, mass+mzmargins), ax=ax, normalise=normalise, show=False)
            pb.progress_one()
        pyplot.legend()
        if show:
            pyplot.show()

    def predict_ions(self, masses, msfunc=0, refine=True, max_dist=3., sumall=None):
        """
        @param refine: refine m/z choice using the marginalised spectrum,
                       which takes longer but returns much more precise
                       m/z choices (default: True)
        @param refine: bool
        @param max_dist: maximum allowable distance, in Da, from the
                         expected mass implied by the highest peak
                         (default: 3 Da)
        @type max_dist: number
        @param sumall: a spectrum to use for verifying peaks in refinement
                       (optional, will sum entire spectrum if omitted)
        @type sumall: tuple (numpy.ndarray, numpy.ndarray)
        """
        # get mass range for this function
        rl, rh = self._get_x_range(msfunc)
        if isinstance(masses, int) or isinstance(masses, float):
            masses = (masses,)
        mzss = []
        for mass in map(float, masses):
            # coarse-grained approximation
            if mass > rl:
                # calculate charge corresponding to ion at middle of range
                mid = int(mass // ((rl + rh) / 2))
                # take up to 2 higher masses and up to 18 le masses
                rng = range(mid+18, mid-2, -1)
                mzs = [(z, mass/z + 1.01) for z in rng
                       if z > 0 and z*rl <= mass and z*rh >= mass]
                # add full mass (not M+H but M) for masses within the range
                if mass < rh:
                    mzs.append((1, mass))
            else:
                raise ValueError("Mass range %.1f-%.1f cannot detect mass %.1f" % (rl, rh, mass))
            if self._debug:
                print("Unrefined m/z values:", mzs, file=sys.stderr)
            # refine m/z choice using marginalised data
            refined = False
            if refine:
                if sumall is None:
                    sumall = self.sum_scans(msfunc=msfunc)
                # try to fit a peak around each m/z
                fit = numpy.array([peak_picking.mz_peak_likeness(*sumall, mz[1]) for mz in mzs])
                if self._debug:
                    print("Peak fitting matrix:", "  m/z diff   obs/exp rat    peak hght ", *('%12.3f %12.4f %12.2e' % tuple(row) for row in fit), sep='\n', file=sys.stderr)
                # ignore peaks that have a bad fit
                fit[fit[:,1] < 0.4] = 0
                fit[fit[:,0] > 0.3] = 0
                # start with highest peak
                highest = fit[:,2].argmax()
                h_x_diff, h_y_ratio, h_peak_h = fit[highest]
                h_z = mzs[highest][0]
                if self._debug:
                    print("Highest peak:", fit[highest], h_z, file=sys.stderr)
                # continue only if fit is good enough
                if h_z != 0 and h_peak_h > 0 and abs(h_x_diff * h_z) <= max_dist:
                    # scan z values until reaching things that don't look
                    # like peaks in the data or intensity is lower than 1%
                    z = h_z
                    x_diff = h_x_diff
                    peak_h = h_peak_h
                    i = highest
                    while z > 1 and abs(x_diff * z) <= max_dist and peak_h >= 0.01 * h_peak_h:
                        i += 1
                        z -= 1
                        # get data for this m/z value
                        if i < len(mzs):
                            x_diff, y_ratio, peak_h = fit[i]
                        else:
                            x_diff, y_ratio, peak_h = peak_picking.mz_peak_likeness(*sumall, mass / z + 1.01)
                    minz = z + 1
                    # same to the other side
                    z = h_z
                    x_diff = h_x_diff
                    peak_h = h_peak_h
                    i = highest
                    while abs(x_diff * z) <= max_dist and peak_h >= 0.01 * h_peak_h:
                        i -= 1
                        z += 1
                        # get data for this m/z value
                        if i > 0:
                            x_diff, y_ratio, peak_h = fit[i]
                        else:
                            x_diff, y_ratio, peak_h = peak_picking.mz_peak_likeness(*sumall, mass / z + 1.01)
                    maxz = z - 1
                    if minz < maxz:
                        mzs = [(z, mass/z + 1.01) for z in range(maxz, minz-1, -1)]
                    else:
                        mzs = ((h_z, mass/h_z + 1.01),)
                    refined = True
                else:
                    print("Warning: m/z values for mass %.1f do not seem to represent peaks in data" % mass, file=sys.stderr)
            # make filter function
            mzss.append(([mz[1] for mz in mzs], refined))
        if self._debug:
            for mass, mzs in zip(masses,mzss):
                print("Ions for mass %.1f:" % mass, file=sys.stderr)
                print(", ".join(["%.2f" % mz for mz in mzs[0]]), file=sys.stderr)
        return mzss

    def search_mass(self, masses, msfunc=0, normalise=False, base_peak_width=0.15, refine=True, override_z_values=None, export_path=None, return_data=False, show=True):
        """
        @param masses: Mass(es) to look for
        @type masses: number of iterable of numbers
        @param msfunc: MS function to use (default: 0)
        @type msfunc: int
        @param normalise: Normalise final XIC data to a 0-1 scale (default: False)
        @type normalise: bool
        @param base_peak_width: The base peak width expected, to decide the
                                margins around each m/z value to look at
                                (default: 0.15 Da)
        @type base_peak_width: float
        @param refine: Refine m/z choice using the marginalised spectrum,
                       which takes longer but returns much more precise
                       integrals as well as goodness of fit estimates
                       (default: True)
        @type refine: bool
        @param override_z_values: A set of z-values to use, instead of any
                                  automatically detected ones (optional)
        @type override_z_values: None or tuple of int
                                 or tuple of tuple of int
        @param export_path: Filename to export figure to (optional)
        @type export_path: str or None
        @param return_data: Return XIC data for further analysis (default: False)
        @type return_data: bool
        @param show: Whether or not to draw the pyplot plot on the screen (default: True)
        @type show: bool
        """
        # prepare
        retval = []
        if isinstance(masses, int) or isinstance(masses, float):
            masses = (masses,)
        # predict m/z values
        if override_z_values is not None:
            # no need to predict z values - saves us a lot of work..!
            try:
                override_z_values = tuple(override_z_values)
            except TypeError:
                raise TypeError("parameter 'override_z_values' must be an iterable of integers (or of iterables of integers)")
            if isinstance(override_z_values[0], int) or isinstance(override_z_values[0], float):
                # a single list of z values
                get_z_vals = lambda idx: sorted(override_z_values, reverse=True)
            else:
                # individual z value lists for each mass
                get_z_vals = lambda idx: sorted(override_z_values[idx], reverse=True)
            # use z values to calculate m/z values
            mzss = []
            for i, mass in enumerate(masses):
                mzss.append(([mass / z + 1.01 for z in get_z_vals(i)], False))
        else:
            # prepare for refinement of z values, if requested
            if refine:
                # sum over all RTs to get flat spectrum for parameter optimisations
                if self._pb:
                    pb = ProgressBar(1, name="Marginalising over RT")
                allsum = self.sum_scans(msfunc=msfunc)
                if self._pb:
                    pb.progress_one()
            # run (first round of) z value prediction
            mzss = self.predict_ions(masses, msfunc, refine=refine)
        # plot total ion chromatogram (TIC)
        print("Plotting TIC...")
        ax, line, data = self.plot_chromatogram(msfunc=msfunc, normalise=normalise, show=False)
        # start going over the masses and isolating corresponding peaks
        for mass, mzs in zip(masses, mzss):
            print("Plotting XIC for mass %.1f..." % mass)
            mzs, refined = mzs
            # predict the width of each peak
            z0 = int(round(mass / (mzs[0] - 1.01)))
            exp_widths = predict_peak_widths(mass, z0, min(len(mzs), z0), base_peak_width)
            # refine mass search further for maximum precision!
            if refine and refined and lmfit is not None:
                # we will run two rounds of ion extraction, first to
                #  determine location of peak, then re-identify the ions
                #  we need to consider, and then re-extract for precision
                data = self.get_xic([(mz-exp_w/2, mz+exp_w/2) for mz, exp_w in zip(mzs, exp_widths)], msfunc=msfunc, plot=False)[2]
                # fit single peak
                print("Refining for mass %.1f..." % mass)
                fit = SVM.fit(data[1],
                              params = SVM.guess(data[1], x = data[0]),
                              x = data[0])
                # identify central region of peak
                normfit = fit.best_fit / fit.best_fit.max()
                left = (normfit > 0.1).argmax()
                right = normfit.shape[0] - 1 - (normfit[::-1] > 0.1).argmax()
                # sum scans in peak
                partsum = self.sum_scans(data[0][left], data[0][right], msfunc=msfunc)
                if self._debug:
                    print("Potential peak identified between %.2f and %.2f, with fit amplitude %.2e and centre %.2f" % (data[0][left], data[0][right], fit.params['amplitude'], fit.params['center']), file=sys.stderr)
                # reidentify m/z values
                mzs = self.predict_ions(mass, msfunc, refine=refine, sumall=partsum)[0]
                # repeat the exercise...
                mzs = mzs[0]
                z0 = int(round(mass / (mzs[0] - 1.01)))
                exp_widths = predict_peak_widths(mass, z0, min(len(mzs), z0), base_peak_width)
                mzranges = [(mz-exp_w/2, mz+exp_w/2) for mz, exp_w in zip(mzs, exp_widths)]
                print("Ions for mass %.1f: %s" % (mass, '; '.join("%.2f - %.2f" % tuple(rng) for rng in mzranges)))
                ax, line, data = self.plot_xic(mzranges, msfunc=msfunc, normalise=normalise, label="M=%.1f, %d ions" % (mass, len(mzs)), ax=ax, show=False)
            else:
                # just plot filtered chromatogram (XIC)
                mzranges = [(mz-exp_w/2, mz+exp_w/2) for mz, exp_w in zip(mzs, exp_widths)]
                print("Ions for mass %.1f: %s" % (mass, '; '.join("%.2f - %.2f" % tuple(rng) for rng in mzranges)))
                ax, line, data = self.plot_xic(mzranges, msfunc=msfunc, normalise=normalise, label="M=%.1f, %d ions" % (mass, len(mzs)), ax=ax, show=False)
            if return_data:
                retval.append(data)
        if export_path is not None:
            pyplot.legend()
            pyplot.savefig(export_path)
        if show:
            pyplot.legend()
            pyplot.show()
            pyplot.close()
        elif export_path is not None:
            pyplot.clf()
        return retval or None

    def extract_multiple_masses(self, masses, z_max, nz, confounding_peaks = None, between_conflicting_peaks = lambda i0, m0, i1, m1: int(i0 > i1), rtrange = None, return_aggregate = True, return_chrom = False, plot = False, ax = None, show = True):
        """
        Very quick and precise way to isolate multiple species that
        roughly co-elute, and quantify the relative contributions of
        each of them to the spectrum, given that they share the z-value
        distribution and the user has prior knowledge of that distribution.

        @param masses: Macromolecule masses to search for
        @type masses: iterable of number
        @param z_max: Maximal z value to predict for
        @type z_max: int
        @param nz: Number of z values to predict for
        @type nz: int
        @param confounding_peaks: List of peaks that may be present and
                                  confound the results; peaks that overlap
                                  with these peaks will be completely
                                  ignored and not used (optional)
        @type confounding_peaks: iterable of 2-tuple of float
        @param between_conflicting_peaks: If two of the masses have
                                          an overlapping peak, this will
                                          be used to decide which species
                                          it counts towards; if this
                                          returns -1 neither will use
                                          it, if 0 the first one, if 1
                                          the second one, and if 2 both
                                          (default: first mass by order
                                                    gets the peak)
        @type between_conflicting_peaks: function (i0: int, m0: float,
                                         i1: int, m1: float) -> int
        @param rtrange: Retention time range to focus on (optional)
        @rtype rtrange: 2-tuple of float
        @param return_aggregate: Return one total per mass (otherwise, a
                                 total will be returned for each predicted
                                 peak (default: True)
        @type return_aggregate: bool
        @param return_chrom: Return chromatograms rather than integrals
                             (default: False)
        @type return_chrom: bool
        @param plot: Plot the individual/aggregate XICs (default: False)
        @type plot: bool
        @param ax: Axes to plot the XICs on (optional)
        @type ax: matplotlib.axes._subplots.AxesSubplot
        @param show: Whether or not to draw the pyplot plot on the screen (default: True)
        @type show: bool

        @return: retention time vector and a vector of integrals OR
                                             XICs OR
                                             individual peak XICs
        @rtype: tuple (numpy.ndarray, numpy.ndarray)
        """
        if plot and ax is None:
            fig, ax = pyplot.subplots()
        # predict peaks
        peaks = [naive_predict_peaks(M, z_max, nz) for M in masses]
        # identify conflicts
        keep = numpy.ones((len(peaks), max(map(len, peaks))), dtype = bool)
        for i in range(len(peaks)):
            for j in range(i + 1, len(peaks)):
                # identify overlapping ranges, by finding ranges in 'peaks'
                #  that overlap with any range in 'confounding_peaks', i.e.
                #  those ranges (r0, r1) that have a range in
                #  'confounding_peaks' (c0, c1) s.t. r0 < c1 and r1 > c0
                overlaps = (numpy.less.outer(peaks[i][:, 0],
                                             peaks[j][:, 1]) &
                            numpy.greater.outer(peaks[i][:, 1],
                                                peaks[j][:, 0]))
                # this gives the indices in the i peak axis and the j peak
                #  axis of conflicting ion ranges
                masses_i, masses_j = numpy.nonzero(overlaps)
                # so we just have to decide what to do with each
                for k in range(len(masses_i)):
                    indi = masses_i[k]
                    indj = masses_j[k]
                    decision = between_conflicting_peaks(i, peaks[i][indi],
                                                         j, peaks[j][indj])
                    if decision < 1:
                        # -1 means neither and 0 means first, so second
                        #  is out of the picture
                        keep[j, indj] = False
                        print("removing %.3f-%.3f for mass %.1f in %s, as it overlaps with a peak for another mass" % (*peaks[j][indj], masses[j], self.label))
                    if decision in (-1, 1):
                        keep[i, indi] = False
                        print("removing %.3f-%.3f for mass %.1f in %s, as it overlaps with a peak for another mass" % (*peaks[i][indi], masses[i], self.label))
        peaks = [peaks[i][keep[i]] for i in range(len(peaks))]
        # remove confounding factors
        if confounding_peaks is not None:
            final_peaks = []
            for i, mass_peaks in enumerate(peaks):
                # same method as above for identifying overlaps
                overlaps = (numpy.less.outer(mass_peaks[:, 0],
                                             confounding_peaks[:, 1]) &
                            numpy.greater.outer(mass_peaks[:, 1],
                                             confounding_peaks[:, 0]))
                bad_peaks = overlaps.sum(1) > 0
                for pmn, pmx in mass_peaks[bad_peaks]:
                    print("removing %.3f-%.3f for mass %.1f in %s, as it overlaps with confounding range" % (pmn, pmx, masses[i], self.label))
                final_peaks.append(mass_peaks[~bad_peaks])
            peaks = final_peaks
        # to avoid fetching too much data, get the minimal range required
        minmz = min([(mpeaks[0][0] if (len(mpeaks) and len(mpeaks[0])) else numpy.inf) for mpeaks in peaks])
        maxmz = max([(mpeaks[-1][-1] if (len(mpeaks) and len(mpeaks[-1])) else 0.) for mpeaks in peaks])
        # gather data
        x, y, z = self.fetch_raw_data(rtrange = rtrange,
                                      mzrange = (minmz, maxmz))
        rt = numpy.array(x)
        # for each species
        dim = [len(peaks)]
        if not return_aggregate:
            # if not aggregate data, we return each individual peak
            dim.append(max(map(len, peaks)))
        if return_chrom:
            # if full chromatograms are requested, we need the time dim
            dim.append(len(rt))
        species = numpy.zeros(dim)
        for j, mpeaks in enumerate(peaks):
            # slice peak by peak
            xics = numpy.zeros((len(mpeaks), len(rt)))
            for k in range(len(mpeaks)):
                peak = mpeaks[k]
                # isolate slice
                ind = (y >= peak[0]) & (y <= peak[1])
                dat = z[:, ind]
                # use lowest point on slice to remove potential
                #  baseline shifts
                #  TODO: verify this is always valid
                dat -= dat.min(0)
                # sum over m/z to receive chromatogram
                dat = dat.sum(1)
                # plot (if requested)
                if plot and not return_aggregate:
                    # choose colour
                    ##col = lab.CMAPS[j % len(lab.CMAPS)](k / (len(mpeaks) - 1))
                    # plot chromatogram
                    ##ax.plot(rt, dat, color = col, label = '%.1f peak %d' % (masses[j], k))
                    ax.plot(rt, dat, label = '%.1f peak %d' % (masses[j], k))
                    # add label to highest peak
                    label = '%.1f' % ((peak[0] + peak[1]) / 2)
                    mx = dat.argmax()
                    ax.text(rt[mx], dat[mx], label, color = col)
                # save data
                xics[k] = dat
            if return_aggregate:
                # aggregate
                aggrg = xics.sum(0)
                # plot integral (if requested)
                if plot:
                    ax.plot(rt, aggrg, label = '%.1f' % masses[j])
                if return_chrom:
                    species[j] = aggrg
                else:
                    # integrate - TAKING INTO ACCOUNT RT SAMPLING VARIABILITY
                    species[j] = integrate_chromatograms(rt, aggrg)
            else:
                if return_chrom:
                    species[j] = xics
                else:
                    # integrate - TAKING INTO ACCOUNT RT SAMPLING VARIABILITY
                    species[j][:corr.shape[0]] = integrate_chromatograms(rt, xics)
        if plot:
            ax.legend()
            if show:
                pyplot.show()
        return rt, species

    def orthogonalise(self, msfunc=0, min_intens=1e3):
        """
        Create orthogonal spectrum (X axis = m/z, Y axis = time, Z axis is
        still intensity) from a spectrum in this raw file.

        @param msfunc: MS function to use (default: 0)
        @type msfunc: int
        @param min_intens: Minimum intensity of ion to regard as "real" signal (default: 1e3)
        @type min_intens: float

        @return: Orthogonal spectrum
        @rtype: OrthoSpec
        """
        total_scans = self._msfunc_params[msfunc]['total_scans']
        # estimate max number of ions
        mz_iter, intensity_iter = self._get_scan(msfunc, total_scans//2)
        tmp = numpy.array(mz_iter)
        diffs = tmp[1:] - tmp[:-1]
        rl, rh = self._get_x_range(msfunc)
        max_mzs = int((rh-rl) // diffs.min())
        if self._debug:
            print("#RTs = %d\nEstimated maximum #m/z's: %d" % (total_scans, max_mzs), file=sys.stderr)
        # create ortho spectrum
        retval = OrthoSpec(self.label, total_scans, max_mzs, min_intens=min_intens)
        if self._pb:
            pb = ProgressBar(total_scans, name='Orthogonalising spectra')
        for i in range(total_scans):
            rt = self._get_rt(msfunc, i)
            mz_iter, intensity_iter = self._get_scan(msfunc, i)
            retval.add_measurement(rt, mz_iter, intensity_iter)
            if self._pb:
                pb.progress_one()
        retval.finalise()
        return retval

    @staticmethod
    def _inv_real_fft(xx, yy):
        sampling = len(xx) / (xx[-1] - xx[0])
        missing = int(numpy.ceil(xx[0] * sampling))
        datxx = numpy.concatenate((numpy.arange(missing) / sampling, xx))
        paddat = numpy.concatenate((numpy.zeros(missing), yy))
        d = 0.5 / xx[-1]
        ##freq = numpy.fft.rfftfreq(2*len(datxx)-1, d)
        rdat = numpy.fft.irfft(paddat)
        return rdat[:len(datxx)]

    def convert_to_wave(self, msfunc=0):
        rts = []
        amps = []
        total_scans = self._msfunc_params[msfunc]['total_scans']
        pb = ProgressBar(total_scans, name='Converting to wave')
        for i in range(total_scans):
            rt = self._get_rt(msfunc, i)
            mz, intens = self._get_scan(msfunc, i)
            waveamps = self._inv_real_fft(mz, intens)
            if i != total_scans - 1:
                dt = (self._get_rt(msfunc, i+1) - rt) / (len(waveamps) + 1)
            else:
                # in edge case, use last interval
                dt = (rt - self._get_rt(msfunc, i-1)) / (len(waveamps) + 1)
            rts.extend(rt + numpy.arange(len(waveamps)) * dt)
            amps.extend(waveamps)
            pb.progress_one()
        return rts, amps

class RawFile(MassSpec):
    def __init__(self, path, progress_bar=True, debug=False):
        try:
            self._info = mlrir.MassLynxRawInfoReader(path)
            self._scans = mlrsr.MassLynxRawScanReader(path)
        except Exception as e:
            raise FileNotFoundError("Could not open \"%s\" as a Waters Raw file\nOriginal exception: %s" % (path, str(e)))
        # get relevant info
        acq_date = datetime.datetime.strptime(' '.join(self._info.GetHeaderItems(range(2,4))), TIME_FMT)
        label = self._info.GetHeaderItem(10)
        num_funcs = self._info.GetNumberofFunctions()
        msfunc_params = [dict(func = self._info.GetFunctionTypeString(self._info.GetFunctionType(i)),
                              ion_mode = self._info.GetIonModeString(self._info.GetIonMode(i)),
                              total_scans = self._info.GetScansInFunction(i))
                         for i in range(num_funcs)]
        # initialise using parent's init function
        MassSpec.__init__(self, acq_date, label, msfunc_params, progress_bar, debug)

    def _get_rt(self, msfunc, i):
        """
        Get the retention time (RT) for the i's scan of the MS function msfunc.

        @param msfunc: MS function to use (default: 0)
        @type msfunc: int
        @param i: Scan number
        @type i: int

        @return: Retention time
        @rtype: number
        """
        return self._info.GetRetentionTime(msfunc, i)

    def _get_scan(self, msfunc, i):
        """
        Get scan data - m/z values and their corresponding inensities - for the
        i's scan of the MS function msfunc.

        @param msfunc: MS function to use (default: 0)
        @type msfunc: int
        @param i: Scan number
        @type i: int

        @return: m/z list and corresponding intensity list
        @rtype: tuple (list, list)
        """
        return self._scans.ReadScan(msfunc, i)

    def _get_x_range(self, msfunc):
        """
        Get the range of values on the x axis (usually m/z or wavelength) for
        the MS function msfunc.

        @param msfunc: MS function to use (default: 0)
        @type msfunc: int

        @return: x value range (minimum, maximum)
        @rtype: tuple (float, float)
        """
        rl, rh = self._info.GetAcquisitionMassRange(msfunc)
        # fix weird apparent MS bug
        rh += 1
        return rl, rh

class SimSpec(MassSpec):
    def __init__(self, acq_date, label, sim_rts, sim_scans):
        self._sim_rts = tuple(sim_rts)
        self._sim_scans = tuple(sim_scans)
        # make function parameters
        total_scans = len(sim_scans)
        func_params = {'func': 'MS', 'total_scans': total_scans}
        # initialise using parent's init function
        MassSpec.__init__(self, acq_date, label, (func_params,))

    def _get_rt(self, msfunc, i):
        # ignore msfunc
        return self._sim_rts[i]

    def _get_scan(self, msfunc, i):
        # ignore msfunc
        return self._sim_scans[i]

    def _get_x_range(self, msfunc):
        # ignore msfunc
        rl = float(int(self._sim_scans[0][0][0]))
        rh = numpy.ceil(self._sim_scans[0][0][-1])
        return rl, rh

class OrthoSpec(object):
    """
    This class represents an orthogonal mass spectrum, where the X axis
    is m/z (rather than time) and the Y axis is time.
    This allows for some analysis in the individual ion level, as well as
    global parameters derived from that.
    """
    def __init__(self, label, num_rts, max_num_mzs, min_intens=0., model=None, debug=False):
        """
        @param model: lmfit model to use (defaults to SkewedVoigtModel if None is received)
        @type model: lmfit.models.Model
        @param debug: Turn on debug printing (default: False)
        @type debug: float
        """
        self.label = label
        self.rts = []
        # XXX: This data structure is really weird and I'm sure I can do
        #      a better job if I spend some time on this
        self.__mzs = {}
        self.cur_i = 0
        self.data = numpy.zeros((num_rts, max_num_mzs))
        if model is None and lmfit is not None:
            model = lmfit.models.SkewedVoigtModel()
        self.model = model
        self.debug = debug
        self._min_intens = min_intens
        self.__final = False

    @property
    def mzs(self):
        if self.__final:
            return self.__mzs
        else:
            return tuple(self.__mzs.keys())

    @mzs.setter
    def mzs(self, mzs):
        """
        Sets m/z list and finalises the spectrum
        """
        if self.__final:
            # already finalised!
            raise RuntimeError("Trying to change the m/z list of a finalised OrthoSpec!")
        self.__mzs = mzs
        self.__final = True

    def add_measurement(self, rt, mz_iter, intensity_iter):
        if self.__final:
            # already finalised!
            raise RuntimeError("Trying to add measurements to a finalised OrthoSpec!")
        # run over observed m/z values
        for mz,intensity in zip(mz_iter, intensity_iter):
            if intensity >= self._min_intens:
                # not using setdefault in order to avoid continuously creating lists that won't be used
                idx = self.__mzs.get(mz, None)
                if idx is None:
                    idx = self.cur_i
                    self.cur_i += 1
                    self.__mzs[mz] = idx
                # add value and mz
                try:
                    self.data[len(self.rts),idx] = intensity
                except Exception as e:
                    print(mz, sorted(abs(numpy.fromiter(self.__mzs.keys(), float)-mz))[:10])
                    raise e
        # add RT to list
        self.rts.append(rt)

    def finalise(self):
        """
        Finalise the spectrum by removing unnecessary columns and reordering the data.
        """
        # remove unnecessary columns
        self.data = self.data[:, :len(self.__mzs)]
        # reorder data
        order = numpy.argsort(self.mzs)
        self.__mzs = numpy.array(self.mzs)[order]
        self.data = self.data[:, order]
        # set finalised flag
        self.__final = True

    def save(self, outpath):
        """
        Save object as a reloadable file.
        NOTE: This function assumes dictionaries are OrderedDicts (Python >= 3.7)
        """
        self.finalise()
        numpy.savez_compressed(open(outpath, 'wb'), label=self.label, rts=self.rts, mzs=self.mzs, data=self.data)

    @classmethod
    def load(cls, inpath, model=None, debug=False):
        """
        Load a saved OrthoSpec from a file.
        """
        alldata = numpy.load(open(inpath, 'rb'))
        label = alldata['label']
        rts = alldata['rts']
        mzs = alldata['mzs']
        data = alldata['data']
        retval = cls(len(rts), len(mzs), model, debug)
        retval.rts = rts
        retval.mzs = mzs
        retval.data = data
        return retval

    def orthogonalise(self):
        """
        Create a simulated MassSpec (SimSpec) object from these data.
        """
        mzs = tuple(sorted(self.mzs.keys()))
        scans = [(self.mzs, self.data[i])
                 for i in range(len(self.rts))]
        return SimSpec(datetime.datetime.now(), 'Sim:' + self.label, self.rts, scans)

    def cluster(self, cos_threshold=1/2, coverage_threshold=0.99, max_clusters=100, progress_bar=True):
        """
        Cluster all m/z values in the spectrum based on their XICs.

        This allows us to essentially deconvolute the peaks in the spectrum
        and see all the ions that co-elute in the chromatography, whether
        because they represent the same molecule (e.g. protein molecule
        with different charges) or because they are different molecules
        that are bound and not separated by HPLC.

        Written 15/11/22, this seems to me like a super promising
        algorithm for analysing mass spectra of small molecules and
        intact proteins.

        @param cos_threshold: The threshold for the normalised dot product
                              of two XICs, above which they are considered
                              similar enough to cluster together (default:
                              1/2, i.e. at most 60 degrees between the
                              vectors in R^#mz space)
        @type cos_threshold: float
        @param coverage_threshold: A stop condition for the clustering
                                   algorithm; when this fraction of the
                                   total ion count (TIC sum) is covered
                                   by clusters, all remaining m/z values
                                   are clustered into one final "misc"
                                   cluster (default: 0.99, i.e. 99%)
        @type coverage_threshold: float
        @param max_clusters: A satefy stop condition for the clustering
                             algorithm, to avoid over-analysis of the data
                             (default: 100 clusters)
        @type max_clusters: int
        @param progress_bar: Show progress bar (default: True)
        @type progress_bar: bool

        @return: List of boolean arrays, each representing the mask for
                 one cluster; the last cluster is the "miscellaneous" one
        @rtype: list of numpy.ndarray
        """
        # create a masked array version of the data, allowing us to ignore
        #  already-clustered m/z values in each round of clustering
        data = numpy.ma.masked_array(self.data)
        # a normalised version of the data is required for dot products to
        #  represent cosines of angles without further normalisation
        norm = data / (data ** 2).sum(0) ** 0.5
        # marginalised data is used to determine coverage
        marg = data.sum(0)
        # maximal coverage = sum of TIC over all m/z
        ticsum = marg.sum()
        # start clustering
        clusters = []
        if progress_bar:
           pb = ProgressBar(len(marg), name='Clustering')
        # check coverage
        while marg.sum() > (1. - coverage_threshold) * ticsum and len(clusters) < max_clusters:
            # choose the highest peak left in any single scan as refence
            ref_peak = data.max(0).argmax()
            # calculate the angles (dot products) of all available XICs to
            #  the XIC of the reference peak
            coss = norm[:, ref_peak].dot(norm)
            # identify ions that co-elute with it
            similar = (coss >= cos_threshold).data
            # this is our cluster
            clusters.append(similar)
            # mask out this cluster from all the working sets
            data[:, similar] = numpy.ma.masked
            norm[:, similar] = numpy.ma.masked
            marg[similar] = numpy.ma.masked
            # advance the progress bar to reflect how many m/z's are
            #  already in clusters
            if progress_bar:
                pb.progress_some(similar.sum())
        # "miscellaneous" cluster contains all unclustered points
        non_clust = ~(sum(clusters).astype(bool))
        clusters.append(non_clust)
        if progress_bar:
            pb.progress_some(non_clust.sum())
        # done!
        return clusters

    def export_clusters(self, cos_threshold=1/2, coverage_threshold=0.99, progress_bar=True, outpath=None, plot=True, show=False):
        """
        Cluster using self.cluster (see doc) then export each cluster
        to a file called {outpath}-cluster<num>.txt
        If outpath is None (default), the clusters will only be returned
        but not exported. This is useful for plotting them.

        @param outpath: basename for exported files (default: None)
        @type outpath: str or None
        @param plot: Should clusters be plotted? (default: True)
        @type plot: bool
        @param show: Whether or not to draw the pyplot plot on the screen (default: False) (note: this tends to be enormous)
        @type show: bool

        @return: List of cluster data and XIC pairs
        @rtype: list of tuple (numpy.ndarray, numpy.ndarray)
        """
        # call clustering function
        ###TODO: ADD MISSING PARAMETERS
        clusters = self.cluster(cos_threshold, coverage_threshold, progress_bar=progress_bar)
        # marginalise data for export
        sumscans = self.data.sum(0)
        # run through clusters, which are mutually-exclusive subsets
        #  of the m/z space
        retval = []
        # this could all be done in one loop, but doing it in two allows
        #  us to reorder the clusters by XIC peak height ~= importance
        for i, cluster in enumerate(clusters):
            # filter data to this cluster, zero out everything else
            cldata = sumscans.copy()
            cldata[~cluster] = 0.
            # get XIC
            xic = self.data[:, cluster].sum(1)
            # add cldata and xic to return list
            retval.append((cldata, xic))
        # reorder clusters by maximal height of peak in XIC
        retval = sorted(retval, key=lambda cl: cl[1].max(), reverse=True)
        # prepare for plotting / export
        if plot:
            # prepare subplots and set margins and spacing
            fig, axes = pyplot.subplots(len(clusters) + 1, 2,
                                        sharex = 'col',
                                        figsize = (15 / 2.54, 5 * len(clusters) / 2.54))
            topbot_margin = 0.5 / (len(clusters) + 1)
            fig.subplots_adjust(bottom = topbot_margin,
                                top = 1. - topbot_margin,
                                hspace = 0.25)
            # plot TIC and sum of all data
            axes[0][0].plot(self.mzs, self.data.sum(0))
            axes[0][1].plot(self.rts, self.data.sum(1))
            axes[0][0].set_ylabel('Sum All')
            axes[-1][0].set_xlabel('m/z')
            axes[-1][1].set_xlabel('time / min')
            # figure out what needs to be done
            if outpath is None:
                _name = 'Plotting'
            else:
                _name = 'Exporting & Plotting'
        else:
            if outpath is None:
                # we're done
                return retval
            else:
                _name = 'Exporting'
        if progress_bar:
            pb = ProgressBar(len(clusters), name=_name)
        # plot / export
        for i, (cldata, xic) in enumerate(retval):
            # export cluster to file
            if outpath is not None:
                with open(outpath + '-cluster%03d.txt' % i, 'w') as w:
                    for j in range(len(self.mzs)):
                        w.write('%f %f\n' % (self.mzs[j], cldata[j]))
            # plot data
            if plot:
                caxes = axes[i + 1]
                ### plot cluster data ###
                caxes[0].plot(self.mzs, cldata)
                ### label highest peak and approximate charge ###
                # find peak
                refp = cldata.argmax()
                # find next peak to the right to estimate charge #
                idx = refp
                # go to next trough
                while cldata[idx+1] < cldata[idx]:
                    idx += 1
                trough = idx
                peak_h = cldata[refp] - cldata[trough]
                # then to next substantial peak
                while idx < len(cldata) - 1 and \
                      (cldata[idx] - cldata[trough] < 0.3 * peak_h or \
                       cldata[idx+1] > cldata[idx]):
                    idx += 1
                # now we have this distance which can hint towards the charge
                est_z = int(round(1 / (self.mzs[idx] - self.mzs[refp])))
                # add label
                caxes[0].text(self.mzs[refp], cldata[refp] * 1.05,
                                '%.3f\n(z \u2248 %d)' % (self.mzs[refp],
                                                      est_z),
                                horizontalalignment='center')
                ### plot XIC of cluster ###
                caxes[1].plot(self.rts, xic)
                # label highest peak
                refp = xic.argmax()
                caxes[1].text(self.rts[refp], xic[refp] * 1.05,
                                '%.2f' % self.rts[refp],
                                horizontalalignment = 'center')
                # note down ion count
                caxes[1].text(1.0, 0.5,
                                'IC = %d' % xic.sum(),
                                fontsize = 'x-small',
                                horizontalalignment = 'right',
                                transform = caxes[1].transAxes)
                ### a bit of an ugly trick to record cluster number ###
                caxes[0].set_ylabel('Cluster %d' % i)
            if progress_bar:
                pb.progress_one()
        if show:
            pyplot.show()
        return retval

    def _fit_voigt(self, xx, yy, maxfev, non_neg_skew=True):
        """
        Fit a skewed voigt profile to data, usually an XIC or a single ion chromatogram.

        @param xx: x axis (time) data
        @type xx: iterable
        @param yy: y axis (intensity) data
        @type yy: iterable
        @param maxfev: Maximum times to run internal function in minimisation
        @type maxfev: int
        @param non_neg_skew: Only allow skew to be positive, i.e. towards the positive numbers, which I believe is always the case in LC-MS (default: True)
        @type non_neg_skew: bool

        @return: Parameters for skewed voigt profile (amplitude, center, sigma, gamma and skew)
                 If the lmfit module could not be imported, these will be rough estimates
        @rtype: dict
        """
        if not isinstance(yy, numpy.ndarray):
            yy = numpy.array(yy)
        # find peak in data (simply maximum); assume dist centre is at peak
        peakidx = yy.argmax()
        maxv = yy[peakidx]
        center = xx[peakidx]
        # find the two closest half maximum points around the peak
        halfmax = maxv / 2
        if peakidx > 0:
            idx_half_left = peakidx - 1
            while yy[idx_half_left] > halfmax and idx_half_left > 0:
                idx_half_left -= 1
            if idx_half_left > 1:
                x_half_left = xx[idx_half_left] + ((halfmax - yy[idx_half_left]) / (yy[idx_half_left+1] - yy[idx_half_left])) * (xx[idx_half_left+1] - xx[idx_half_left])
            else:
                x_half_left = xx[0]
            x_half_left = min(xx[idx_half_left], x_half_left)
        else:
            # we won't be able to fit a peak without seeing more of it...
            return dict(amplitude=maxv, center=center, sigma=0.2, skew=0)
        if peakidx < len(xx) - 1:
            idx_half_right = peakidx + 1
            while yy[idx_half_right] > halfmax and idx_half_right < len(xx) - 1:
                idx_half_right += 1
            if idx_half_right < len(xx) - 1:
                idx_half_right -= 1
                x_half_right = xx[idx_half_right] + ((halfmax - yy[idx_half_right]) / (yy[idx_half_right+1] - yy[idx_half_right])) * (xx[idx_half_right+1] - xx[idx_half_right])
            else:
                x_half_right = xx[-1]
            x_half_right = max(xx[idx_half_right], x_half_right)
        else:
            # we won't be able to fit a peak without seeing more of it...
            return dict(amplitude=maxv, center=center, sigma=0.2, skew=0)
        # calculate full width at half maximum (FWHM)
        fwhm = x_half_right - x_half_left
        # estimate the skew of the distribution
        left_half = center - x_half_left
        right_half = x_half_right - center
        try:
            skew = -1. / 0.23 * numpy.log(fwhm / right_half - 1)
        except ZeroDivisionError as e:
            print(e, x_half_right, x_half_left, center)
            raise(e)
        if non_neg_skew and skew < 0:
            skew = 0.
        # the next two factors were experimentally determined to reflect the effect of the skew on the variation and amplitude
        skew_effect = (1. + numpy.exp(-0.671 * skew))
        skew_sigma_effect = numpy.exp(.463 - .92 * skew) + 0.517
        # estimate the variation
        sigma_factor = (1 + (1 + 2 * numpy.log(2))**0.5) * 2**0.5 / skew_effect * skew_sigma_effect
        sigma = fwhm / sigma_factor
        # estimate the amplitude
        amp_factor = sigma * (2 * numpy.pi)**0.5 * skew_effect
        amplitude = maxv * amp_factor
        # fit data (if lmfit is available)
        if self.model is not None:
            params = self.model.make_params(amplitude=amplitude, center=center, sigma=sigma, skew=skew, max_nfev=maxfev)
            # limit amplitude, center (and potentially skew) to non-negative values
            params['amplitude'].min = 0
            params['amplitude'].max = 2 * maxv
            params['center'].min = -1
            params['center'].max = xx[-1] + 1
            if non_neg_skew:
                params['skew'].min = 0
            # fit
            fit = self.model.fit(yy, params, x=xx)
            if self.debug:
                print(fit.fit_report())
                print()
                print("amp: %f -> %f (factor %.2f)" % (amplitude, fit.best_values['amplitude'], fit.best_values['amplitude'] / amplitude))
                print("sig: %f -> %f (factor %.2f)" % (sigma, fit.best_values['sigma'], fit.best_values['sigma'] / sigma))
            res = fit.best_values
        else:
            if self.debug:
                print("No lmfit available, using rough estimates.")
            res = dict(amplitude=amplitude, center=center, sigma=sigma, skew=skew)
        # return fitted (or estimated) parameters for the skewed voigt profile
        return res

    def fit_chromatograms(self, maxfev=17, progress_bar=True):
        """
        Fit a voigt profile to (the highest peak of) each m/z's chromatogram.
        This helps reduce noise, accentuate hidden signals (e.g. small peak
        within larger peak) and compress the data to #ions x 5 floats.

        @param maxfev: Maximum times to run internal function in minimisation (default: 17)
        @type maxfev: int
        @param progress_bar: Whether or not to show a progress bar (default: True)
        @type progress_bar: bool

        @return: Fitted spectra
        @rtype: FittedSpec
        """
        fitted = FittedSpec(self.model, self.rts)
        if progress_bar:
            pb = ProgressBar(len(self.mzs), 'Fitting spectra')
        for mz in sorted(self.mzs.keys()):
            intensities = self.data[:,self.mzs[mz]]
            params = self._fit_voigt(self.rts, intensities, maxfev)
            fitted.add_fit(mz, params)
            if progress_bar:
                pb.progress_one()
        return fitted

class FittedSpec(object):
    VERSION_INFO = (1, 2020)

    def __init__(self, model, rts):
        self.__model = model
        self.rts = numpy.array(rts)
        self.fits = {}
        self.__simdata = None

    def add_fit(self, mz, params):
        self.fits[mz] = params

    def _simulate(self, fltr=None):
        if fltr is not None or self.__simdata is None:
            # simulate data
            simdata = {}
            for mz in filter(fltr, self.fits.keys()):
                simdata[mz] = self.__model.func(self.rts, **self.fits[mz])
            if fltr is None:
                self.__simdata = simdata
            return simdata
        return self.__simdata

    def simulate(self):
        """
        Make a simulated orthogonal spectrum from the fitted spectrum.

        @return: Simulated spectrum
        @rtype: OrthoSpec
        """
        simdata = self._simulate()
        # XXX: BROKEN
        retval = OrthoSpec('Sim:'+self.label, self.__model)
        retval.rts = self.rts.copy()
        retval.data = simdata.copy()
        return retval

    def _get_chromatogram(self, fltr=None):
        if fltr is not None or self.__simdata is None:
            # simulate data
            simdata = self._simulate(fltr)
        # either return just-calculated simdata or load from memory
        if fltr is None:
            simdata = self.__simdata
        # sum simulations
        tics = numpy.sum([numpy.array(v) for v in simdata.values()], 0)
        return self.rts, tics

    def plot_chromatogram(self, normalise=False, fltr=None, show=True):
        rts, tics = self._get_chromatogram(fltr=fltr)
        if normalise:
            tics /= tics.max()
        if fltr is None:
            label = "Simulated TIC"
        else:
            label = "Simulated XIC"
        plt = pyplot.plot(rts, tics, label=label)
        pyplot.xlabel('RT (min)')
        pyplot.ylabel(('Normalised simulated ' if normalise else 'Simulated ') + ('XIC','TIC')[fltr is None])
        if show:
            pyplot.legend()
            pyplot.show()
            pyplot.close()
        return plt

    def save(self, outpath):
        """
        File Format
        ***********
        Overall:    header (6) | model name (V) | RTs (V) | MZFits (V)
        Header:     signature (3) [0xF1773D] | version (1) | year (2)
        Model name: namelen (1) | name (namelen)
        RTs:        rtcount (4) | times (8 x rtcount)
        MZFits:     fitcount (4) | <mz (8) | amplitude (8) | center (8) | sigma (8) | skew (8)> (40 x fitcount)
        """
        w = open(outpath, 'wb')
        written = 0
        # write header
        w.write(bytes.fromhex('F177ED') + struct.pack('<BH', *self.VERSION_INFO))
        written += 6
        # write model name
        model_name = self.__model.name
        l_model_name = len(model_name)
        w.write(struct.pack('<B', l_model_name) + bytes(model_name, 'ascii'))
        written += 1 + l_model_name
        # write RT list
        l_rts = len(self.rts)
        w.write(struct.pack('<L', l_rts))
        written += 4
        w.write(struct.pack('<%dd' % l_rts, *self.rts))
        written += 8 * l_rts
        # write fit parameters list
        l_fits = len(self.fits)
        w.write(struct.pack('<L', l_fits))
        written += 4
        for mz, params in self.fits.items():
            w.write(struct.pack('<5d', mz, params['amplitude'], params['center'], params['sigma'], params['skew']))
            written += 40
        # finish
        w.close()
        return written

    @classmethod
    def load(cls, filepath):
        """
        For file format see 'save'
        """
        f = open(filepath, 'rb')
        # read header
        header = f.read(6)
        if header[:3] != bytes.fromhex('F177ED'):
        ##if header[:3] != bytes.fromhex('F177ED'):
            raise TypeError("File is not an FSP (Fitted SPectrum) file!")
        version, vyear = struct.unpack('<BH', header[3:6])
        if version > cls.VERSION_INFO[0] or vyear > cls.VERSION_INFO[1]:
            raise TypeError("The given file is from a newer version or year than this module! (this module v%d, %4d; the file claims v%d, %4d)" % (*cls.VERSION_INFO, version, vyear))
        if version == 1:
            # read model name (unused at the moment)
            namelen = struct.unpack('<B', f.read(1))[0]
            model_name = f.read(namelen)
            # TODO fix
            model = lmfit.models.SkewedVoigtModel()
            # read RT list
            rtcount = struct.unpack('<L', f.read(4))[0]
            rts = struct.unpack('<%dd' % rtcount, f.read(8*rtcount))
            # initialise return object
            retval = cls(model, rts)
            # read fit parameters list
            fitcount = struct.unpack('<L', f.read(4))[0]
            for i in range(fitcount):
                # read fit params
                mz, amp, cen, sig, skew = struct.unpack('<5d', f.read(40))
                # update return object
                retval.fits[mz] = dict(amplitude=amp, center=cen, sigma=sig, skew=skew)
        else:
            raise NotImplementedError("Does not know how to handle version %d" % version)
        return retval


def approximate_mass(*mz_series):
    """
    Approximate the mass of a macromolecule using at least two m/z
    datapoints, up to 3 significant digits, without any prior knowledge.
    This will also give an approximate charge for the highest m/z.
    Can be used to optimise the parameters of UniDec.
    """
    mz_series = [x-1.01 for x in sorted(mz_series, reverse=True)]
    possible_charges = []
    for i in range(len(mz_series) - 1):
        # a = M/z, b = M/(z+1) => z = 1 / (a/b - 1)
        possible_charges.append(int(round(1. / (float(mz_series[i]) / mz_series[i+1] - 1.))))
    # decide on most agreed-upon charge for first ion
    most_likely_z0 = int(round((numpy.array(possible_charges) - numpy.arange(len(possible_charges))).mean()))
    # M = z * (M/z)
    possible_masses = (numpy.arange(len(possible_charges)+1) + most_likely_z0) * mz_series
    return possible_masses.mean(), possible_masses.std(), most_likely_z0


def main(args):
    fnames = args.fnames
    for i in range(len(fnames)):
        rf = RawFile(fnames[i])
        bname = fnames[i].rsplit('.',1)[0]
        outname = rf.label.replace(' ','_').replace('/','').replace('*','star').replace(':',';')
        try:
            wv1 = rf.plot_absorbance(280, show=False)
            wv2 = rf.plot_absorbance(354, show=False)
            wv3 = rf.plot_absorbance(564, show=False)
            if args.plot_only:
                pyplot.show()
            else:
                pyplot.savefig('.'.join((bname, outname, 'A%d' % wv1, 'A%d' % wv2, 'A%d' % wv3, 'pdf')))
            pyplot.close('all')
        except ValueError as e:
            print("Warning: %s" % e, file=sys.stderr)
        ### old version ###
        ##rf.export_peaks(bname + '.' + outname, plot_peaks=True)
        ### new version ###
        # plot and save chromatogram
        rf.plot_chromatogram(show = args.plot_only)
        if not args.plot_only:
            pyplot.savefig('.'.join((bname, outname, 'pdf')))
            # filter out excess data to clean it up and save space
            ort = rf.orthogonalise()
            # save cleaned, compressed datafile for future analyses
            ort.save('.'.join((fnames[0].rsplit('.',1)[0], 'ort')))
            # divide m/z's into clusters, plot and export each to a file
            ort.export_clusters(outpath='.'.join((bname, outname)), plot=True, show=False)
            # export cluster plots
            pyplot.savefig('.'.join((bname, outname, 'clusters', 'pdf')))
            pyplot.close('all')
    return 0

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('fnames', help="MassLynx .raw file (directory)", nargs='*')
    parser.add_argument('-p', '--plot_only', action="store_true", help="Only plot chromatogram without further processing")
    # parse arguments
    args = parser.parse_args(args)
    sys.stderr.flush()
    # check arguments
    if not args.fnames:
        parser.print_help()
        fname = input("Enter RAW file name: ")
        if not fname:
            return 1
        else:
            args.fnames = fname.strip('"').strip("'"),
    # return
    return args


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    if args is None:
        sys.exit(1)
    sys.exit(main(args))
