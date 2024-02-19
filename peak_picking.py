import numpy
import scipy.stats
try:
    import lmfit
except (ImportError, ModuleNotFoundError) as e:
    lmfit = None
    print("Warning: Will produce less precise results as lmfit is not installed.", file=sys.stderr)


DEFAULT_WINDOW_SIZE = 10
MAD_TO_STD_NORMAL_FACTOR = 1.4826


def get_median_median_absolute_difference(seq, window_size=DEFAULT_WINDOW_SIZE):
    """
    This function runs a sliding window on a sequence, calculated MAD (median
    absolute difference) in each window, and returns the median of all MADs.
    I believe this should give a rather robust measure of the noise level.

    See following wikipedia articles on MADs and robust statistics:
    https://en.wikipedia.org/wiki/Median_absolute_deviation
    https://en.wikipedia.org/wiki/Robust_measures_of_scale
    https://en.wikipedia.org/wiki/Robust_statistics
    Also see this StackOverflow discussion on peak detection:
    https://stackoverflow.com/questions/22583391/peak-signal-detection-in-realtime-timeseries-data
    """
    MADs = numpy.array([scipy.stats.median_absolute_deviation(seq[i:i+window_size]) for i in range(len(seq)-window_size)])
    return numpy.median(MADs)

class Peak(object):
    """
    Adapted from https://www.sthu.org/blog/13-perstopology-peakdetection/index.html
    """
    def __init__(self, startidx):
        self.born = self.left = self.right = startidx
        self.died = None

    def get_persistence(self, seq):
        return numpy.inf if self.died is None else seq[self.born] - seq[self.died]

def get_persistent_homology(seq, min_persistence=None):
    """
    Adapted from https://www.sthu.org/blog/13-perstopology-peakdetection/index.html
    """
    peaks = []
    # Maps indices to peaks
    idxtopeak = [None for s in seq]
    # Sequence indices sorted by values
    indices = range(len(seq))
    indices = sorted(indices, key = lambda i: seq[i], reverse=True)

    # Process each sample in descending order
    for idx in indices:
        lftdone = (idx > 0 and idxtopeak[idx-1] is not None)
        rgtdone = (idx < len(seq)-1 and idxtopeak[idx+1] is not None)
        il = idxtopeak[idx-1] if lftdone else None
        ir = idxtopeak[idx+1] if rgtdone else None

        # New peak born
        if not lftdone and not rgtdone:
            peaks.append(Peak(idx))
            idxtopeak[idx] = len(peaks)-1

        # Directly merge to next peak left
        if lftdone and not rgtdone:
            peaks[il].right += 1
            idxtopeak[idx] = il

        # Directly merge to next peak right
        if not lftdone and rgtdone:
            peaks[ir].left -= 1
            idxtopeak[idx] = ir

        # Merge left and right peaks
        if lftdone and rgtdone:
            # Left was born earlier: merge right to left
            if seq[peaks[il].born] > seq[peaks[ir].born]:
                peaks[ir].died = idx
                peaks[il].right = peaks[ir].right
                idxtopeak[peaks[il].right] = idxtopeak[idx] = il
            else:
                peaks[il].died = idx
                peaks[ir].left = peaks[il].left
                idxtopeak[peaks[ir].left] = idxtopeak[idx] = ir

    if min_persistence:
        peaks = [peak for peak in peaks if peak.get_persistence(seq) >= min_persistence]

    return peaks

def pick_peaks(seq, mmad_window_size=DEFAULT_WINDOW_SIZE):
    """
    Given a chromatogram, return a likely partition of the sequence to sub-
    sequences, each containing exactly one peak.
    """
    medmad = get_median_median_absolute_difference(seq, mmad_window_size)
    # use 3 "standard deviations" (estimated using MAD) as minimum peak height
    peaks = get_persistent_homology(seq, 3 * MAD_TO_STD_NORMAL_FACTOR * medmad)
    # death points of peaks give us good partition points
    partitions = sorted([peak.died for peak in peaks if peak.died is not None])
    return partitions

def mz_peak_likeness(spec_x, spec_y, mz, margin = 0.5, model = lmfit.models.VoigtModel()):
    """
    Try fitting a Voigt model peak to data around the given m/z (+-margin).
    If the m/z value represents a peak, the intensity at that point should
    be close to or above the height of the fitted peak, and it should be
    close to the centre.
    The function therefore returns the distance from the centre and the
    ratio between the value at 'mz' and the height of the fitted peak, as
    well as the latter for additional information.

    @param spec_x: The x (m/z) axis of a mass spectrum
    @type spec_x: numpy.ndarray
    @param spec_y: The y (intensity) axis of a mass spectrum
    @type spec_y: numpy.ndarray
    @param mz: The m/z value to test for "peakness"
    @type mz: float
    @param margin: The margin to either direction from the chosen m/z value
                   to use as reference (default: 0.5 Da/e)
    @type margin: float
    @param model: The model to use for fitting the peak (default: Voigt)
    @type model: lmfit.models.Model

    @return: difference from m/z to centre of fitted peak,
             relative intensity at m/z compared to height of fitted peak,
             height of fitted peak
    @rtype: tuple (float, float, float)
    """
    if lmfit is None:
        raise NotImplementedError("fit_peak relies on lmfit, which does not seem to be installed")
    # find edges of window to look at
    left = abs(spec_x - (mz - margin)).argmin()
    right = abs(spec_x - (mz + margin)).argmin()
    if right - left < 3:
        # too narrow!
        return 0., 0., 0.
    # isolate window
    xx = spec_x[left:right+1]
    yy = spec_y[left:right+1]
    # identify baseline
    mid = len(yy) // 2
    leftmin = yy[:mid].argmin()
    rightmin = mid + yy[mid:].argmin()
    line_m = (yy[rightmin] - yy[leftmin]) / (rightmin - leftmin)
    line_n = yy[leftmin] - line_m * leftmin
    # subtract baseline
    yy = yy - (numpy.arange(len(yy)) * line_m + line_n)
    # fit (Voigt) peak
    params = model.guess(yy, x=xx)
    fit = model.fit(yy, params, x=xx)
    # return difference of chosen point from peak m/z value and height
    selected = yy[abs(xx - mz).argmin()]
    peak_h = fit.params['height'].value
    xdiff = mz - fit.params['center'].value
    ydiff = selected - peak_h
    return xdiff, selected / peak_h, peak_h
