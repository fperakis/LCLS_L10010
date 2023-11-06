import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from scipy.stats import binned_statistic, binned_statistic_dd

import h5py

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%% CORRELATION PLOTTING FUNCTION %%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

def pol1(x, intercept, m):
    return (intercept + x*m)


def gauss(x, mean, sigma, height=1.):
    if sigma==0:
        return 1e32
    return (height* np.exp(-0.5 * ((x-mean)/sigma)**2))


def correlationFit(var1, var2, labels=[None,None], ax=None, use_hist=False):
    # nanfilt on var1 and var2
    filt = np.logical_not(np.isnan(var1))
    filt = np.logical_and(filt, np.logical_not(np.isnan(var2)))
    var1 = var1[filt]
    var2 = var2[filt]
    
    # initial parameters for pol1 & fit
    var1_low = np.nanpercentile(var1,0.5)
    var1_high = np.nanpercentile(var1,99.5)
    d_var1 = var1_high - var1_low

    var2_low_vals = var2[np.argwhere( ((var1_low-d_var1*0.05) < var1) & (var1 < (var1_low+d_var1*0.05)) )]
    var2_high_vals = var2[np.argwhere( ((var1_high-d_var1*0.05) > var1) & (var1 < (var1_high+d_var1*0.05)) )]
    var2_low = var2_low_vals.mean()
    var2_high = var2_high_vals.mean()
#     print(var2_low)
#     print(var2_high)
    d_var2 = var2_high_vals.mean()-var2_low_vals.mean()

    par_est_pol1 = (var1_low-d_var2/d_var1*var2_low), d_var2/d_var1
    fit_params_p1, fit_cov_p1 = curve_fit(pol1, var1, var2, par_est_pol1)
    
    if ax is not None:
        plot_var1 = np.linspace(np.nanmin(var1), np.nanmax(var1), 200)
        plot_var1_pol1 = pol1(plot_var1, fit_params_p1[0], fit_params_p1[1])
        
        fit_label = '{}: {:.5f}*({}) + {:.5f}'.format(labels[1], fit_params_p1[1], \
                                              labels[0], fit_params_p1[0])
        
        if np.percentile(var1,99)> ax.get_xlim()[1]:
            ax.set_xlim(0, 1.2*np.percentile(var1,99))
        if use_hist==False:
            ax.plot(var1, var2, '.', label=fit_label, markersize=0.5, color='purple')
            ax.legend(loc='upper left')
        else:
            bins = [np.linspace(np.percentile(var1,1), 1.2*np.percentile(var1,99),100), 
                    np.linspace(np.percentile(var2,1), 1.2*np.percentile(var2,99),100)]
            ax.hist2d(var1, var2, bins=bins, cmap='magma', label=fit_label)
        
        ax.plot(plot_var1, plot_var1_pol1, '--', color='orange', linewidth=1.5)
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
    return ax, (fit_params_p1, fit_cov_p1)



def residualPlot(var1, var2, fitResult, labels=[None, None], ax=None, relative_residual=False, \
                 nBins=100, nBinsRes=100):
    fit_params_p1 = fitResult[0]
    var1_pol1 = pol1(var1, fit_params_p1[0], fit_params_p1[1] )

    #calculate the residuals & the 'relative' residuals
    if relative_residual:
        res = (var2-var1_pol1)/var2
        resLabel = 'Rel. residual'
    else:
        res = (var2-var1_pol1)
        resLabel = 'Residual'
    res = res.astype(np.float64) # why do I even need this hack?!
    resfilt = np.isfinite(res)
    var1 = var1[resfilt]
    res = res[resfilt]
    
    # estimate gauss parameters
    resMed = np.median(res)
    resWidth = np.percentile(res, 80)-np.percentile(res, 15)
    thisresDim = (-3*resWidth, 3*resWidth)
    
    bins = [100, np.linspace(thisresDim[0], thisresDim[1], 100)]
    ax[0].hist2d(var1, res, bins=bins, cmap='magma')
    ax[0].axhline(y=0, color='yellow', linewidth=1)
    ax[0].set_xlim(0, 1.2*np.percentile(var1,99))
    ax[0].set_ylim(thisresDim)
    ax[0].set_xlabel('{}'.format(labels[0]))
    ax[0].set_ylabel('{} {}'.format(resLabel, labels[1]))
    
    var1_BinEdges = np.linspace(np.nanmin(var1), np.nanmax(var1), nBins) 
    idx = np.digitize(var1, var1_BinEdges)
    nEntries_Bin = np.bincount(idx)
    resBinned = np.bincount(idx, weights=res)
  
    resHis = np.histogram(res, np.linspace(thisresDim[0], thisresDim[1], nBinsRes))
#     resHisDim = (0, max(resHisDim[1], resHis[0].max()*1.05))
    ax[1].plot(0.5*(resHis[1][:-1]+resHis[1][1:]), resHis[0], 'o', color='orange')
    ax[1].set_xlabel(resLabel)
    ax[1].set_ylabel('N_events')
    
    params_estimate = resMed, resWidth, max(resHis[0]) 
    xVals = 0.5*(resHis[1][1:]+resHis[1][:-1])
    
    try:
        fit_params, fit_cov = curve_fit(gauss, xVals, resHis[0], params_estimate)
        plot_resFit = np.linspace(thisresDim[0], thisresDim[1], 200)
        plot_res_gauss = gauss(plot_resFit, fit_params[0], fit_params[1], fit_params[2] )
        label = 'Width: {:.5f}'.format(fit_params[1])
        ax[1].plot(plot_resFit, plot_res_gauss, color='red', linewidth=2, label=label)
        ax[1].set_ylim(-0.05*np.max(plot_res_gauss), 1.25*np.max(plot_res_gauss))
        ax[1].legend(loc='upper right')
    except:
        fit_params, fit_cov = (None, None)
    
    return ax, (fit_params, fit_cov)


def correlation_residual_plot(var1, var2, labels=[None,None], axs=None, relative_residual=False, 
                         use_hist=False):
    if axs is None:
        fig = plt.figure(constrained_layout=True, figsize=(12,9))
        gs = GridSpec(3, 5, figure=fig)
        ax1 = fig.add_subplot(gs[:2, :])
        ax1.set_title('{} vs {}'.format(labels[0], labels[1]), fontsize=18)
        ax2 = fig.add_subplot(gs[2, :3])
        ax3 = fig.add_subplot(gs[2, 3:])
        axs = [ax1, ax2, ax3]
        
    ax1, fitRes = correlationFit(var1, var2, labels=labels, ax=axs[0], 
                                 use_hist=use_hist)
    axs, fitResG = residualPlot(var1, var2, fitRes, labels=labels, ax=[axs[1], axs[2]], \
                                relative_residual=relative_residual, nBins=100, nBinsRes=100)
    axs = [ax1, axs[0], axs[1]]
    return axs, fitRes, fitResG



"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ANALYSIS FUNCTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
def bin_data_ndim(data, bin_keys=None, bin_centers=None, statistic='sum', weight_key=None, weight_fun=None):
    """    
    Inputs:
        data:
            Pandas dataframe or dictionary
        bin_key: list
            key in df to bin (e.g. jitter corrected delay).
        bin_centers: list
            definition of bins. If None, bins to the unique values (i.e. average scan steps). If given, must be the same
            length as bin_key
        statistic:
            function to apply to the binned data. Default: sum
        weight_key:
            key to use for the weights. If none weights of 1 are used.
        weight_fun:
            function to apply to all weights. Must take numpy array input as argument and return 
            an array of the same size (example: np.square, np.sqrt, etc).
    
    Outputs:
        Result are stored in a dictionary whose columns are named by the original key + the statistic label.
        The bin centers are returned as axisName_bin
    """
    
    """ Compute bin edges """
    if isinstance(bin_keys, str):
        bin_keys = [bin_keys]
    if bin_centers is None:
        bin_centers = [None for key in bin_keys]
    if not isinstance(bin_centers, list):
        bin_centers = [bin_centers]
    
    if len(bin_keys)!=len(bin_centers):
        raise TypeError("bin_keys and bin_centers must be the same length.")
        
    bin_edges = []
    bin_axis = []
    for ii, bin_key in enumerate(bin_keys):
        bins = bin_centers[ii]
        if bins is None:
            bins = np.unique(data[bin_key])
            bin_centers[ii] = bins
        bins = np.asarray(bins)
        binDiff = np.diff(bins)
        binDiff = np.append(binDiff, binDiff[-1])
        binEdges = bins+binDiff/2
        binEdges = np.append(binEdges[0]-binDiff[0], binEdges)
        bin_edges.append(binEdges)
        bin_axis.append(data[bin_key])
    
    """ Bin data """
    stat_data = [val for key, val in data.items()]
    stats, bin_edges, bin_idx = binned_statistic_dd(bin_axis, stat_data, 
                                                    bins=bin_edges,
                                                    statistic=statistic,
                                                    expand_binnumbers=True)
    counts, _, _ = binned_statistic_dd(bin_axis, None,
                                    bins=bin_edges,
                                    statistic='count',
                                    expand_binnumbers=True)
    binned = {}
    binned['counts'] = counts
    for ii,key in enumerate(bin_keys):
        binned[key+'_bin'] = bin_centers[ii]
    for ii,key in enumerate(data.keys()):
        binned[key] = stats[ii]
    return binned, binEdges


def weighted_bin_data(data, bin_key=None, bin_centers=None, weight_key=None, weight_fun=None, covariance=None):
    """    
    Inputs:
        data:
            Pandas dataframe or dictionary
        bin_key: 
            key in df to bin (e.g. jitter corrected delay).
        bin_centers:
            definition of bins. If None, bins to the unique values (i.e. average scan steps).
        weight_key:
            key to use for the weights. If none no weights of 1 are used.
        weight_fun:
            function to apply to all weights. Must take numpy array input as argument and return 
            an array of the same size (example: np.square, np.sqrt, etc).
        covariance:
            list with two quantities we want the co-variance of.
    
    Outputs:
        All keys binned according to Inputs.
        Result are stored in a dictionary whose columns are named by the original key + the statistic label.
    """
    
    """ Compute bin edges """
    if bin_centers is None:
        bin_centers = np.unique(data[bin_key])
    bin_centers = np.asarray(bin_centers)
    binDiff = np.diff(bin_centers)
    binDiff = np.append(binDiff, binDiff[-1])
    binEdges = bin_centers+binDiff/2
    binEdges = np.append(binEdges[0]-binDiff[0], binEdges)
    
    """ Bin bin_key """
    bin_idx = np.digitize(data[bin_key], bins=binEdges)
    idx_in_range = np.arange(bin_centers.size)+1 # only take bin_idx that are actually in the bins
    
    binned = {}
    binned[bin_key] = bin_centers
    binned['count'] = np.zeros((idx_in_range.size))
    
    """ Set weights """
    if weight_key is not None:
        weights = data[weight_key]
        if weight_fun is not None:
            weights = weight_fun(weights)
    
    """ Bin all keys """
    for key in data.keys():
        if data[key].ndim>1:
            binned[key+'_mean'] = np.zeros((idx_in_range.size, *data[key].shape[1:]))
            binned[key+'_var'] = np.zeros((idx_in_range.size, *data[key].shape[1:]))
            binned[key+'_sum'] = np.zeros((idx_in_range.size, *data[key].shape[1:]))
        else:
            binned[key+'_mean'] = np.zeros((idx_in_range.size))
            binned[key+'_var'] = np.zeros((idx_in_range.size))
            binned[key+'_sum'] = np.zeros((idx_in_range.size))
    if covariance is not None:
        binned['covariance'] = np.zeros((idx_in_range.size,2,2))

    for ii,idx in enumerate(idx_in_range):
        inbin = bin_idx==idx
        binned['count'][ii] = inbin.sum()
        if weight_key is not None:
            binweights = weights[inbin]
        else:
            binweights = None
            
        for key in data.keys():
            mean, var = weighted_avg_and_var(data[key][inbin], weights=binweights)
            binned[key+'_mean'][ii] = mean
            binned[key+'_var'][ii] = var
            sum_in_bin = np.nansum(data[key][inbin])
            binned[key+'_sum'][ii] = sum_in_bin
        
        if covariance is not None:
            cov_data = np.stack((data[covariance[0]][inbin], data[covariance[1]][inbin]), axis=0)
            binned['covariance'][ii] = np.cov(cov_data)
    
    return binned, binEdges


def roi_bkgRoi(image_in, roi, bkg_roi, safe_extend=[5,5]):
    """
    Returns the total intensity (sum of all pixels) in a roi and corresponding background based on a background
    roi from an input image. The function checks for overlap between the roi and bkg_roi, and takes it into account.
    """

    # check for intersection
    temp_roi = roi + np.array([[-safe_extend[0], safe_extend[0]], 
                               [-safe_extend[0], safe_extend[0]]])  # extended roi for safe background intensity
    fintersect = (temp_roi[0][0] < bkg_roi[0][1] and bkg_roi[0][0] < temp_roi[0][1] and
                  temp_roi[1][0] < bkg_roi[1][1] and bkg_roi[1][0] < temp_roi[1][1])

    if fintersect:
        intersect = [[max(temp_roi[0][0], bkg_roi[0][0]), min(temp_roi[0][1], bkg_roi[0][1])],
                     [max(temp_roi[1][0], bkg_roi[1][0]), min(temp_roi[1][1], bkg_roi[1][1])]]
    else:
        intersect = []

    temp_roi = intersect
#    return temp_roi

    if len(image_in.shape) == 2:
        image_in = image_in[None,:,:]
        
    img_roi = image_in[:, roi[0][0]:roi[0][1], roi[1][0]:roi[1][1]]
    img_bkg_roi = image_in[:, bkg_roi[0][0]:bkg_roi[0][1], bkg_roi[1][0]:bkg_roi[1][1]]
    if fintersect:
        img_temp_roi = image_in[:, temp_roi[0][0]:temp_roi[0][1], temp_roi[1][0]:temp_roi[1][1]]

    size_roi = img_roi.shape[1] * img_roi.shape[2]
    size_bkg_roi = img_bkg_roi.shape[1] * img_bkg_roi.shape[2]
    if fintersect:
        size_temp_roi = img_temp_roi.shape[1] * img_temp_roi.shape[2]

    intensity_roi = np.nansum(img_roi,axis=(1,2))
    intensity_bkg_roi = np.nansum(img_bkg_roi,axis=(1,2))
    if fintersect:
        intensity_temp_roi = np.nansum(img_temp_roi,axis=(1,2))

        intensity_bkg_roi = (intensity_bkg_roi-intensity_temp_roi) / (size_bkg_roi-size_temp_roi) * size_roi
    else:
        intensity_bkg_roi = intensity_bkg_roi / size_bkg_roi * size_roi
    
    intensity = np.asarray(intensity_roi)
    bkg = np.asarray(intensity_bkg_roi)
        
    return np.squeeze(intensity), np.squeeze(bkg)



"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% STATISTICS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
def weighted_avg_and_var(x, weights=None, axis=0):
    """
    Return the weighted average and variance.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(x, weights=weights, axis=axis)
    variance = np.average((x-average)**2, weights=weights, axis=axis)
    return average, variance


def variance_of_ratio(mean1, mean2, var1, var2, cov):
    """ Ref: http://stat.cmu.edu/~hseltman/files/ratio.pdf """
        # handle the case where the cov is given as a matrix
    if cov.ndim==3:
        cov = cov[:,0,1]
    elif cov.ndim==2:
        cov = cov[0,1]
    
    v1 = var1/mean1**2
    v2 = var2/mean2**2    
    v3 = cov/mean1/mean2
    return (mean1/mean2)**2 * (v1+v2-2*v3)


def combine_mean(means, counts):
    """ Combine the means of subsets of the data.
    """
    k = counts
    n = np.sum(k, axis=0)
    return np.sum(k*means, axis=0)/n


def combine_variance(means, variances, counts):
    """
    Combine variances of subsets of the data.
    Ref: https://stats.stackexchange.com/questions/10441/how-to-calculate-the-variance-of-a-partition-of-variables
    
    Parameters
    ---------
    means : np.ndarray
        Mean of the variable for each data subset
    variances : np.ndarray
        Variance of the variable for each data subset
    counts
        Number of elements in each subset
        
    Returns
    -------
    Combined variance
    """
    k = counts
    g = len(k)
    n = np.sum(k, axis=0)
    mean_all = combine_mean(means, counts)
    # print(mean_all)
    # mean_all = np.average(means, weights=k)
    var = np.sum((k-1)*variances + k*(means-mean_all)**2, axis=0)
    var = 1/(n-1)*var
    return var


def combine_covariance(means_a, means_b, covs, counts):
    """
    Combine the covariances of subsets of the data.
    
    Parameters
    ---------
    means_a : np.ndarray
        Mean of the variable a for each data subset
    means_b : np.ndarray
        Mean of the variable b for each data subset
    covs : np.ndarray
        List of the covariance between variable a and b. Can be either 
        the full covariance matrix (output of np.cov) or just the off-diagonal 
        term. Only the off-diagonal term is considered for the combination.
    counts
        Number of elements in each subset
        
    Returns
    -------
    Combined covariance (only the off-diagonal term)
    """
    # Make all inputs np arrays
    for arr in [means_a, means_b, covs, counts]:
        arr = np.asarray(arr)
    
    # get only off-diag term if cov matrices are passed
    if covs.ndim == 3:
        covs = covs[:,0,1]
    
    k = counts
    g = len(k)
    n = np.sum(k, axis=0)
    mean_all_a = combine_mean(means_a, k)
    mean_all_b = combine_mean(means_b, k)
    
    c1 = np.sum((k-1)*covs, axis=0)
    c2 = np.sum(k*(mean_all_a*mean_all_b - mean_all_b*means_a \
                   - mean_all_a*means_b + means_a*means_b), axis=0)
    return (c1+c2)/(n-1)



"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% DATA SAVING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
def dict_to_h5(d, fname):
    """
    Recursively save dictionary as h5. 
    Parameters:
    d: dictionary
    fname: filename to which dictionary must be saved
    """
    with h5py.File(fname, 'w') as f:
        _dict_to_h5(d,f)
    return

def _dict_to_h5(d, f):
    """ Recursively save dictionary as h5 """
    for k, v in d.items():
        if isinstance(v, dict):
            group = f.create_group(k)
            _dict_to_h5(v, group)
        elif isinstance(v, int) or isinstance(v, float):
            v = [v]
            f.create_dataset(k, data=v)
        elif isinstance(v, np.ndarray) or isinstance(v, list):
            f.create_dataset(k, data=v)
        else:
            print('Data type not understood for key {}.\n'.format(key))
            print(v)
    return






"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% OLD %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
def binData(data, bin_centers=None, bin_key='dl', statkeys = None,
            statfuns=[np.nanmean,np.nanmedian,np.nanstd,len], statlbls=['_mean','_median','_std','_count']):
    """    
    Inputs:
        data:
            Pandas dataframe or dictionary
        bin_key: 
            key in df to bin (e.g. jitter corrected delay).
        bin_centers:
            definition of bins. If None, bins to the unique values (i.e. average scan steps).
        statkeys:
            keys for which to apply statistics; None means all (default).
        statfuns/statlbls:
            function list/string label list of non-weighted statistics functions to be applied.
    
    Outputs:
        Result are stored in a dictionary whose columns are named by the original key + the statistic label.
    """
    
    if statkeys is None:
        statkeys=data.keys()
        
    if bin_centers is None:
        bin_centers = np.unique(data[bin_key])
    
    bin_centers = np.asarray(bin_centers)
        
    dict_out = dict()
    bin_name = bin_key
    
    """ Compute bin edges """
    binDiff = np.diff(bin_centers)
    binDiff = np.append(binDiff, binDiff[-1])
    binEdges = bin_centers+binDiff/2
    binEdges = np.append(binEdges[0]-binDiff[0], binEdges)
    
    dict_out[bin_key] = bin_centers
        
    for nkey,key in enumerate(statkeys):
        for fun,lbl in zip(statfuns,statlbls):
            stat, temp, bin_number = binned_statistic(data[bin_key], data[key], 
                                                      statistic=fun, bins=binEdges)
            dict_out[key+lbl] = stat
    
    return dict_out, binEdges