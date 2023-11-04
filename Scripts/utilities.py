import numpy as np
import matplotlib.pyplot as plt
import h5py
import psana
from tqdm import tqdm

h = 4.135667516*1e-18 # kev*sec
c = 3*1e8 # m/s

def wavelength(energy):
    return h / energy

# -- visit function for h5 viewing the structure of files
def visit_func(name, node):
    '''
    Return all groups and datasets name and shapes of h5 file called name
    '''
    if isinstance(node, h5py.Group):
        print(node.name)
    elif isinstance(node, h5py.Dataset):
        if (node.dtype == 'object') :
            print (node.name, 'is an object Dataset')
        else:
            print('\t', node.name, node.shape)
    else:
        print(node.name, 'is an unknown type')

def reconstruct_img(photons_i, photons_j, shape):
    nx, ny = shape
    phot_img, _,_ = np.histogram2d(photons_j+0.5, photons_i+0.5, bins=[np.arange(nx+1),np.arange(ny+1)])     
    return phot_img

def getProb_img(photons_i, photons_j, mask, Np=12):
    nx,ny = mask.shape
    #nroi as the number of rois
    nroi = int(mask.max())
    p = np.zeros((1, Np,nroi))
    ave0  = reconstruct_img(photons_i, photons_j, (nx, ny))
    for roi_num in range(1,nroi+1):
        npx = np.where(mask == roi_num)[0].size
        p[0,:,roi_num-1] = np.bincount(np.int32(ave0[mask==roi_num].ravel()),minlength = Np)[:Np]
        p[0,-1,roi_num-1] = ave0[mask == roi_num].sum()
        p[0,:,roi_num-1] /=float(npx)
    return p

def intensity_influence_estimator(ps, kbar, branch_filter, nroi = 65000,  color = None, label = None, binmin = 1e-5, binmax = None, nbin_edge = 10):
#     ps = ps[branch_filter].copy()
#     kbar = kbar[branch_filter].copy()
    binmin = np.log10(binmin)
    if binmax is None:
        binmax = np.log10(kbar.max())
    else:
        binmax = np.log10(binmax)
#     print (binmin, binmax)
    kbar_bin = np.logspace(binmin, binmax,  nbin_edge)
#     print (kbar_bin)
    w = np.digitize(kbar, kbar_bin)
#     print (w)
    ps_bin = np.zeros((nbin_edge-1, 3))
    ps_bin_error = np.zeros((nbin_edge-1, 3))

    kbar_bin = np.zeros((nbin_edge-1))
    kbar_bin_error = np.zeros((nbin_edge-1))
    p2_shot_noise_error = np.zeros((nbin_edge-1))
    
    for i in range(1, nbin_edge):
        w0 = np.where(w == i)[0]
        nframe = w0.size
#         print (nframe)
        if nframe < 100: ps_bin[i-1] = np.nan
        else:
            ps_bin[i-1] = np.mean(ps[w0], axis = 0)
            ps_bin_error[i-1] = np.std(ps[w0], axis = 0)/np.sqrt(nframe)
        if ps_bin_error[i-1, 2] == 0: ps_bin[i-1] = np.nan
        kbar_bin[i-1] = np.mean(kbar[w0], axis = 0)
        kbar_bin_error[i-1] = np.std(kbar[w0], axis = 0)
        # shot noise error, not alpha implemented right now
        p2_shot_noise_error[i-1] = np.sqrt(ps_bin[i-1, 2]/nroi/nframe)
        
    beta = ps_bin[:,2]*2/kbar_bin**2-1
    delta_beta = ps_bin_error[:,2]*2/kbar_bin**2
    if color is None: color = 'b'
    plt.errorbar(kbar_bin, beta, yerr= delta_beta, label = label, capsize = 2, fmt = 'o-', color = color)
#     print (beta, delta_beta)
    return kbar_bin, kbar_bin_error, ps_bin, ps_bin_error, p2_shot_noise_error

def p0_dist(beta, kbar):
    return (1/(beta*kbar+1)**(1./beta))

def p1_dist(beta, kbar):
    return kbar - (1+beta)*kbar**2

def p2_dist(beta, kbar):
    return 0.5*(1+beta)*kbar**2

def p_dist(beta, kbar):
    p_calc = np.zeros(kbar.shape)
    p_calc[:,1] = p1_dist(beta, kbar[:,1])
    p_calc[:,2] = p2_dist(beta, kbar[:,2])
    p_calc[:,0] = 1 - p_calc[:,1] - p_calc[:,2]
    return p_calc

def chisqs(ps, kbar, beta, npx):
    #chisqs only based on ps
    kbar = np.tile(kbar,(3,1)).transpose()
#     print (kbar.shape)
    return -2*np.nansum(ps*npx*np.log(p_dist(beta, kbar)/ps))

def getContrast(ps, kbar, npx):
    betas = np.linspace(-0.998,1, 1000)
    chi2 = np.zeros(betas.size)
    for ii,beta in enumerate(betas):
        chi2[ii] = chisqs(ps = ps, kbar = kbar ,beta = beta, npx = npx)
    pos = np.argmin(chi2)
    beta0 = betas[pos]
    #curvature as error analysis
    dbeta = np.diff(betas)[0]
    delta_beta = np.sqrt(dbeta**2/(chi2[pos+1]+chi2[pos-1]-2*chi2[pos]))
    return betas, chi2, beta0, delta_beta