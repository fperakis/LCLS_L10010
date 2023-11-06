import numpy as np
import matplotlib.pyplot as plt
import h5py
import psana
from tqdm import tqdm


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



def calc_Q_pixel(E = 9.5, ps = 50e-6, L = 5.5):
    lam = 12.398/E
    k = np.pi*2/lam
    tth = ps/L
    delta_Q = tth*k
    return delta_Q

def getROI(shape,center,rmin=345, rmax = 385):
    #for center, index 0 is y and index 1 is x
    x, y = np.indices((shape[0],shape[1]))
    r = np.hypot(x-center[1],y-center[0])
    mask = x*0
    mask[(r>rmin)&(r<rmax)] = 1
    return mask

def getCenter(p1,p2,p3):
    x1, y1 = np.float_(p1)
    x2, y2 = np.float_(p2)
    x3, y3 = np.float_(p3)
    A = x1*(y2-y3)-y1*(x2-x3)+x2*y3-x3*y2
    B = (x1**2+y1**2)*(y3-y2)+(x2**2+y2**2)*(y1-y3)+(x3**2+y3**2)*(y2-y1)
    C = (x1**2+y1**2)*(x2-x3)+(x2**2+y2**2)*(x3-x1)+(x3**2+y3**2)*(x1-x2)
    D = (x1**2+y1**2)*(x3*y2-x2*y3)+(x2**2+y2**2)*(x1*y3-x3*y1)+(x3**2+y3**2)*(x2*y1-x1*y2)
    center = np.array([-C/2/A,-B/A/2])
    r = np.sqrt((B**2+C**2-4*A*D)/4/A**2)
#     print (A,B,C,D)
    return center, r