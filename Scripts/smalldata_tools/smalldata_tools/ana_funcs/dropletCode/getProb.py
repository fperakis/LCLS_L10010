import numpy as np
import scipy.special as sp

def NB_dist(k,M,kavg,I0=1.):
    temp1 = sp.gammaln(k+M)-sp.gammaln(k+1)-sp.gammaln(M)
    temp2 = -k*np.log(1 + M/kavg)
    temp3 = -M*np.log(1 + kavg/M)
    return I0*np.exp(temp1+temp2+temp3)

def chisqs(p,kavg,M,nroi):
    N = np.size(kavg)
    k = np.reshape((np.arange((4))),(4,1))
    k = np.tile(k,(1,N))
    kavg = np.tile(kavg,(4,1))
    return -2*np.nansum((p*nroi*np.log(1/p*NB_dist(k,M,kavg,1.))))

def getContrast(ps, nroi,low, high, Mmax):
    ps = np.transpose(ps)
    kavg = ps[-1]
    kavg_filter = (kavg>=low)&(kavg<=high)
    kavg = kavg[kavg_filter]
    ps = ps[:,kavg_filter]
    
    nn = (Mmax-1)*1000+1
    Ms = np.linspace(1,Mmax,nn)
    chi2 = np.zeros(Ms.size)
    for ii,M0 in enumerate(Ms):
        chi2[ii] = chisqs(p = ps[:4], kavg = ps[-1],M = M0, nroi = nroi)
    pos = np.argmin(chi2)
    M0 = Ms[pos]
    #curvature as error analysis
    dM = Ms[1] - Ms[0]
    try:
        delta_M = np.sqrt(dM**2/(chi2[pos+1]+chi2[pos-1]-2*chi2[pos]))
    except:
        delta_M = 0.
    return M0, delta_M


def getProb_img(photonlist, mask, Np=12):
    nx,ny = mask.shape
    #nroi as the number of rois
    nroi = int(mask.max())
    p = np.zeros((1,Np,nroi))
    ave0, xedges, yedges = np.histogram2d(photonlist[:,1]+0.5, photonlist[:,2]+0.5, bins=[np.arange(nx+1),np.arange(ny+1)])
    for roi_num in range(1,nroi+1):
        npx = np.where(mask == roi_num)[0].size
        p[0,:,roi_num-1] = np.bincount(np.int32(ave0[mask==roi_num].ravel()),minlength = Np)[:Np]
        p[0,-1,roi_num-1] = ave0[mask == roi_num].sum()
        p[0,:,roi_num-1] /=float(npx)
    return p
    
