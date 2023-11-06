import numpy as np
from smalldata_tools.ana_funcs.dropletCode.convert_img import convert_img
from smalldata_tools.ana_funcs.dropletCode.loopdrops import *
from smalldata_tools.ana_funcs.dropletCode.getProb import *
import scipy.ndimage.measurements as measurements
import skimage.measure as measure
import scipy.ndimage.filters as filters
from scipy import sparse
from smalldata_tools.DetObject import DetObjectFunc

class droplet2Func(DetObjectFunc):
    '''
    return_img: whether or not to return the img with how many photons at each coordinate
    threshold: # (noise sigma for) threshold
    mask: pass a mask in here, is None: use mask stored in DetObject
    aduspphot: 
    offset: 
    
    uses convert_img to make droplets to analyze 
    uses loopdrops to find the photons in the droplets (don't forget to append the ones)
    
    counts the number of photons at each (rounded) coordinate
    returns either photonlist or img depending on return_img
    '''
    def __init__(self, **kwargs):
        self.return_img = kwargs.get('return_img',False)
        if self.return_img is False:
            self._name = kwargs.get('name', 'ragged_droplet')
        else:
            self._name = kwargs.get('name', 'droplet')
        super(droplet2Func, self).__init__(**kwargs)
        self.threshold = kwargs.get('threshold', None)
        self.droplet_mask = kwargs.get('droplet_mask', None)
        self.mask = kwargs.get('mask', None)
        self.aduspphot = kwargs.get('aduspphot', 0)
        self.offset = kwargs.get('offset', 0)
        self.photpts = np.arange(1000000)*self.aduspphot-self.aduspphot+self.offset
        if self.mask is None:
            print('A mask MUST be passed to droplet2Func.')
        self.nroi = int(self.mask.max())

    def setFromDet(self, det):
        super(droplet2Func, self).setFromDet(det)
        if self.droplet_mask is None:
            self.droplet_mask = det.cmask
        else:
            self.droplet_mask = np.logical_and(self.droplet_mask, det.cmask)
        
    def process(self, data):
        sum_img = None
        img = data* 17
        #make droplets
        ones,ts,pts,h,b = convert_img(img,self.threshold,self.photpts,self.droplet_mask)
        #find photons
        photonlist = loopdrops(ones,ts,pts,self.aduspphot,self.photpts)
        photonlist = np.append(ones[:,[0,2,1]], photonlist, axis=0) # indexes are inverted for ones because of c vs python indexing
        if sum_img is None:
            sum_img = img.copy()
            hh = h.copy()
        else:
            sum_img += img
            hh += h.copy()
            
        nx, ny = img.shape
        
        phot_img, xedges, yedges = np.histogram2d(photonlist[:,1]+0.5, photonlist[:,2]+0.5, bins=[np.arange(nx+1),np.arange(ny+1)])
        
        # look at this
        p = getProb_img(photonlist, self.mask, 12)
        # output dictionary
        d = {}
#         print (d['prob'])
        if self.return_img is False:  
            d['photon_i'] = photonlist[:,2]
            d['photon_j'] = photonlist[:,1]
            for i in range(self.nroi):
                d['prob_{}'.format(i)] = p[0,:,i]
        else:
            d['img'] = phot_img
        return d
     