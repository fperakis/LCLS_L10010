#Original from #/XF11ID/analysis/Analysis_Pipelines/Develop/chxanalys/chxanalys/chx_correlation.py
# ######################################################################
# change from mask's to indices
# unravel fftconvolve to perform fewer ffts
# move ffts of mask to __init__.
########################################################################

"""
This module is for functions specific to cross correlations
"""
import numpy as np
from scipy.fftpack.helper import next_fast_len

# for a convenient status bar, default is off
try:
    #import tqdm #py2.7
    from tqdm import tqdm #py3.5
except ImportError:
    def tqdm(iterator):
        return(iterator)

class crosscor:
    '''
        Compute a 1D or 2D cross-correlation on data.
        This uses a mask, which may be binary (array of 0's and 1's),
        or a list of non-negative integer id's to compute cross-correlations
        separately on.
        The symmetric averaging scheme introduced here is inspired by a paper
        from Schatzel, although the implementation is novel in that it
        allows for the usage of arbitrary masks. [1]_
        Examples
        --------
        >> ccorr = crosscor(mask.shape, mask=mask)
        >> # correlated image
        >> cimg = cc(img1)
        or, mask may may be ids
        >> cc = crosscor(ids)(
        #(where ids is same shape as img1)
        >> cc1 = cc(img1)
        >> cc12 = cc(img1, img2)
        # if img2 shifts right of img1, point of maximum correlation is
        # shifted right from correlation center (cc.centers[i])
        References
        ----------
        .. [1] Schatzel, Klaus, Martin Drewel, and Sven Stimac. "Photon
               correlation measurements at large lag times: improving
               statistical accuracy." Journal of Modern Optics 35.4 (1988):
               711-718.
    '''
    # TODO : when mask is None, don't compute a mask, submasks
    def __init__(self, shape, mask=None, normalization=None):
        '''
            Prepare the spatial correlator for various regions specified by the
            id's in the image.
            Parameters
            ----------
            shape : 1 or 2-tuple
                The shape of the incoming images or curves. May specify 1D or
                2D shapes by inputting a 1 or 2-tuple
            mask : 1D or 2D np.ndarray of int, optional
                Each non-zero integer represents unique bin. Zero integers are
                assumed to be ignored regions. If None, creates a mask with
                all points set to 1
            normalization: string or list of strings, optional
                These specify the normalization and may be any of the
                following:
                    'regular' : divide by pixel number
                    'symavg' : use symmetric averaging
                Defaults to ['regular'] normalization
            Delete argument wrap as not used. See fftconvolve as this
            expands arrays to get complete convolution, IE no need
            to expand images of subregions.
        '''
        if normalization is None: normalization = ['regular']
        elif not isinstance(normalization, list): normalization = list([normalization])
        self.normalization = normalization

        if mask is None: #we can do this easily now.
            mask = np.ones(shape)

        # initialize subregion information for the correlations
        #first find indices of subregions and sort them by subregion id
        pii,pjj = np.where(mask)
        bind=mask[pii,pjj]
        ord=np.argsort(bind)
        bind=bind[ord];pii=pii[ord];pjj=pjj[ord] #sort them all

        #make array of pointers into position arrays
        pos=np.append(0,1+np.where(np.not_equal(bind[1:], bind[:-1]))[0])
        pos=np.append(pos,len(bind))
        #pos is a pointer such that (pos[i]:pos[i+1])
        # are the indices in the position arrays of subregion i.
        self.pos=pos
        self.ids = bind[pos[:-1]]
        self.nids = len(self.ids)
        sizes=np.array([[pii[pos[i]:pos[i+1]].min(),pii[pos[i]:pos[i+1]].max(),\
                         pjj[pos[i]:pos[i+1]].min(),pjj[pos[i]:pos[i+1]].max()]\
                           for i in range(self.nids)])
        self.pii=pii; self.pjj=pjj
        self.offsets = sizes[:,0:3:2].copy()

        self.sizes = 1+(np.diff(sizes)[:,[0,2]]).copy()  #make sizes be for regions
        centers = np.array(self.sizes.copy())//2
        self.centers=centers.copy()
        fshapes=centers #reuse centers array
        if len(self.ids) == 1:
            self.centers = self.centers[0,:]
        # loop through and precalculate submask info
        self.mma1s=list()  #fft of submask
        self.maskcors=list() #autocor of submask
        for reg in range(self.nids):
            i = self.pii[pos[reg]:pos[reg+1]]-self.offsets[reg,0]
            j = self.pjj[pos[reg]:pos[reg+1]]-self.offsets[reg,1]
            #(i,j) is pixels in subregion
            # set up size for fft with padding
            shape = 2*self.sizes[reg,:] - 1
            fshape = [next_fast_len(int(d)) for d in shape]
            fshapes[reg]=fshape

            submask = np.zeros(self.sizes[reg,:])
            submask[i,j]=1
            mma1=np.fft.rfftn(submask, fshape) #for mask
            self.mma1s.append(mma1)
                #do correlation by ffts
            maskcor= np.fft.irfftn(mma1 * mma1.conj(), fshape)
            maskcor = _centered(maskcor, self.sizes[reg,:]) #make smaller??
            # choose some small value to threshold
            maskcor *= maskcor > .5
            self.maskcors.append(maskcor)
        self.fshapes=fshapes.copy()

    def __call__(self, img1, img2=None, normalization=None,prflag=False):
        ''' Run the cross correlation on an image/curve or against two
                images/curves
            Parameters
            ----------
            img1 : 1D or 2D np.ndarray
                The image (or curve) to run the cross correlation on
            img2 : 1D or 2D np.ndarray
                If not set to None, run cross correlation of this image (or
                curve) against img1. Default is None.
            normalization : string or list of strings
                normalization types. If not set, use internally saved
                normalization parameters
            prflag : if True, run tqdm on loop
            Returns
            -------
            ccorrs : 1d or 2d np.ndarray
                A list of images for the correlations. The zero correlation is
                located at shape//2 where shape is the 1 or 2-tuple
                shape of the array
        '''
        if normalization is None:
            normalization = self.normalization

        if img2 is None:
            self_correlation = True
        else:
            self_correlation = False

        ccorrs = list()

        pos=self.pos
        #loop over individual regions
        if(not prflag):
            def tqdm(iterator):
                return(iterator)
        for reg in range(self.nids):
            ii = self.pii[pos[reg]:pos[reg+1]]
            jj = self.pjj[pos[reg]:pos[reg+1]]
            i = ii-self.offsets[reg,0]
            j = jj-self.offsets[reg,1]
            #WE now have two sets of positions of the subregions
            #(i,j) in subregion and (ii,jj) in images.
            # set up size for fft with padding
            #shape = 2*self.sizes[reg,:] - 1
            fshape = self.fshapes[reg,:]

            mma1= self.mma1s[reg]
            maskcor= self.maskcors[reg]

            tmpimg=np.zeros(self.sizes[reg,:])
            tmpimg[i,j]=img1[ii,jj]
            im1=np.fft.rfftn(tmpimg, fshape) #image 1
            if self_correlation:
                ccorr = np.fft.irfftn(im1 * im1.conj(),fshape)#[fslice])
                ccorr = _centered(ccorr, self.sizes[reg,:])
            else:
                ndim=img1.ndim
                tmpimg2=np.zeros_like(tmpimg)
                tmpimg2[i,j]=img2[ii,jj]
                im2=np.fft.rfftn(tmpimg2, fshape) #image 2
                ccorr = np.fft.irfftn(im1 *im2.conj(),fshape)#[fslice])
                ccorr = _centered(ccorr, self.sizes[reg,:])
            # now handle the normalizations
            if 'symavg' in normalization:

                mim1=np.fft.rfftn(tmpimg, fshape)
                Icorr = np.fft.irfftn(mim1 * mma1.conj(),fshape)#[fslice])
                Icorr = _centered(Icorr, self.sizes[reg,:])
                # do symmetric averaging
                if self_correlation:
                    #reverse =[slice(None,None,-1) for i in range(Icorr.ndim)]
                    #Icorr2 =Icorr[reverse] #use symmetry of correlation
                    Icorr2 = np.fft.irfftn(mma1 * mim1.conj(),fshape)#[fslice])
                    Icorr2 = _centered(Icorr2, self.sizes[reg,:])
                else:
                    mim2=np.fft.rfftn(tmpimg2, fshape)
                    Icorr2 = np.fft.irfftn(mma1 * mim2.conj(), fshape)
                    Icorr2 = _centered(Icorr2, self.sizes[reg,:])
                # there is an extra condition that Icorr*Icorr2 != 0
                w = np.where(np.abs(Icorr*Icorr2) > 0)
                ccorr[w] *= maskcor[w]/Icorr[w]/Icorr2[w]

            if 'regular' in normalization:
                # only run on overlapping regions for correlation
                w = np.where(maskcor > .5)

                if self_correlation:
                    ccorr[w] /= maskcor[w] * np.average(tmpimg[w])**2
                else:
                    ccorr[w] /= maskcor[w] * np.average(tmpimg[w])*np.average(tmpimg2[w])
            ccorrs.append(ccorr)

        if len(ccorrs) == 1:
            ccorrs = ccorrs[0]

        return ccorrs

def _centered(img,sz):
    n=sz//2
    img=np.take(img,np.arange(-n[0],sz[0]-n[0]),0,mode="wrap")
    img=np.take(img,np.arange(-n[1],sz[1]-n[1]),1,mode="wrap")
    return img

