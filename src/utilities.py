from pathlib import Path
from typing import Union
import sys
import re

import numpy as np
import matplotlib.pyplot as plt
import h5py
import xarray as xr

# constants
PLANCK_CONSTANT = 4.135667516*1e-18 # kev*sec
SPEED_OF_LIGHT = 3*1e8 # m/s

# setup
SDD = 4.3 # sample-detector distance in meters
ENERGY = 9.5 # energy of the x-rays in keV
WAVELENGTH = PLANCK_CONSTANT*SPEED_OF_LIGHT/ENERGY*1e10 # wavelength of the x-rays (A-1)
BEAMSTOP_RADIUS = 1.5e-3 # meter

# epix params
EPIX_PIXEL_SIZE = 50e-6 # meter
EPIX_NPIXEL = (704, 768)
EPIX_CENTER = (int(EPIX_NPIXEL[0]/2), int(EPIX_NPIXEL[0]/2))

# smalldata path
EXP = 'xppl1001021'
EXP_PATH = Path('/sdf/data/lcls/ds/{}/{}'.format(EXP[:3], EXP))
SMALLDATA = Path(EXP_PATH, 'hdf5', 'smalldata')
JUNGFRAU_DATA = Path(EXP_PATH, 'hdf5', 'smalldata', 'jungfrau')
EPIX2_DATA = Path(EXP_PATH, 'hdf5', 'smalldata', 'epix2_only')
RESULTS = Path(EXP_PATH, 'results')
SHARED = Path(RESULTS, 'shared')
EPIX5_DATA = Path(SHARED, 'epix5_img')
IQ_PATHS = Path(SHARED, "jungfrau_Iq")
print('Smalldata path exists:', JUNGFRAU_DATA.exists())
print('Jungfrau path exists:', JUNGFRAU_DATA.exists())
print('epix2 path exists:', EPIX2_DATA.exists())
print('iq path exists:', IQ_PATHS.exists())


def saveEpixImgs(images, run, output_path):
    imgs_reconstructed = np.zeros([4, 704, 768])
    for epix in range(4):
        output_run_path = Path(SHARED, 'output', 'images', f'run{run}').resolve()
        if not output_run_path.exists():
            output_run_path.mkdir()
        np.save(Path(output_run_path, f'epix_{epix+1}_img_run{run}.npy'), imgs_reconstructed[epix])


def wavelength(energy):
    return PLANCK_CONSTANT / energy


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


class L10010H5Loader:
    """
    A class for loading and accessing data from L10010 beamtime files. The reason for implementing
    this class is to provide a more convenient way to access the data that is split between different h5 files,
    specifically the smalldata and integrated h5 files.

    Args:
        run (int): The run number.

    Attributes:
        - run (int): The run number.
        - file_path (Path): The path to the h5 file.
        - integrated_file_path (Path): The path to the integrated h5 file.
        - _integrated_runs (list): List of integrated runs.
        - _integrated (bool): Flag indicating if the run is integrated.
        - _lomdiode (None): Placeholder for lomdiode attribute.
        - _cc (None): Placeholder for cc attribute.
        - _vcc (None): Placeholder for vcc attribute.
        - _jg_pyfai_azav (None): Placeholder for jg_pyfai_azav attribute.
        - _jg_pyfai_q (None): Placeholder for jg_pyfai_q attribute.
        - _delay_stage_location (None): Placeholder for delay_stage_location attribute.
        - _total_mask (list): List of total masks.
        - _epix_masks (list): List of epix masks.
        - _epix_photons_i (list): List of epix photons i.
        - _epix_photons_j (list): List of epix photons j.
        - _epix_images (list): List of epix images.

    Methods:
        - __enter__(): Enter method for context manager.
        - __exit__(): Exit method for context manager.
        - __getitem__(key: str) -> np.ndarray | list: Get item method for accessing data.

    """

    def __init__(self, run):
        self.run = run
        self.file_path = Path(JUNGFRAU_DATA, f'{EXP}_Run{run:04d}.h5')
        self._integrated_runs = []
        for file in IQ_PATHS.glob("run*"):
            self._integrated_runs.append(re.findall(r'\d+', file.as_posix())[1])
        
        if self._integrated_runs.__contains__(str(run)):
            self._integrated = True
            self.integrated_file_path = Path(IQ_PATHS, f"run{run}_Iq.h5")
        else:
            self._integrated = False
        
        # h5 attrs initialization
        self._lomdiode = None
        self._cc = None
        self._vcc = None
        self._jg_pyfai_azav = None
        self._jg_pyfai_q = None
        self._delay_stage_location = None
        self._total_mask = []
        self._epix_masks = []
        self._epix_photons_i = []
        self._epix_photons_j = []
        self._epix_images = []

    def __enter__(self):
        self.file = h5py.File(self.file_path, 'r')
        self.integrated_file = h5py.File(self.integrated_file_path, 'r')
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.file.close()
        self.integrated_file.close()
    
    def __getitem__(self, key: str) -> Union[np.ndarray, list]:
        """
        Get item method for accessing data. Will return data from the correct h5 file depending on the key.

        Args:
            - key (str): The key to access the data.

        Returns:
            - np.ndarray | list: The requested data.

        """
        if key == "jg_pyfai_azav" or key == "Iq":
            return self.integrated_file["Iq"][()]
        elif key == "jg_pyfai_q" or key == "q":
            return self.integrated_file["common_q"][()]
        elif key == "delay":
            return self.integrated_file['/delay_ps'][:]
        elif key == "total_mask":
            for i in range(1, 5):
                _mask = self.file[f'/UserDataCfg/epix_alc{i}/mask'][:].astype(bool)
                _bad_pixel_mask = np.load(Path(SHARED, 'mask', f'mask_epix{i}_combined_hy_v1.npy'))
                self._total_mask.append(_mask * _bad_pixel_mask)
            return self.total_mask
        elif key == "epix_images":
            self._epix_images.append(self.file[f'/Sums/epix_alc{i}_calib_dropped_square'][:])
            return self._epix_images
        elif key == "epix_photons":
            for i in range(1,5):
                _photons_i = self.file[f'/epix_alc2/ragged_droplet_photon_i'][:]
                _photons_j = self.file[f'/epix_alc2/ragged_droplet_photon_j'][:]
                self._epix_photons_i.append(_photons_i)
                self._epix_photons_j.append(_photons_j)
            return self._epix_photons_i, self._epix_photons_j
        elif key == "lomdiode":
            return self.file['/lombpm/channels'][:,1]
        elif key == "cc" or key == "vcc":
            self._diodes = self.file['/diode2/channels'][()]
            self._diodes = np.append(self._diodes, self.file['/diodeU/channels'][()], axis = 1)
            if key == "cc":
                return self._diodes[:, 0]
            else:
                return self._diodes[:, 4]
        elif key == "delay_stage_location":
            return self.file['/epicsAll/sd_delay'][:]
    
    def print_attrs(self) -> None:
        with open("h5_attrs.txt", "w") as sys.stdout:
            print("SMD h5 attrs:\n")
            self.file.visititems(visit_func)
            print("\nIntegrated h5 attrs:\n")
            self.integrated_file.visititems(visit_func)
    
def plotChannelsIq(
    iq_data: xr.DataArray, norm:bool=True, title:str=None, save_path:Path=None
) -> None:
    from matplotlib.colors import LogNorm
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(10,10), layout='tight')
    _,_,_, h1 = ax1.hist2d(iq_data.cc, iq_data.vcc, bins=50, norm=LogNorm())
    fig.colorbar(h1)
    ax1.set_xlabel('cc', fontsize=16)
    ax1.set_ylabel('vcc', fontsize=16)
    
    iq_mean = iq_data.mean("delay")
    if norm:
        iq_mean_plot = iq_mean / iq_mean.sel(q=slice(3, 4.5)).sum("q")
        iq_mean_plot /= iq_mean_plot.max()
    else:
        iq_mean_plot = iq_mean
    ax2.plot(
        iq_data.q, 
        iq_mean_plot,
        label=r"$\langle I(q) \rangle_{delay}$", 
        color="k",
        linewidth=2,
        zorder=3
    )
    ax2.plot(
        iq_data.q, 
        iq_mean_plot,
        color="w",
        linewidth=2.5,
        zorder=2,
    )
    for counter, i in enumerate(iq_data[::100]):
        if norm:
            iq_individual_plot = i / i.sel(q=slice(3, 4.5)).sum("q")
            iq_individual_plot /= iq_individual_plot.max()
        else:
            iq_individual_plot = i
        ax2.plot(
            iq_data.q, 
            iq_individual_plot,
            label=r"$I(q)$ individual delay" if counter==0 else None,
            color="g",
            linewidth=2,
            alpha=0.2,
            zorder=1
        )
    ax2.legend()
    ax2.set_xlim(iq_data.q.min(), iq_data.q.max())
    ax2.set_xlabel("q (1/$\AA$)", fontsize=16)
    ax2.set_ylabel("I(q) (a.u.)", fontsize=16)
    
    if norm:
        iq_data = iq_data / iq_data.sel(q=slice(3, 4.5)).sum("q")
        iq_data /= iq_data.max()
    _,_,_, h2 = ax3.hist2d(iq_data.cc.values, iq_data.mean("q"), bins=50, cmap="PuBuGn_r", norm=LogNorm())
    fig.colorbar(h2)
    ax3.set_xlabel("cc", fontsize=16)
    ax3.set_ylabel(r"$\langle I(q) \rangle_{q}$", fontsize=16)
    _,_,_, h3 = ax4.hist2d(iq_data.vcc.values, iq_data.mean("q"), bins=50, cmap="YlOrRd_r", norm=LogNorm())
    fig.colorbar(h3)
    ax4.set_xlabel("vcc", fontsize=16)
    ax4.set_ylabel(r"$\langle I(q) \rangle_{q}$", fontsize=16)
    
    if not title:
        if norm:
            fig.suptitle(f"run {iq_data.run} normalized", fontsize=18)
        else:
            fig.suptitle(f"run {iq_data.run}", fontsize=18)
    else:
        fig.suptitle(title, fontsize=18)
    
    if save_path:
        plt.savefig(save_path, dpi=300)