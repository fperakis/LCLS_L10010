import numpy as np
import h5py as h5
import sys
import time

# Parse Arguement
run_num = int(sys.argv[1])
epix = 5

# Define parameters -- update!
exp_name = 'xppl1001021'
output_path = '/sdf/data/lcls/ds/xpp/{}/results/output/'.format(exp_name)
smalldata_path = '/sdf/data/lcls/ds/xpp/{}/hdf5/smalldata/'.format(exp_name)

# Define functions
def reconstruct_img(photons_i, photons_j, shape):
    nx, ny = shape
    phot_img, _,_ = np.histogram2d(photons_j+0.5, photons_i+0.5, bins=[np.arange(nx+1),np.arange(ny+1)])                         
    return phot_img


    
# Load the result from droplet analysis
with h5.File(smalldata_path + '{}_Run{:04d}.h5'.format(exp_name, run_num),'r') as f:
    mask = f['UserDataCfg/epix_alc{}/mask'.format(epix)][()]#this is the general mask we use for photonization    
    photons_j = f['epix_alc{}/ragged_droplet_photon_j'.format(epix)][()]
    photons_i = f['epix_alc{}/ragged_droplet_photon_i'.format(epix)][()]
    i_sample = f['lombpm']['channels'][:,1]
    cc = np.array(f['ai/ch03'])
    vcc = np.array(f['ai/ch02'])
    
# Load all the masks
mask = mask.astype(bool)

user_mask = np.load("/sdf/data/lcls/ds/xpp/xppl1001021/results/shared/mask_epix5.npy")
user_mask = user_mask.astype(bool)
#bad_pixel_mask = np.load('/sdf/data/lcls/ds/xpp/xpplx9221/results/haoyuan/mask_epix{}_combined_hy_v1.npy'.format(epix))

total_mask = (mask * user_mask ).astype(bool)

# Process each pattern in this run
shape = mask.shape
nframe = int(len(photons_i))
#imgs_reconstruct = np.zeros(np.r_[nframe, shape])
    


roi=np.load('/sdf/data/lcls/ds/xpp/xppl1001021/results/shared/roi.npy')    
    
roi_with_mask = roi * total_mask
pixel_num = float(np.sum(roi_with_mask))

# Create holders for the result
kbar = np.zeros(nframe)
beta = np.zeros(nframe)

# Get the total photon count and probability per shot for all runs and patterns
tic = time.time()
for i in range(nframe):
    imgs_reconstruct = reconstruct_img(photons_i[i], photons_j[i], shape)
    kbar[i] = np.sum(imgs_reconstruct[roi_with_mask])/ pixel_num
    beta[i] = np.var(imgs_reconstruct[roi_with_mask].flatten())/np.sqrt(kbar[i])


    if i // 1000 == 0:
        toc = time.time()
        print(toc - tic)
        
# Get the analytical contrast expression

np.savez(output_path + 'contrast_run_{}_epix_{}'.format(run_num, epix),
         beta=beta,
         kbar=kbar,
         cc=cc,
         vcc=vcc,
         i_sample=i_sample)