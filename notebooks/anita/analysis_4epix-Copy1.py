import numpy as np
import h5py as h5
import sys
import time

# Parse Arguement
run_num = int(sys.argv[1])


# Define parameters -- update!
exp_name = 'xppl1001021'
output_path = '/sdf/data/lcls/ds/xpp/{}/results/output/'.format(exp_name)
smalldata_path = '/sdf/data/lcls/ds/xpp/{}/hdf5/smalldata/'.format(exp_name)

# Define functions
def reconstruct_img(photons_i, photons_j, shape):
    nx, ny = shape
    phot_img, _,_ = np.histogram2d(photons_j+0.5, photons_i+0.5, bins=[np.arange(nx+1),np.arange(ny+1)])                         
    return phot_img

N_epix = 4
#user_mask = np.load('/sdf/data/lcls/ds/xpp/xpplx9221/results/kyounes/mask/standard_mask.npy')
#user_mask = user_mask.astype(bool)


# Load the result from droplet analysis
with h5.File(smalldata_path + '{}_Run{:04d}.h5'.format(exp_name, run_num),'r') as f:
    print(f.keys())
    photons_i = []
    photons_j = []
    
    total_mask = []#this is the general mask we use for photonization
    
    for epix in range(N_epix):
        print(epix+1)
        photons_j.append( f['epix_alc{}/ragged_droplet_photon_j'.format(epix+1)][:])
        photons_i.append( f['epix_alc{}/ragged_droplet_photon_i'.format(epix+1)][:])
        mask =      f['UserDataCfg/epix_alc{}/mask'.format(epix+1)][()]
        # Load all the masks
        mask = mask.astype(bool)
        bad_pixel_mask = np.load('/sdf/data/lcls/ds/xpp/xpplx9221/results/haoyuan/mask_epix{}_combined_hy_v1.npy'.format(epix+1))
        total_mask.append((mask  * bad_pixel_mask).astype(bool))

    i_sample = f['lombpm']['channels'][:,1]
    cc = np.array(f['ai/ch03'])
    vcc = np.array(f['ai/ch02'])
    


# Process each pattern in this run
shape = mask.shape
nframe = int(len(photons_i[0]))
#imgs_reconstruct = np.zeros(np.r_[nframe, shape])

pixel_num = float(np.sum(total_mask))

# Create holders for the result
kbar = np.zeros(nframe)
p0 = np.zeros(nframe)
p1 = np.zeros(nframe)
p2 = np.zeros(nframe)
p3 = np.zeros(nframe)

# Get the total photon count and probability per shot for all runs and patterns
tic = time.time()
for i in range(nframe):
    kbar[i] = 0
    p0[i] = 0
    p1[i] = 0
    p2[i] = 0
    p3[i] = 0

    
    for epix in range (N_epix):
        imgs_reconstruct = reconstruct_img(photons_i[epix][i], photons_j[epix][i], shape)
        kbar[i] += np.sum(imgs_reconstruct[total_mask[epix]])/ pixel_num
        p, p_bin_edge = np.histogram(imgs_reconstruct[total_mask[epix]].flatten(),bins=[-0.5, 0.5, 1.5, 2.5, 3.5])
    
        p0[i] += p[0] / pixel_num
        p1[i] += p[1] / pixel_num
        p2[i] += p[2] / pixel_num
        p3[i] += p[3] / pixel_num

    if i // 1000 == 0:
        toc = time.time()
        print(toc - tic)
        
# Get the analytical contrast expression

#beta_2ph = (2 * p2 - kbar * p1) / (kbar * (p1 - 2 * p2))




beta = (2 * p2 - kbar * p1) / (kbar * (p1 - 2 * p2))
#np.savez(output_path + 'contrast_run_{}_delay_{}_pulse_{}'.format(run_num, delay, pulse),## this would be ideal but not implemented
np.savez(output_path + 'test_contrast_run_{}'.format(run_num),

         beta=beta,
         p1=p1,
         p2=p2,
         p0=p0,
         p3=p3,
         kbar=kbar,
         cc=cc,
         vcc=vcc,
         i_sample=i_sample)