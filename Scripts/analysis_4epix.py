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

#user_mask = np.load('/sdf/data/lcls/ds/xpp/xpplx9221/results/kyounes/mask/standard_mask.npy')
#user_mask = user_mask.astype(bool)
mask=[[],[],[],[]]
N_epix=4

mask[0]=np.ones([704, 768])
mask[1]=np.ones([704, 768])
mask[2]=np.ones([704, 768])
mask[3]=np.ones([704, 768])

mask[0][0:354,370:390]=0
mask[0][0:154,680:710]=0

mask[1][350:,635]=0
mask[1][350:,636]=0
mask[1][350:,637]=0
mask[3][507:509,477:479]=0

pixel_out=8
pixel_center1=383
pixel_center2=352
deltapixel=3

for epix in range(N_epix):
    mask[epix][0:pixel_out,:]=0
    mask[epix][:,0:pixel_out]=0
    mask[epix][-pixel_out:,:]=0
    mask[epix][:,-pixel_out:]=0
    
    mask[epix][:,pixel_center1-deltapixel:pixel_center1+deltapixel]=0
    mask[epix][pixel_center2-deltapixel:pixel_center2+deltapixel,:]=0

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
        mask_standard =      f['UserDataCfg/epix_alc{}/mask'.format(epix+1)][()]
        # Load all the masks
        mask_standard = mask_standard.astype(bool)
        bad_pixel_mask = np.load('/sdf/data/lcls/ds/xpp/xpplx9221/results/haoyuan/mask_epix{}_combined_hy_v1.npy'.format(epix+1))
        total_mask.append((mask[epix]*mask_standard  * bad_pixel_mask).astype(bool))

    i_sample = f['lombpm']['channels'][:,1]
    cc = np.array(f['ai/ch03'])
    vcc = np.array(f['ai/ch02'])
    


# Process each pattern in this run
shape = mask_standard.shape
nframe = int(len(photons_i[0]))
#imgs_reconstruct = np.zeros(np.r_[nframe, shape])

pixel_num = float(np.sum(total_mask))

# Create holders for the result
kbar = np.zeros([N_epix, nframe])
beta = np.zeros([N_epix, nframe])
p0 = np.zeros([N_epix, nframe])
p1 = np.zeros([N_epix, nframe])
p2 = np.zeros([N_epix, nframe])
p3 = np.zeros([N_epix, nframe])

# Get the total photon count and probability per shot for all runs and patterns
tic = time.time()
pixel_num=[]
for epix in range (N_epix):
    pixel_num.append( float(np.sum(total_mask[epix])))
for i in range(nframe):
    for epix in range (N_epix):
        imgs_reconstruct = reconstruct_img(photons_i[epix][i], photons_j[epix][i], shape)
        kbar[epix,i] += np.sum(imgs_reconstruct[total_mask[epix]])/ pixel_num
        p, p_bin_edge = np.histogram(imgs_reconstruct[total_mask[epix]].flatten(),bins=[-0.5, 0.5, 1.5, 2.5, 3.5])
    
        p0[epix,i] = p[0] / pixel_num[epix]
        p1[epix,i] = p[1] / pixel_num[epix]
        p2[epix,i] = p[2] / pixel_num[epix]
        p3[epix,i] = p[3] / pixel_num[epix]

    if i // 10000 == 0:
        toc = time.time()
        print(i/nframe,toc - tic)
        
# Get the analytical contrast expression

#beta_2ph = (2 * p2 - kbar * p1) / (kbar * (p1 - 2 * p2))


delayStageLocation = f['/epicsAll/sd_delay'][:]
delay = np.mean(0.939 * (delayStageLocation - 6.96))


for epix in range(N_epix):
    beta[epix,:] = (2 * p2[epix,:] - kbar[epix,:] * p1[epix,:]) / (kbar[epix,:] * (p1[epix,:] - 2 * p2[epix,:]))
    
    
#np.savez(output_path + 'contrast_run_{}_delay_{}_pulse_{}'.format(run_num, delay, pulse),## this would be ideal but not implemented
np.savez(output_path + 'contrast_run_{}_delay_{}'.format(run_num, ),## this would be ideal but not implemented
         beta=beta,
         p1=p1,
         p2=p2,
         p0=p0,
         p3=p3,
         kbar=kbar,
         cc=cc,
         vcc=vcc,
         i_sample=i_sample)