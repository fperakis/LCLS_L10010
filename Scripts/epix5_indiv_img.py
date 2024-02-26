import psana
from utilities_mpi import printMsg
import numpy as np
import time
import socket
import argparse
import os
import logging
import pyFAI

##########################################################
# parameter for save
##########################################################
result_h5_save_dir = "/sdf/data/lcls/ds/xpp/xppl1001021/results/shared/epix5_img/"
########################################################## 

hostname = socket.gethostname()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--run', 
                    help='run', 
                    type=str, 
                    default=os.environ.get('RUN_NUM', ''))

args = parser.parse_args()
run = args.run

logger.debug('Args to be used for epix5 individual image save : {0}'.format(args))

ds = psana.MPIDataSource('exp=xppl1001021:run={}:smd'.format(run))
det = psana.Detector('epix_alc5')
delay_stage_location = psana.Detector('sd_delay')
save_file_path = f'{result_h5_save_dir}run{run}_epix5_indiv_img.h5'
smldata = ds.small_data(save_file_path, gather_interval=100)

# load roi and mask
roi = np.load('/sdf/data/lcls/ds/xpp/xppl1001021/results/shared/roi_03.npy')
mask = np.load('/sdf/data/lcls/ds/xpp/xppl1001021/results/shared/mask/mask_epix5_231105.npy')
mask = ~(mask.T).astype('bool')
roi_with_mask = roi*mask
pixel_num = float(np.sum(roi_with_mask))

# define arrays to be saved
# kbar = np.zeros(len(ds.events())
# beta = np.zeros(len(ds.events())
# delays = np.zeros(len(ds.events())
kbar, beta, delays = [], [], []
t0 = time.time()

for nevt,evt in enumerate(ds.events()):
    
    if nevt % 500 == 0:
        print(f"done until {nevt}-th img")
        
    printMsg(nevt, evt.run(), ds.rank, ds.size)
    img = det.image(evt)
    
    kbar = np.sum(img[roi_with_mask]) / pixel_num
    # beta = np.var(img[roi_with_mask].flatten()) / (kbar[nevt])**2
    smldata.event(kbar = kbar)     
    smldata.event(beta = np.var(img[roi_with_mask].flatten()) / (kbar)**2)     
    
    stage_value = delay_stage_location(evt)
    # delays[nevt] = 0.939 * (stage_value - 6.96)
    delay_ps = 0.939 * (stage_value - 6.96)
    smldata.event(delay_ps = delay_ps)
                  
# np.savez(result_h5_save_dir + f'run{run}_epix5',
#          beta=beta,
#          kbar=kbar, 
#          delays=delays)
                  
# ds.break_after(100) # stop iteration after 100 events (break statements do not work reliably with MPIDataSource).
# t0 = time.time()
# for nevt,evt in enumerate(ds.events()):
#     if nevt % 500 == 0:
#         print(f"done until {nevt}-th img")
#     printMsg(nevt, evt.run(), ds.rank, ds.size)
#     img = det.image(evt)
#     smldata.event(img = img)
#     stage_value = delay_stage_location(evt)
#     delay_ps = 0.939 * (stage_value - 6.96)
#     smldata.event(delay_ps = delay_ps)

print('rank %d on %s is finished'%(ds.rank, hostname))
smldata.save()

logger.debug(f'epix5 indiv image of run {run} is saved : {save_file_path}')
