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
# parameter for pyFAI integration
##########################################################
npt = 1024
unit = "q_A^-1"
poni_file_path = "/sdf/data/lcls/ds/xpp/xppl1001021/results/shared/poni/jungfrau_231103.poni"
mask_file_path = "/sdf/data/lcls/ds/xpp/xppl1001021/results/shared/mask/mask_jungfrau_calib_231104.npy"
result_h5_save_dir = "/sdf/data/lcls/ds/xpp/xppl1001021/results/shared/jungfrau_Iq/"
########################################################## 

hostname=socket.gethostname()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--run', 
                    help='run', 
                    type=str, 
                    default=os.environ.get('RUN_NUM', ''))

args = parser.parse_args()
run = args.run

logger.debug('Args to be used for Iq integration of run: {0}'.format(args))

mask = np.load(mask_file_path)
ai = pyFAI.load(poni_file_path)

ds = psana.MPIDataSource('exp=xppl1001021:run={}:smd'.format(run))
det = psana.Detector('jungfrau1M_alcove')
delay_stage_location = psana.Detector('sd_delay')
save_file_path = f'{result_h5_save_dir}run{run}_Iq.h5'
smldata = ds.small_data(save_file_path, gather_interval=100)

# ds.break_after(100) # stop iteration after 3 events (break statements do not work reliably with MPIDataSource).
common_q = None
t0 = time.time()
for nevt,evt in enumerate(ds.events()):
    printMsg(nevt, evt.run(), ds.rank, ds.size)
    img = det.image(evt)
    q, Iq = ai.integrate1d(img, npt, mask=mask, unit=unit)
    if common_q is None:
        common_q = q
    smldata.event(Iq = Iq)
    stage_value = delay_stage_location(evt)
    delay_ps = 0.939 * (stage_value - 6.96)
    smldata.event(delay_ps = delay_ps)

smldata.save(common_q = common_q)
print('rank %d on %s is finished'%(ds.rank, hostname))
smldata.save()

logger.debug(f'Iq of run {run} is saved : {save_file_path}')
