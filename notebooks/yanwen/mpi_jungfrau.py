import psana
import sys
from utilities_mpi import printMsg
import numpy as np
import time
import resource
import socket

hostname=socket.gethostname()

run = 24
ds = psana.MPIDataSource('exp=xppl1001021:run={}:smd'.format(run))
det = psana.Detector('jungfrau1M_alcove')
smldata = ds.small_data('jungfrau/run{}.h5'.format(run),gather_interval=100)
 
#ds.break_after(100) # stop iteration after 3 events (break statements do not work reliably with MPIDataSource).
sum_img = None
t0 = time.time()
for nevt,evt in enumerate(ds.events()):
    printMsg(nevt, evt.run(), ds.rank, ds.size)
    img = det.image(evt)
    if sum_img is None:
        sum_img = img
    else:
        sum_img += img
    smldata.event(img = img)

smldata.save(sum_img = sum_img)
print('rank %d on %s is finished'%(ds.rank, hostname))
smldata.save()
