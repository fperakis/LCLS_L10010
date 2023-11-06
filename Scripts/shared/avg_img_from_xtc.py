#!/usr/bin/env python3

import argparse
import multiprocessing as mp

import numpy as np
import tqdm
import psana


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--run', 
                        help='run', 
                        type=str,)
    parser.add_argument('--experiment', 
                        help='experiment name', 
                        type=str,)
    parser.add_argument('--detector',
                        help='detector name',
                        type=str, 
                        default='jungfrau1M_alcove')
    parser.add_argument('--nimgs',
                        help='number of images to average',
                        type=int, 
                        default=100)

    args = parser.parse_args()

    ds = psana.MPIDataSource(f'exp={args.experiment}:run={args.run}:smd')
    detname = args.detector
    det = psana.Detector(detname)
    nimg = args.nimgs
    
    for nevt,evt in tqdm(enumerate(ds.events())):
        if nevt == nimg: break
        img = det.image(evt)
        if nevt == 0:
            imgs = np.zeros(np.r_[nimg, img.shape])
        imgs[nevt] = det.image(evt)
    np.save(f'/sdf/data/lcls/ds/xpp/{args.experiment}/results/shared/jungfrau_avg_img_run{int(args.run):04d}', imgs.mean(axis = 0))

    
