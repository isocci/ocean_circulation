#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 11:54:53 2020

@author: isa

This function returns the following masks:
Caribbean, Mediterranean, SE_Asia, Arctic, N_Pacific, 
Eq_Pacific, S_Pacific, N_Atlantic, Eq_Atlantic, S_Atlantic, 
Arabian, Bay_Bengal, Eq_Indian, S_Indian, Southern
"""
import numpy as np
import matplotlib.pyplot as plt
import regionmask
from regionmask.core.utils import create_lon_lat_dataarray_from_bounds as make_lon_lat
regions = [8,19,38,46,47,48,49,50,51,52,53,54,55,56,57]
lon = np.arange(-180, 180)
lat = np.arange(72.5, -72.5, -1)

masks = []

def get_masks():
    for region in regions:
        a = regionmask.defined_regions.ar6.ocean[[region]].mask(lon,lat,wrap_lon=False).to_masked_array().mask.astype(int)
        c=abs(a-1).astype('float')
        c[c == 0] = np.nan
    
        #fixing the alignment of masks and our data
        c=np.flipud(c)
        c = np.roll(c, 160, axis = 1)
        c = np.roll(c, -8, axis = 0)
    
        masks.append(c)
    return masks
'''text_kws = dict(color="#67000d", fontsize=7, bbox=dict(pad=0.2, color="w"))
regionmask.defined_regions.ar6.ocean.plot(text_kws=text_kws, add_land=True)
plt.tight_layout()'''

