# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 10:23:21 2020

@author: Isadora Zampol
"""
import numpy as np

def update(existingCumulative, newVal):
    (count, mean, M2) = existingCumulative
    count += 1
    delta = newVal - mean
    mean += delta / count
    delta2 = newVal - mean
    M2 += delta * delta2
    return (count, mean, M2)


# Retrieve the mean, variance and sample variance from an aggregate
def finalise(existingCumulative):
    (count, mean, M2) = existingCumulative
    if count < 2:
        return float("nan")
    else:
        (mean, variance, sampleVariance) = (mean, M2 / count, M2 / (count - 1))
        return (mean, variance, sampleVariance)


def pick_axis (data, axis):
    new_data = []

    for i in range(np.shape(data.tolist())[axis]):
           new_data.append(np.take(data.tolist(),i,axis=axis))
    print (np.shape(new_data))

    return np.asarray(new_data)


def reshape (data):
        data = np.stack(data,axis=0)
        data = data.reshape(-1, 12*15, 145, 360)[0]
        return data


def variance(data, axis, choice=''):
    if axis != 0:
        data = pick_axis(data, axis)
    state= (1, np.array(data[0,:,:], dtype=float), np.zeros(data[0,:,:].shape))

    for i in range(np.shape(data)[0]):
        current = data[i,:,:]
        state = update(state, current)

    mean, var0, var1 = finalise(state)

    if (choice=='m'):
        return mean
    else:
        return var1


def get_slopes (data):
    slopes_mask = np.full((145,360), np.nan)
    slopes = np.zeros((145,360))
    months = np.arange(0,12*15)

    for lat in np.arange(0,145):
        for lon in np.arange(0,360):
            yfinite = np.isfinite(data[:, lat, lon])
            if (months[yfinite].size > 5):
                slopes [lat][lon], k = np.polyfit(months[yfinite], data[:, lat, lon][yfinite], 1)
                slopes_mask[lat][lon] = 1

    return slopes, slopes_mask, months
