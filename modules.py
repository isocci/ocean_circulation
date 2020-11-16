# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 10:23:21 2020

@author: Isadora Zampol
"""
import numpy as np

N_YEARS = 15

#update the count and mean 
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

#reshape the data so the selected axis becomes the first
def pick_axis (data, axis):
    new_data = []

    for i in range(np.shape(data.tolist())[axis]):
           new_data.append(np.take(data.tolist(),i,axis=axis))
    print (np.shape(new_data))

    return np.asarray(new_data)

#reshape data into [months, lat, lon]
def reshape (data):
        data = np.stack(data,axis=0)
        data = data.reshape(-1, 12*N_YEARS, 145, 360)[0]
        return data


def variance(data, axis=0, choice=''):
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

#computes linear trend at each location that has all data points
def get_slopes (data):
    slopes_mask = np.full(data[0].shape, np.nan)
    slopes = np.zeros(data[0].shape)
    months = np.arange(0,12*N_YEARS)
    kov = np.zeros(data[0].shape)

    for lat in np.arange(0,data[0].shape[0]):
        for lon in np.arange(0,data[0].shape[1]):
            yfinite = np.isfinite(data[:, lat, lon])
            if (months[yfinite].size == 12*N_YEARS):
                a, b = np.polyfit(months[yfinite], data[:, lat, lon][yfinite], 1, full = False, cov = True)
                slopes[lat][lon]= a [0]
                kov [lat][lon] = np.sqrt(np.diag(np.asarray(b))) [0]
                slopes_mask[lat][lon] = 1
    return slopes, slopes_mask, months, kov

#keep only the locations with all data points available
def clean(data):
    new_data = np.full(data.shape, np.nan)
    months = np.arange(0,12*N_YEARS)
    for lat in np.arange(0,data[0].shape[0]):
        for lon in np.arange(0,data[0].shape[1]):
            yfinite = np.isfinite(data[:, lat, lon])
            if (months[yfinite].size == 12*N_YEARS):
                new_data[:, lat, lon] = data[:, lat, lon]
    return new_data

#create a set of pcs by projecting data onto an eof
def projected_pcs(solver, data):
        # Check that the shape of the data is compatible with the EOFs
        solver._verify_projection_shape(data, solver._originalshape)
        input_ndim = data.ndim
        eof_ndim = len(solver._originalshape) + 1
        # Create a slice object for truncating the EOFs
        slicer = slice(0, solver.neofs)
        # Weight the data set with the weighting used for EOFs
        data = data.copy()
        wts = solver.getWeights()
        if wts is not None:
            data = data * wts
        # Flatten the data to get [time, space] 2d data
        if eof_ndim > input_ndim:
            data = data.reshape((1,) + data.shape)
        records = data.shape[0]
        channels = np.product(data.shape[1:])
        data_flat = data.reshape([records, channels])
        # Locate the non-missing values and isolate them.
        if not solver._valid_nan(data_flat):
            raise ValueError('data and EOFs must have NaNs at the same locations')
        nonMissingIndex = np.where(np.logical_not(np.isnan(data_flat[0])))[0]
        data_flat = data_flat[:, nonMissingIndex]
        # Locate missing (NaNs) values in data and EOFs
        eofNonMissingIndex = np.where(
            np.logical_not(np.isnan(solver._flatE[0])))[0]
        if eofNonMissingIndex.shape != nonMissingIndex.shape or \
                (eofNonMissingIndex != nonMissingIndex).any():
            raise ValueError('data and EOFs must have NaNs at the same locations')
        eofs_flat = solver._flatE[slicer, eofNonMissingIndex]
        
        # Project the data set onto the EOFs 
        projected_pcs = np.dot(data_flat, eofs_flat.T)
        if eof_ndim > input_ndim:
            # If an extra dimension was created, remove it 
            projected_pcs = projected_pcs[0]
        return projected_pcs

#create a set of eofs by projecting data onto the pcs
def projected_eofs(solver, data):
        data_flat = data.reshape(-1, 180,145*360)
        # Project principal components onto the data to compute the EOFs
        projected_eofs = np.dot(solver._P[:], data_flat)
        # Reshape the pseudo EOFs
        projected_eofs = projected_eofs.reshape((solver._records,) + solver._originalshape)
        # Weighting EOFs
        if solver._weights is not None:
            projected_eofs = projected_eofs * solver._weights
        return projected_eofs