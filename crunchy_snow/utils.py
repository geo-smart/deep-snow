import numpy as np
import torch

def calc_dowy(doy):
    'calculate day of water year from day of year'
    if doy < 274:
        dowy = doy + (365-274)
    elif doy >= 274:
        dowy = doy-274
    return dowy

def calc_norm(tensor, minmax_list):
    '''
    normalize a tensor between 0 and 1 using a min and max value stored in a list
    '''
    normalized = (tensor-minmax_list[0])/(minmax_list[1]-minmax_list[0])
    normalized = torch.nan_to_num(normalized, 0)
    return normalized

def undo_norm(tensor, minmax_list):
    '''
    undo tensor normalization
    '''
    original = (tensor*(minmax_list[1]-minmax_list[0]))+minmax_list[0]
    return original

def db_scale(x, epsilon=1e-10):
    # Add epsilon only where x is zero
    x_with_epsilon = np.where(x==0, epsilon, x)
    # Calculate the logarithm
    log_x = 10 * np.log10(x_with_epsilon)
    # Set the areas where x was originally zero back to zero
    log_x[x==0] = 0
    return log_x


