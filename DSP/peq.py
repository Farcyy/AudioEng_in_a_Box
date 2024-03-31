import numpy as np
import tensorflow as tf 
import scipy
from numba import jit 

#Biquad Implementation
#@jit(nopython=True)
def biqaud(
    gain_dB,
    cutoff_freq,
    q_factor,
    sample_rate,
    filter_type,
):

    
    """Use design parameters to generate coeffieicnets for a specific filter type.

    Args:
        gain_dB (float): Shelving filter gain in dB.
        cutoff_freq (float): Cutoff frequency in Hz.
        q_factor (float): Q factor.
        sample_rate (float): Sample rate in Hz.
        filter_type (str): Filter type.
            One of "low_shelf", "high_shelf", or "peaking"

    Returns:
        b (np.ndarray): Numerator filter coefficients stored as [b0, b1, b2]
        a (np.ndarray): Denominator filter coefficients stored as [a0, a1, a2]
    """

    dtype=np.float32
    A = 10 ** (gain_dB / 40.0)
    w0 = 2.0 * np.pi * (cutoff_freq / sample_rate)
    alpha = np.sin(w0) / (2.0 * q_factor)

    cos_w0 = np.cos(w0)
    sqrt_A = np.sqrt(A)

    if filter_type == 0: #high-shelf
        b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * sqrt_A * alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
        b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * sqrt_A * alpha)
        a0 = (A + 1) - (A - 1) * cos_w0 + 2 * sqrt_A * alpha
        a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
        a2 = (A + 1) - (A - 1) * cos_w0 - 2 * sqrt_A * alpha
    elif filter_type == 1: #low-shelf
        b0 = A * ((A + 1) - (A - 1) * cos_w0 + 2 * sqrt_A * alpha)
        b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
        b2 = A * ((A + 1) - (A - 1) * cos_w0 - 2 * sqrt_A * alpha)
        a0 = (A + 1) + (A - 1) * cos_w0 + 2 * sqrt_A * alpha
        a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
        a2 = (A + 1) + (A - 1) * cos_w0 - 2 * sqrt_A * alpha
    elif filter_type == 2: #peaking
        b0 = 1 + alpha * A
        b1 = -2 * cos_w0
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * cos_w0
        a2 = 1 - alpha / A
    #else:
    #    pass
    #    raise ValueError(f"Invalid filter_type: {filter_type}.")

    b = np.array([b0, b1, b2]) / a0
    a = np.array([a0, a1, a2]) / a0

    return b, a


#PEQ Implementation
# Adapted from https://github.com/csteinmetz1/pyloudnorm/blob/master/pyloudnorm/iirfilter.py
#@jit(nopython=True)
def parametric_eq(x, sample_rate, params):
    #print("###################")
    #print("error handling")
    #print("###################")

    low_shelf_gain_dB = params[0]
    low_shelf_cutoff_freq = params[1]
    low_shelf_q_factor = params[2]

    first_band_gain_dB = params[3]
    first_band_cutoff_freq = params[4]
    first_band_q_factor = params[5]

    second_band_gain_dB = params[6]
    second_band_cutoff_freq = params[7]
    second_band_q_factor = params[8]

    third_band_gain_dB = params[9]
    third_band_cutoff_freq = params[10]
    third_band_q_factor = params[11]

    fourth_band_gain_dB = params[12]
    fourth_band_cutoff_freq = params[13]
    fourth_band_q_factor = params[14]

    high_shelf_gain_dB = params[15]
    high_shelf_cutoff_freq = params[16]
    high_shelf_q_factor = params[17]


    dtype=np.float32


    # -------- apply low-shelf filter --------
    b, a = biqaud(
        low_shelf_gain_dB,
        low_shelf_cutoff_freq,
        low_shelf_q_factor,
        sample_rate,
        1,
    )

    #x = biquad_filter(b, a, x)
    sos0 = np.concatenate((b, a))
    sos0 = sos0.reshape((1, 6))
    x = scipy.signal.sosfilt(sos0, x)
    #x = scipy.signal.lfilter(b, a, x)

    # -------- apply first-band peaking filter --------
    b, a = biqaud(
        first_band_gain_dB,
        first_band_cutoff_freq,
        first_band_q_factor,
        sample_rate,
        2,
    )

    #x = biquad_filter(b, a, x)
    sos1 = np.concatenate((b, a))
    sos1 = sos1.reshape((1, 6))
    x = scipy.signal.sosfilt(sos1, x)
    #x = scipy.signal.lfilter(b, a, x)

    # -------- apply second-band peaking filter --------
    b, a = biqaud(
        second_band_gain_dB,
        second_band_cutoff_freq,
        second_band_q_factor,
        sample_rate,
        2,
    )

    #x = biquad_filter(b, a, x)
    sos2 = np.concatenate((b, a))
    sos2 = sos2.reshape((1, 6))
    x = scipy.signal.sosfilt(sos2, x)
    #x = scipy.signal.lfilter(b, a, x)

    # -------- apply third-band peaking filter --------
    b, a = biqaud(
        third_band_gain_dB,
        third_band_cutoff_freq,
        third_band_q_factor,
        sample_rate,
        2,
    )

    #x = biquad_filter(b, a, x)
    sos3 = np.concatenate((b, a))
    sos3 = sos3.reshape((1, 6))
    x = scipy.signal.sosfilt(sos3, x)
    #x = scipy.signal.lfilter(b, a, x)

    # -------- apply fourth-band peaking filter --------
    b, a = biqaud(
        fourth_band_gain_dB,
        fourth_band_cutoff_freq,
        fourth_band_q_factor,
        sample_rate,
        2,
    )

    #x = biquad_filter(b, a, x)
    sos4 = np.concatenate((b, a))
    sos4 = sos4.reshape((1, 6))
    x = scipy.signal.sosfilt(sos4, x)
    #x = scipy.signal.lfilter(b, a, x)

    # -------- apply high-shelf filter --------
    b, a = biqaud(
        high_shelf_gain_dB,
        high_shelf_cutoff_freq,
        high_shelf_q_factor,
        sample_rate,
        0,
    )

    #x = biquad_filter(b, a, x)
    sos5 = np.concatenate((b, a))
    sos5 = sos5.reshape((1, 6))
    x = scipy.signal.sosfilt(sos5, x)
    #x = scipy.signal.lfilter(b, a, x)

    x = x.astype(dtype)
    
    return x
