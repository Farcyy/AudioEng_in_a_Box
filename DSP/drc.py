import numpy as np
import tensorflow as tf 
import scipy
from numba import jit 

@jit(nopython=True)
def drc(
    x,
    sample_rate,
    params, #threshold, ratio, attack_time, release_time, knee_dB, makeup_gain_dB,
):
    
    """
    Dynamic Range Compression (DRC) Function

    This function applies dynamic range compression to an input audio signal `x`.
    Compression is a form of automated gain control, reducing the dynamic range of the
    audio signal. The compression characteristics are defined by the parameters provided.

    Parameters:
    - x (np.ndarray): The input audio signal.
    - sample_rate (int): The sample rate of the audio signal in Hz.
    - params (list of floats): Compression parameters in the following order:
        0. threshold (dB): The level above which compression is applied. Range: -60 dB to 0 dB.
        1. ratio: The ratio of input change to output change above the threshold. Range: 1 to 10.
        2. attack_time (s): The time it takes for the compressor to react to an input above the threshold. Range: 0.0001s to 0.1s.
        3. release_time (s): The time it takes for the compressor to revert to no compression after the input drops below the threshold. Range: 0.005s to 3s.
        4. knee_dB: The transition range around the threshold where compression changes from no compression to full compression. Range: 0 dB to 24 dB.
        5. makeup_gain_dB: The gain applied after compression to bring the signal back to a desired level. Range: 0 dB to 20 dB.

    Returns:
    - np.ndarray: The compressed audio signal.

    The compression process includes converting the signal to a dB scale, applying knee
    smoothing, and calculating gain changes over time with specified attack and release times.
    The `@jit` decorator from Numba is used to compile this function to machine code for performance.
    """

    threshold = params[0] #-60 dB - 0 dB
    ratio = params[1] #1-10
    attack_time = params[2] #0.0001s - 0.1s
    release_time = params[3] #0.005s - 3s
    knee_dB = params[4] #0dB-24dB
    makeup_gain_dB = params[5] #0dB-20dB
    dtype=np.float32

    N = len(x)
    #dtype = x.dtype
    y = np.zeros(N, dtype=dtype)

    # Initialize separate attack and release times
    # Where do these numbers come from
    alpha_A = np.exp(-np.log(9) / (sample_rate * attack_time))
    alpha_R = np.exp(-np.log(9) / (sample_rate * release_time))

    # Turn the input signal into a uni-polar signal on the dB scale
    x_G = 20 * np.log10(np.abs(x) + 1e-8)  # x_uni casts type

    # Ensure there are no values of negative infinity
    x_G = np.clip(x_G, -96, None)

    # Static characteristics with knee
    y_G = np.zeros(N, dtype=dtype)

    # Below knee
    idx = np.where((2 * (x_G - threshold)) < -knee_dB)[0]
    y_G[idx] = x_G[idx].flatten() 

    # At knee
    idx = np.where((2 * np.abs(x_G - threshold)) <= knee_dB)[0]
    y_G[idx] = x_G[idx].flatten() + (
        (1 / ratio) * (((x_G[idx].flatten() - threshold + knee_dB) / 2) ** 2)
    ) / (2 * knee_dB)

    # Above knee threshold
    idx = np.where((2 * (x_G - threshold)) > knee_dB)[0]
    y_G[idx] = threshold + ((x_G[idx].flatten() - threshold) / ratio)
    x_L = x_G.flatten() - y_G.flatten()


    # this loop is slow but not vectorizable due to its cumulative, sequential nature. @autojit makes it fast(er).
    y_L = np.zeros(N, dtype=dtype)

    for n in range(1, N):
        # smooth over the gainChange
        if x_L[n].all() > y_L[n - 1].all():  # attack mode
            y_L[n] = (alpha_A * y_L[n - 1]) + ((1 - alpha_A) * x_L[n])
        else:  # release
            y_L[n] = (alpha_R * y_L[n - 1]) + ((1 - alpha_R) * x_L[n])

    # Convert to linear amplitude scalar; i.e. map from dB to amplitude
    y_L = np.clip(y_L, -96, None)
    lin_y_L = np.power(10.0, (-y_L / 20.0))
    y = lin_y_L * x  # Apply linear amplitude to input sample

    y *= np.power(10.0, makeup_gain_dB / 20.0)  # apply makeup gain

    return y
