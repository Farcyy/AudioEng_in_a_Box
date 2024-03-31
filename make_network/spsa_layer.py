import tensorflow as tf
import numpy as np
import scipy.signal
from numba import jit
from numba import jit
#from DSP.peq import parametric_eq
#from DSP.drc import drc
from own_config import config


tf.random.set_seed(0)
np.random.seed(0)

# Implementation of SPSA Processor (https://github.com/adobe-research/DeepAFx/blob/main/scripts/custom_grad_example6.py)

tf.random.set_seed(0)
np.random.seed(0)

"""def truncate_params(params, decimal_places=3):
    scale = 10 ** decimal_places
    return tf.floor(params * scale) / scale
"""

sample_rate=config['std_sample_rate']

#Biquad Implementation
@jit(nopython=True)
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
            One of 0 (low_shelf), 1(high_shelf), or 2(peaking)

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

    b = np.array([b0, b1, b2]) / a0
    a = np.array([a0, a1, a2]) / a0

    return b, a

#PEQ Implementation
# Adapted from https://github.com/csteinmetz1/pyloudnorm/blob/master/pyloudnorm/iirfilter.py
def parametric_eq(x, sample_rate, params):
    
    """
    Applies a series of parametric equalization filters to an input audio signal.

    This function sequentially applies a low shelf filter, four band peaking filters, and a high shelf filter to the input signal, based on the parameters provided. The filtering process utilizes second-order sections (SOS) for stability and precision.

    Parameters:
    - x (np.ndarray): The input audio signal as a NumPy array.
    - sample_rate (int): The sample rate of the audio signal in Hz.
    - params (list of floats): Parameters for the equalization filters. This list should contain 18 elements, divided as follows:
        - [0], [1], [2]: Low shelf filter gain (in dB), cutoff frequency (in Hz), and Q factor, respectively.
        - [3] to [14]: Four sets of parameters for the band peaking filters, each set comprising gain (in dB), cutoff frequency (in Hz), and Q factor. These sets are for the first, second, third, and fourth band peaking filters, respectively.
        - [15], [16], [17]: High shelf filter gain (in dB), cutoff frequency (in Hz), and Q factor, respectively.

    Returns:
    - np.ndarray: The filtered audio signal as a NumPy array, with the same shape as the input.
    """

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

# Apply-Function for PEQ
def apply_peq(x, peq_arg):

    """
    Applies parametric equalization based on a set of arguments to an audio signal.

    This function first sanitizes the `peq_arg` array to ensure no NaN values are present, then calculates the parameters for a series of equalization filters (one low shelf, four peaking bands, and one high shelf) based on the input arguments. These parameters are then used to apply parametric equalization to the input audio signal. The function supports input in both TensorFlow tensor and NumPy array formats, seamlessly converting between them as necessary.

    Parameters:
    - x (np.ndarray or tf.Tensor): The input audio signal. Can be a TensorFlow tensor or a NumPy array.
    - peq_arg (np.ndarray or tf.Tensor): A tensor or array containing the parameters for the equalization filters. Expected to have specific values that map to equalizer settings (gain, cutoff frequency, and Q factor for various bands).

    Returns:
    - np.ndarray or tf.Tensor: The equalized audio signal, in the same format as the input signal (NumPy array or TensorFlow tensor).

    Requires:
    - This function requires the NumPy library (imported as `np`) and TensorFlow (imported as `tf`) for its execution.
    - The function `parametric_eq`, which actually applies the parametric equalization, must be defined elsewhere in the code.

    Note: 
    - The `peq_arg` array is expected to have 18 elements, corresponding to 6 sets of parameters (gain, cutoff frequency, and Q factor) for the low shelf, four peaking bands, and high shelf filters.
    - The function includes error handling to replace any infinities or NaN values in the output with zeros."""

    #Accounting for corrupted input
    peq_arg=tf.where(tf.math.is_nan(peq_arg), tf.zeros_like(peq_arg), peq_arg)

    low_shelf_gain_dB = peq_arg[0]*48-24
    low_shelf_cutoff_freq = 16 + (np.exp(peq_arg[1]) - 1)  / (np.e - 1) * (128-16)
    low_shelf_q_factor = peq_arg[2]*8+0.707
    first_band_gain_dB = peq_arg[3]*48-24 
    first_band_cutoff_freq = 128 + (np.exp(peq_arg[4]) - 1) / (np.e - 1) * (512 - 128) 
    first_band_q_factor = peq_arg[5]*8+0.707
    second_band_gain_dB = peq_arg[6]*48-24
    second_band_cutoff_freq = 512 + (np.exp(peq_arg[7]) - 1)  / (np.e - 1) * (1024 - 512) 
    second_band_q_factor = peq_arg[8]*8+0.707
    third_band_gain_dB = peq_arg[9]*48-24
    third_band_cutoff_freq = 1024 + (np.exp(peq_arg[10]) - 1)  / (np.e - 1) * (2048 - 1024)
    third_band_q_factor = peq_arg[11]*8+0.707
    fourth_band_gain_dB = peq_arg[11]*48-24
    fourth_band_cutoff_freq =  2048 + (np.exp(peq_arg[12]) - 1) / (np.e - 1) * (4096 - 2048) 
    fourth_band_q_factor = abs(peq_arg[14])*8+0.707
    high_shelf_gain_dB = peq_arg[15]*48-24
    high_shelf_cutoff_freq = 4096 + (np.exp(peq_arg[16]) - 1)  / (np.e - 1) * (10240 - 4096)
    high_shelf_q_factor = peq_arg[17]*8+0.707

    dtype=np.float32

    """print   (
            "\n+++PEQ-Paramter+++",
            "\nLowCut", low_shelf_gain_dB, low_shelf_cutoff_freq, low_shelf_q_factor, 
            "\nFirstBand", first_band_gain_dB, first_band_cutoff_freq, first_band_q_factor, 
            "\nSecondBand", second_band_gain_dB, second_band_cutoff_freq, second_band_q_factor, 
            "\nThirdBand", third_band_gain_dB, third_band_cutoff_freq, third_band_q_factor, 
            "\nFourthBand", fourth_band_gain_dB, fourth_band_cutoff_freq, fourth_band_q_factor, 
            "\nHighCut", high_shelf_gain_dB, high_shelf_cutoff_freq, high_shelf_q_factor
            )"""

    peq_arg=np.array([
        low_shelf_gain_dB, low_shelf_cutoff_freq, low_shelf_q_factor,
        first_band_gain_dB, first_band_cutoff_freq, first_band_q_factor,
        second_band_gain_dB, second_band_cutoff_freq, second_band_q_factor,
        third_band_gain_dB, third_band_cutoff_freq, third_band_q_factor,
        fourth_band_gain_dB, fourth_band_cutoff_freq, fourth_band_q_factor,
        high_shelf_gain_dB, high_shelf_cutoff_freq, high_shelf_q_factor
        ])

    #Convert to numpy function
    tensor=False
    if(isinstance(x, tf.Tensor)):
        x = x.numpy().astype(np.float32)
        tensor=True

    #Apply EQ
    output = parametric_eq(
        x = x,
        sample_rate = sample_rate,
        params = peq_arg
    )

    #Convert back to Tensor
    if(tensor==True):
        output = tf.convert_to_tensor(output, dtype=tf.float32)
        
    #Numerical Stability
    if not np.isfinite(output).all():
                  output[np.isinf(output)] = 0
                  output[np.isnan(output)] = 0

    return output

@jit(nopython=True)
def drc(
    x,
    sample_rate,
    params, #threshold, ratio, attack_time, release_time, knee_dB, makeup_gain_dB,
):
    """
    Applies dynamic range compression to an audio signal.

    Dynamic range compression reduces the volume of loud sounds or amplifies quiet sounds by narrowing or compressing an audio signal's dynamic range. The compression parameters are adjustable, allowing for flexible control over the compression effect.

    Parameters:
    - x (np.ndarray): The input audio signal as a NumPy array.
    - sample_rate (int): The sample rate of the audio signal in Hz.
    - params (list of floats): Parameters for the dynamic range compression. This list should contain 6 elements:
        - threshold (float): The threshold level (in dB) above which compression is applied. Range: -60 dB to 0 dB.
        - ratio (float): The compression ratio. Range: 1 (no compression) to 10 (high compression).
        - attack_time (float): The time (in seconds) it takes for the compression to kick in after the signal exceeds the threshold. Range: 0.0001s to 0.1s.
        - release_time (float): The time (in seconds) it takes for the compression to cease after the signal falls below the threshold. Range: 0.005s to 3s.
        - knee_dB (float): The soft knee size (in dB), which controls the transition smoothness from uncompressed to compressed signal. Range: 0 dB (hard knee) to 24 dB (soft knee).
        - makeup_gain_dB (float): The gain (in dB) applied to the signal after compression to make up for the loss in volume. Range: 0 dB to 20 dB.

    Returns:
    - np.ndarray: The compressed audio signal as a NumPy array, with the same shape as the input.

    Note:
    - The compression is implemented using a feed-forward design with adjustable attack and release times, knee width, and makeup gain.
    - This function is designed for single-channel audio signals (mono).
    """

    threshold = params[0] #-60 dB - 0 dB
    ratio = params[1] #1-10
    attack_time = params[2] #0.0001s - 0.1s
    release_time = params[3] #0.005s - 3s
    knee_dB = params[4] #0dB-24dB
    makeup_gain_dB = params[5] #0dB-20dB
    dtype=np.float32

    N = len(x)
    y = np.zeros(N, dtype=dtype)

    # Initialize separate attack and release times
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

    y_L = np.zeros(N, dtype=dtype)
    
    for n in range(1, N):
    # Determine if we are in attack or release mode
        attack_mode = x_L[n] > y_L[n - 1]
        
        # Compute alpha based on mode
        alpha = alpha_A if attack_mode else alpha_R
        
        # Update y_L[n] efficiently
        y_L[n] = alpha * y_L[n - 1] + (1 - alpha) * x_L[n]

    # Convert to linear amplitude scalar; i.e. map from dB to amplitude
    y_L = np.clip(y_L, -96, None)
    lin_y_L = np.power(10.0, (-y_L / 20.0))
    y = lin_y_L * x  # Apply linear amplitude to input sample

    y *= np.power(10.0, makeup_gain_dB / 20.0)  # apply makeup gain

    return y

def apply_drc(x, drc_arg):
    """
    Apply Dynamic Range Compression (DRC) to the input signal 'x' based on the given DRC parameters 'drc_arg'.
    
    Parameters:
        x (np.ndarray or tf.Tensor): Input signal.
        drc_arg (np.ndarray or tf.Tensor): DRC parameters: [threshold, ratio, attack_time, release_time, knee_dB, makeup_gain_dB].
        
    Returns:
        np.ndarray or tf.Tensor: Output signal after applying DRC.
    """
    
    # Correcting corrupted input
    drc_arg = tf.where(tf.math.is_nan(drc_arg), tf.zeros_like(drc_arg), drc_arg)
    drc_arg = tf.where(tf.equal(drc_arg, 0), tf.fill(tf.shape(drc_arg), 1e-10), drc_arg)

    # Extracting DRC parameters
    threshold = 0 - drc_arg[0] * 60  # -60 dB - 0 dB
    ratio = drc_arg[1] * 10
    attack_time = max(drc_arg[2] / 10, 0.0001)  # 0.0001s - 0.1s
    release_time = max(drc_arg[3] * 3, 0.005)  # 0.005s - 3s
    knee_dB = drc_arg[4] * 24  # 0dB - 24dB
    makeup_gain_dB = drc_arg[5] * 20  # 0dB - 20dB

    drc_arg = np.array([threshold, ratio, attack_time, release_time, knee_dB, makeup_gain_dB])

    # Printing DRC parameters (optional)
    """
    print("\n+++DRC-Parameter+++"
          "\nThreshold", drc_arg[0],
          "\nRatio", drc_arg[1],
          "\nAttack Time", drc_arg[2],
          "\nRelease Time", drc_arg[3],
          "\nKnee", drc_arg[4],
          "\nMakeUp", drc_arg[5])
    """

    """Dynamic range compressor implemented in TensorFlow."""

    # Converting input to Numpy Array if it's a Tensor
    tensor = False
    if isinstance(x, tf.Tensor):
        x = x.numpy().astype(np.float32)
        tensor = True

    # Applying DRC
    output = drc(
        x=x,
        sample_rate=config['std_sample_rate'],
        params=drc_arg
    )

    # Converting back to Tensor if the input was a Tensor
    if tensor:
        output = tf.convert_to_tensor(output, dtype=tf.float32)

    # Ensuring numerical stability
    if not np.isfinite(output).all():
        output[np.isinf(output)] = 0
        output[np.isnan(output)] = 0

    return output

#SPSA for PEQ
# Define SPSA Function for perturbed equilibrium
@tf.custom_gradient
def spsa_peq(x, y):

    # Define epsilon for numerical differentiation
    epsilon = 0.005
    
    # Function to be applied to each element of the batch
    def _func(xe, ye):
        """Function applied to each element of the batch"""
        # Apply perturbed equilibrium function
        x = apply_peq(xe, ye)
        return x

    # Function to compute for entire batch
    def func(x, y):
        # Iterate over batch items
        z = []
        for i in range(x.shape[0]):
            z.append(_func(x[i], y[i]))
        z = tf.stack(z)
        return z

    # Gradient function
    def grad_fn(dy):
        """Gradient applied to each batch"""

        # Gradient function applied to each element of the batch
        def _grad_fn(dye, xe, ye):
            """Gradient applied to each element of the batch"""

            # Gradient w.r.t. x
            J_plus = _func(xe + epsilon, ye)
            J_minus = _func(xe - epsilon, ye)
            gradx = (J_plus -  J_minus) / (2.0 * epsilon)
            vecJxe = gradx * dye

            # Gradient w.r.t. y
            yc = ye.numpy()
            # Pre-allocate vector * Jacobian output
            vecJye = np.zeros_like(ye)

            # Iterate over each parameter and compute the output
            for i in range(ye.shape[0]):
                yc[i] = yc[i] + epsilon
                J_plus = _func(xe, yc)
                yc[i] = yc[i] - 2 * epsilon
                J_minus = _func(xe, yc)
                grady = (J_plus - J_minus) / (2.0 * epsilon)
                yc[i] = yc[i] + 1 * epsilon
                vecJye[i] = np.dot(np.transpose(dye), grady)

            return vecJxe, vecJye

        dy1 = []
        dy2 = []
        for i in range(dy.shape[0]):
            vecJxe, vecJye = _grad_fn(dy[i], x[i], y[i])
            dy1.append(vecJxe)
            dy2.append(vecJye)
        return tf.stack(dy1), tf.stack(dy2)

    return func(x, y), grad_fn

#SPSA for DRC
@tf.custom_gradient
def spsa_drc(x, y):

    epsilon = 0.005

    def _func(xe, ye):
        """Function applied to each element of the batch"""
        x=apply_drc(xe, ye)   
        return x

    def func(x, y):

        # Iterate over batch item
        z = []
        for i in range(x.shape[0]):
            z.append(_func(x[i], y[i]))
        z = tf.stack(z)
        return z

    def grad_fn(dy):
        """Gradient applied to each batch"""

        def _grad_fn(dye, xe, ye):
            """Gradient applied to each element of the batch"""

            # Grad w.r.t x. NOTE: this is approximate and should +-epsilon for each element
            J_plus = _func(xe + epsilon, ye)
            J_minus = _func(xe - epsilon, ye)
            gradx = (J_plus -  J_minus)/(2.0*epsilon)
            vecJxe = gradx * dye

            # Grad w.r.t y
            yc = ye.numpy()

            # pre-allocate vector * Jaccobian output
            vecJye = np.zeros_like(ye)

            # Iterate over each parameter and compute the output
            for i in range(ye.shape[0]):

                yc[i] = yc[i] + epsilon
                J_plus = _func(xe, yc)
                yc[i] = yc[i] - 2*epsilon
                J_minus = _func(xe, yc)
                grady = (J_plus -  J_minus)/(2.0*epsilon)
                yc[i] = yc[i] + 1*epsilon
                vecJye[i] = np.dot(np.transpose(dye), grady)

            return vecJxe, vecJye

        dy1 = []
        dy2 = []
        for i in range(dy.shape[0]):
            vecJxe, vecJye = _grad_fn(dy[i], x[i], y[i])
            dy1.append(vecJxe)
            dy2.append(vecJye)
        return tf.stack(dy1), tf.stack(dy2)

    return func(x, y), grad_fn

# Create a Keras Layer for PEQ 
class PEQLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(PEQLayer, self).__init__()

    def build(self, input_shape):
        # Store the input shape for later use
        self.oshape = input_shape
        super(PEQLayer, self).build(input_shape)

    def call(self, inputs):
        # Separate inputs into x and params
        x = inputs[0]
        params = inputs[1]

        # Apply SPSA function using tf.py_function
        ret = tf.py_function(func=spsa_peq,  # Custom SPSA function defined earlier
                             inp=[x, params],  # Inputs to the SPSA function
                             Tout=tf.float32)  # Output datatype
        ret.set_shape(x.get_shape())  # Set the shape of the returned tensor

        return ret

# Create a Keras Layer for Compressor 
class DRCLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(DRCLayer, self).__init__()

    def build(self, input_shape):
        self.oshape = input_shape
        super(DRCLayer, self).build(input_shape)

    def call(self, inputs):
        x = inputs[0]
        params = inputs[1]

        """print("+++Layer Parameter+++")
        print("\n Sginal: ", x )"""

        ret = tf.py_function(func=spsa_drc,
                             inp=[x, params],
                             Tout=tf.float32)
        ret.set_shape(x.get_shape())
        
        return ret

# Combined DSP Layer
class DSPLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(DSPLayer, self).__init__()
        # Initialize the PEQ and DRC sub-layers
        self.peq_layer = PEQLayer()
        self.drc_layer = DRCLayer()

    def build(self, input_shape):
        # Store the input shape for later use
        self.oshape = input_shape
        # Build the PEQ and DRC sub-layers
        self.peq_layer.build(input_shape)
        self.drc_layer.build(input_shape)
        super(DSPLayer, self).build(input_shape)

    def call(self, inputs):
        # Separate inputs into signal and params
        signal, params = inputs
        # Split the parameters into PEQ and DRC parameters
        peq_params, drc_params = tf.split(params, [18, 6], axis=1)
        # Apply PEQ layer to the signal
        peq_output = self.peq_layer([signal, peq_params])
        # Apply DRC layer to the PEQ output
        drc_output = self.drc_layer([peq_output, drc_params])
        return drc_output

