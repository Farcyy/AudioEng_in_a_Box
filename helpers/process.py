
import tensorflow_io as tfio
import tensorflow as tf
import own_config as config


#from skimage.transform import resize

import tensorflow as tf
import tensorflow_io as tfio

import librosa
import tensorflow as tf

def load_and_preprocess(file_path):
    # Wrap the audio loading and conversion to mono in a tf.py_function
    # This allows integrating non-TensorFlow Python code with TensorFlow operations
    def load_audio_in_mono(file_path):
        # Decode the file path tensor to a Python string
        file_path = file_path.numpy().decode("utf-8")
        # Load the audio file with librosa, ensuring it is converted to mono
        audio, sr = librosa.load(file_path, sr=None, mono=True)
        return audio.astype('float32')

    # Use tf.py_function to call the Python function with TensorFlow tensors
    audio_tensor = tf.py_function(load_audio_in_mono, [file_path], tf.float32)
    
    # Setting the shape is necessary because tf.py_function does not infer output shapes
    audio_tensor.set_shape([None])

    return audio_tensor

def load_and_preprocesstfio(file_path):
    # Load the audio file as a waveform tensor.
    audio_io = tfio.audio.AudioIOTensor(file_path, dtype=tf.float32)
    # Slice the AudioIOTensor to get the tensor.
    # This operation is lazy; it does not read the file until the tensor is accessed.
    audio_tensor = audio_io.to_tensor()
    # If you have additional preprocessing steps, you can add them here.
    # For example, you could trim silence from the beginning and end.
    
    # Check if the audio is stereo (2 channels) and convert it to mono if true.
    print(type(audio_tensor))
    print('<-- audio_tensor.shape[-1]: -->')
    print(audio_tensor.shape)
    if audio_tensor.shape[-1] == 2:
        # If the audio has two channels, reduce it to mono by averaging the channels.
        audio_tensor = tf.reduce_mean(audio_tensor, axis=-1, keepdims=True)
    
    # Return the preprocessed audio tensor.
    return audio_tensor

def load_and_preprocesnew(file_path):
    # Load the audio file
    audio_binary = tf.io.read_file(file_path)
    audio, sample_rate = tf.audio.decode_wav(audio_binary)
    
    # Convert stereo audio to mono by averaging the channels
    if audio.shape[1] > 1:
        audio = tf.reduce_mean(audio, axis=1, keepdims=True)
    
    # Additional preprocessing steps can be added here
    
    return audio


def load_and_preprocessold(file_path):
    audio_io = tfio.audio.AudioIOTensor(file_path, dtype=tf.float32)
    # Slice the AudioIOTensor to get the tensor
    audio_tensor = audio_io.to_tensor()
    # Add preprocessing steps here if necessary
    # ...
    # If the audio tensor has two channels (stereo), convert it to mono by taking the mean
    print(type(audio_tensor))
    print('<-- audio.tensor.shape[-1]: -->')
    print(audio_tensor.shape)
    if audio_tensor.shape[-1] == 2:
        audio_tensor = tf.reduce_mean(audio_tensor, axis=-1, keepdims=True)
    return audio_tensor


def load_audio_with_fft(file_path):
    audio_binary = tf.io.read_file(file_path)
    waveform, _ = tf.audio.decode_wav(audio_binary)
    # Assuming a mono channel for waveform
    waveform = tf.squeeze(waveform, axis=-1)
    # Apply Short-Time Fourier Transform (STFT)
    stft = tf.signal.stft(waveform, frame_length=config.config['frame_length'], frame_step=config.config['frame_step'])
    return stft

def load_audio(file_path):
    audio_binary = tf.io.read_file(file_path)
    waveform, _ = tf.audio.decode_wav(audio_binary)
    # Assuming a mono channel for waveform
    waveform = tf.squeeze(waveform, axis=-1)
    # Apply Short-Time Fourier Transform (STFT)

    return waveform










