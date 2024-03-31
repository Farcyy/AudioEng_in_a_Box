##### helper functions for prepreprocessing pipeline notebook : "Preprocessing Pipeline.ipynb"

import librosa
import os
import tensorflow as tf
import torch
import numpy as np
import math
from tqdm import tqdm

#NOTE: Schmeißt sonst Fehler!
from preprocessing_helpers.compressor import compressor
from preprocessing_helpers.peq import parametric_eq

from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor, as_completed
import multiprocessing
from concurrent.futures import TimeoutError
import concurrent.futures


import numpy as np
from scipy import signal
import torch
import soundfile as sf

def apply_IR(audio_tensors, audio_sr, ir_list, random_seed):
    np.random.seed(random_seed)
    
    processed_tensors = []
    
    for audio_tensor in audio_tensors:
        ir_tensor = ir_list[np.random.randint(len(ir_list))]

        audio_np = audio_tensor.numpy() if isinstance(audio_tensor, torch.Tensor) else audio_tensor
        ir_np = ir_tensor.numpy() if isinstance(ir_tensor, torch.Tensor) else ir_tensor

        convolved_np = np.zeros_like(audio_np)
        for ch_idx in range(audio_np.shape[0]):
            convolved_channel = signal.fftconvolve(audio_np[ch_idx], ir_np[ch_idx], mode='full')
            # Ensure the convolved signal matches the original audio length
            if len(convolved_channel) > audio_np.shape[1]:
                # Trim the convolved signal if longer than original
                trim_start = (len(convolved_channel) - audio_np.shape[1]) // 2
                convolved_np[ch_idx] = convolved_channel[trim_start:trim_start + audio_np.shape[1]]
            else:
                # Pad the convolved signal if shorter than original
                pad_start = (audio_np.shape[1] - len(convolved_channel)) // 2
                convolved_np[ch_idx, pad_start:pad_start + len(convolved_channel)] = convolved_channel

        convolved_tensor = torch.from_numpy(convolved_np)
        processed_tensors.append(convolved_tensor)
    
    return processed_tensors



def define_styleparams(style):
    if style == "neutral":
        # ----------- compressor -------------
        threshold = -((torch.rand(1) * 10.0).numpy().squeeze() + 20.0)
        attack_sec = (torch.rand(1) * 0.020).numpy().squeeze() + 0.050
        release_sec = (torch.rand(1) * 0.200).numpy().squeeze() + 0.100
        ratio = (torch.rand(1) * 0.5).numpy().squeeze() + 1.5
        # ----------- parametric eq -----------
        low_shelf_gain_dB = (torch.rand(1) * 2.0).numpy().squeeze() + 1.0
        low_shelf_cutoff_freq = (torch.rand(1) * 120).numpy().squeeze() + 80
        first_band_gain_dB = 0.0
        first_band_cutoff_freq = 1000.0
        high_shelf_gain_dB = (torch.rand(1) * 2.0).numpy().squeeze() + 1.0
        high_shelf_cutoff_freq = (
            torch.rand(1) * 2000
        ).numpy().squeeze() + 6000
    elif style == "broadcast":
        # ----------- compressor -------------
        threshold = -((torch.rand(1) * 10).numpy().squeeze() + 40)
        attack_sec = (torch.rand(1) * 0.025).numpy().squeeze() + 0.005
        release_sec = (torch.rand(1) * 0.100).numpy().squeeze() + 0.050
        ratio = (torch.rand(1) * 2.0).numpy().squeeze() + 3.0
        # ----------- parametric eq -----------
        low_shelf_gain_dB = (torch.rand(1) * 4.0).numpy().squeeze() + 2.0
        low_shelf_cutoff_freq = (torch.rand(1) * 120).numpy().squeeze() + 80
        first_band_gain_dB = 0.0
        first_band_cutoff_freq = 1000.0
        high_shelf_gain_dB = (torch.rand(1) * 4.0).numpy().squeeze() + 2.0
        high_shelf_cutoff_freq = (
            torch.rand(1) * 2000
        ).numpy().squeeze() + 6000
    elif style == "telephone":
        # ----------- compressor -------------
        threshold = -((torch.rand(1) * 20.0).numpy().squeeze() + 20)
        attack_sec = (torch.rand(1) * 0.005).numpy().squeeze() + 0.001
        release_sec = (torch.rand(1) * 0.050).numpy().squeeze() + 0.010
        ratio = (torch.rand(1) * 1.5).numpy().squeeze() + 1.5
        # ----------- parametric eq -----------
        low_shelf_gain_dB = -((torch.rand(1) * 6).numpy().squeeze() + 20)
        low_shelf_cutoff_freq = (
            torch.rand(1) * 200
        ).numpy().squeeze() + 200
        first_band_gain_dB = (torch.rand(1) * 4).numpy().squeeze() + 12
        first_band_cutoff_freq = (
            torch.rand(1) * 1000
        ).numpy().squeeze() + 1000
        high_shelf_gain_dB = -((torch.rand(1) * 6).numpy().squeeze() + 20)
        high_shelf_cutoff_freq = (
            torch.rand(1) * 2000
        ).numpy().squeeze() + 4000
    elif style == "bright":
        # ----------- compressor -------------
        ratio = 1.0
        threshold = 0.0
        attack_sec = 0.050
        release_sec = 0.250
        # ----------- parametric eq -----------
        low_shelf_gain_dB = -((torch.rand(1) * 6).numpy().squeeze() + 20)
        low_shelf_cutoff_freq = (
            torch.rand(1) * 200
        ).numpy().squeeze() + 200
        first_band_gain_dB = 0.0
        first_band_cutoff_freq = 1000.0
        high_shelf_gain_dB = (torch.rand(1) * 6).numpy().squeeze() + 20
        high_shelf_cutoff_freq = (
            torch.rand(1) * 2000
        ).numpy().squeeze() + 8000

    elif style == "warm":
        # ----------- compressor -------------
        ratio = 1.0
        threshold = 0.0
        attack_sec = 0.050
        release_sec = 0.250
        # ----------- parametric eq -----------
        low_shelf_gain_dB = (torch.rand(1) * 6).numpy().squeeze() + 20
        low_shelf_cutoff_freq = (
            torch.rand(1) * 200
        ).numpy().squeeze() + 200
        first_band_gain_dB = 0.0
        first_band_cutoff_freq = 1000.0
        high_shelf_gain_dB = -(torch.rand(1) * 6).numpy().squeeze() + 20
        high_shelf_cutoff_freq = (
            torch.rand(1) * 2000
        ).numpy().squeeze() + 8000

    else:
        raise ValueError(f"Invalid style: {style}.")

    # Return a dictionary of parameters
    return {
        'threshold': threshold,
        'attack_sec': attack_sec,
        'release_sec': release_sec,
        'ratio': ratio,
        'low_shelf_gain_dB': low_shelf_gain_dB,
        'low_shelf_cutoff_freq': low_shelf_cutoff_freq,
        'first_band_gain_dB': first_band_gain_dB,
        'first_band_cutoff_freq': first_band_cutoff_freq,
        'high_shelf_gain_dB': high_shelf_gain_dB,
        'high_shelf_cutoff_freq': high_shelf_cutoff_freq
    }


def get_random_patch_tf(x, sample_rate, length_samples):
    length = int(length_samples)
    max_start_idx = x.shape[-1] - length  # Calculate the maximum possible start index
    if max_start_idx <= 0:
        print("Error: Audio file is too short for the desired patch length.")
        return None

    silent = True
    attempts = 0
    while silent and attempts < 10:  # Add a limit to attempts to avoid infinite loops
        start_idx = tf.random.uniform(shape=(), minval=0, maxval=max_start_idx, dtype=tf.int32)
        stop_idx = start_idx + length
        x_crop = x[:, start_idx:stop_idx]

        # Check for silence
        frames = length // sample_rate
        silent_frames = []
        for n in range(frames):
            start_frame_idx = int(n * sample_rate)
            stop_frame_idx = start_frame_idx + sample_rate
            x_frame = x_crop[:, start_frame_idx:stop_frame_idx]
            if tf.reduce_mean(x_frame ** 2) > 3e-4:
                silent_frames.append(False)
            else:
                silent_frames.append(True)
        silent = all(silent_frames)  # Use all() to ensure the entire patch isn't silent
        attempts += 1

    if silent:
        print("Warning: Unable to find a non-silent patch in the audio after multiple attempts. Returning white noise.")
        # Generate white noise of the same shape as the desired patch
        x_crop = tf.random.uniform(shape=(x.shape[0], length), minval=-1.0, maxval=1.0, dtype=tf.float32)

    # Normalize the cropped audio
    x_crop = x_crop / tf.reduce_max(tf.abs(x_crop))

    return x_crop



def get_file_paths(directory): ### for simple folder structure without subfolders
    file_paths = []  # List to store file paths
    print(f"Looking in directory: {directory}")  # Debug print
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            #print(f"Found file: {file_path}")  # Debug print
            file_paths.append(file_path)
    return file_paths

def get_file_paths_sub(directory): ### for folder structure like jamendo, with subfolders!
    file_paths = []  # List to store file paths
    print(f"Looking in directory: {directory}")  # Debug print
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths






def apply_styleTransfer(audio_tensor_list, sr, style, normalize = False):
    style_params = define_styleparams(style)
    processed_audio_list = []

    # Use tqdm to track the progress
    with tqdm(total=len(audio_tensor_list), desc="Applying style transfer") as pbar:
        for audio_tensor in audio_tensor_list:
            # Convert the input audio_tensor from TensorFlow to NumPy and then to PyTorch tensor
            audio_tensor = torch.tensor(audio_tensor.numpy())

            processed_audio_tensor = torch.zeros_like(audio_tensor)

            for ch_idx in range(audio_tensor.shape[0]):
                x_peq_ch = parametric_eq(
                    audio_tensor[ch_idx, :],
                    sr,
                    low_shelf_gain_dB=style_params['low_shelf_gain_dB'],
                    low_shelf_cutoff_freq=style_params['low_shelf_cutoff_freq'],
                    first_band_gain_dB=style_params['first_band_gain_dB'],
                    first_band_cutoff_freq=style_params['first_band_cutoff_freq'],
                    high_shelf_gain_dB=style_params['high_shelf_gain_dB'],
                    high_shelf_cutoff_freq=style_params['high_shelf_cutoff_freq'],
                )

                x_comp_ch = compressor(
                    x_peq_ch,
                    sr,
                    threshold=style_params['threshold'],
                    ratio=style_params['ratio'],
                    attack_time=style_params['attack_sec'],
                    release_time=style_params['release_sec'],
                )

                # Convert x_comp_ch to a PyTorch tensor
                x_comp_ch = torch.tensor(x_comp_ch)

                processed_audio_tensor[ch_idx, :] = x_comp_ch
                
            processed_audio_list.append(processed_audio_tensor)
            pbar.update(1)  # Update the progress bar

    # Concatenate the list of tensors into a single tensor
    processed_audio_tensor = torch.stack(processed_audio_list, dim=0)

    return processed_audio_tensor






def load_audio_file(filepath, target_sr=24000):
    """
    Load a single audio file, resample it, and convert it to a TensorFlow tensor.
    """
    audio, sr = librosa.load(filepath, sr=target_sr, mono=False)
    audio_tensor = tf.convert_to_tensor(audio, dtype=tf.float32)
    return audio_tensor, sr

def batch_generator(audio_filepaths, target_sr=24000, batch_size=10):
    """
    Generate batches of audio data.
    """
    total_batches = len(audio_filepaths) // batch_size + (1 if len(audio_filepaths) % batch_size > 0 else 0)
    for i in tqdm(range(0, len(audio_filepaths), batch_size), total=total_batches, desc="Processing batches"):
        batch_paths = audio_filepaths[i:i+batch_size]
        batch_tensors = []
        batch_srs = []
        for path in batch_paths:
            tensor, sr = load_audio_file(path, target_sr)
            batch_tensors.append(tensor)
            batch_srs.append(sr)
        yield batch_tensors, batch_srs


def load_and_preprocess(audio_filepaths, target_sr=24000, normalize = False):
    """
    simple loading and resampling!
    Load audio files, resample them to a target sample rate, and convert to TensorFlow tensors.
    
    :param musdb_filepaths: List of file paths to audio files.
    :param target_sr: Target sampling rate for resampling. Default is 22050 Hz.
    :return: A tuple of lists containing audio tensors and their sample rates.
    """
    audiotensors = []
    audio_srs = []
    songnames = []
    
    for filepath in audio_filepaths:
        # Load the audio file and resample it to the target_sr
        audio, sr = librosa.load(filepath, sr=target_sr, mono=False)
        
        # Convert the audio array to a TensorFlow tensor
        audio_tensor = tf.convert_to_tensor(audio, dtype=tf.float32)
        
        # Append the tensor and sample rate to their respective lists
        if normalize == True:
            #audio_tensor = tf.clip_by_value(audio_tensor, -1.0, 1.0)  # Ensures values are within the range
            max_val = tf.reduce_max(tf.abs(audio_tensor)) ## normalising over both channels with same max value! (instead of normalizing each channel by its max value)
            audio_tensor = audio_tensor / max_val

        # Append the normalized tensor and sample rate to their respective lists
        audiotensors.append(audio_tensor)
        audio_srs.append(sr)

    return audiotensors, audio_srs

# Function to be executed in parallel
def process_audio_file(file_path, target_sr, no_samples_styletransfer):
    # Load the audio file
    audio, sr = librosa.load(file_path, sr=None, mono=False)  # Load as stereo
    
    # Resample if the sample rate is different from the target sample rate
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    
    # Ensure audio is two-dimensional
    if audio.ndim == 1:
        audio = np.expand_dims(audio, axis=0)  # Adds a new dimension, making it 2D
    
    # Convert the audio array to a TensorFlow tensor
    audio_tensor = tf.convert_to_tensor(audio, dtype=tf.float32)
    
    # Apply the get_random_patch_tf function
    audio_schnippsel = get_random_patch_tf(audio_tensor, target_sr, no_samples_styletransfer)
    
    return audio_schnippsel

def load_and_apply_random_patch(file_paths, target_sr, no_samples_styletransfer):
    schnippsel_list = []
    num_cores = max(1, os.cpu_count() - 2)  # Use number of cores minus two, but at least one

    # Function to process an individual audio file
    def process_audio_file(file_path, target_sr, no_samples_styletransfer):
        try:
            #print(f"Processing file: {file_path}")  # Log the file being processed
            audio, sr = librosa.load(file_path, sr=None, mono=False)  # Load as stereo
            if sr != target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            if audio.ndim == 1:
                audio = np.vstack((audio, audio))
            audio_tensor = tf.convert_to_tensor(audio, dtype=tf.float32)
            audio_schnippsel = get_random_patch_tf(audio_tensor, target_sr, no_samples_styletransfer)
            return audio_schnippsel
        except Exception as exc:
            print(f'An error occurred processing {file_path}: {exc}')
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_cores) as executor:
        # Create a dictionary to track futures to file paths
        future_to_file = {}
        for file_path in file_paths:
            future = executor.submit(process_audio_file, file_path, target_sr, no_samples_styletransfer)
            future_to_file[future] = file_path

        # Initialize tqdm progress bar
        progress = tqdm(total=len(file_paths), desc="Processing audio files")

        # Process futures as they complete
        for future in concurrent.futures.as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                audio_schnippsel = future.result(timeout=60)
                if audio_schnippsel is not None:
                    schnippsel_list.append(audio_schnippsel)
            except concurrent.futures.TimeoutError:
                print(f"A file processing timed out and will be skipped: {file_path}")
            except Exception as exc:
                print(f'An error occurred processing {file_path}: {exc}')
            finally:
                progress.update(1)

        progress.close()

    max_length = min(no_samples_styletransfer, max(len(x) for x in schnippsel_list if x is not None))

    for i in range(len(schnippsel_list)):
        x = schnippsel_list[i]
        if x is not None:
            if len(x) < max_length:
                schnippsel_list[i] = np.pad(x, (0, max_length - len(x)), mode='constant')
            elif len(x) > max_length:
                schnippsel_list[i] = x[:max_length]

    stacked_audio = tf.stack([x for x in schnippsel_list if x is not None], axis=0)

    return stacked_audio



def split_data(filepaths, train_ratio=0.9):
    """
    Splits the filepaths into training and testing sets based on the given ratio.
    
    :param filepaths: List of file paths to split
    :param train_ratio: Ratio of filepaths to include in the training set
    :return: A tuple of (train_filepaths, test_filepaths)
    """
    # Determine the split index
    split_index = math.ceil(len(filepaths) * train_ratio)
    
    # Split the filepaths
    train_filepaths = filepaths[:split_index]
    test_filepaths = filepaths[split_index:]
    
    return train_filepaths, test_filepaths


def save_tensor_as_audio(tensor, filenames, directory, sample_rate=44100):
    """
    Saves each slice of a tensor as an audio file with provided filenames,
    changing their extension to .wav. Adjusted to handle TensorFlow tensors.
    
    Parameters:
    - tensor: A TensorFlow tensor or similar, shaped as (n_samples, n_channels, n_audio_samples).
    - filenames: A list of filenames corresponding to each sample in the tensor.
    - directory: The directory to save the audio files to.
    - sample_rate: The sample rate of the audio files (in Hz).
    """
    # Ensure the output directory exists
    os.makedirs(directory, exist_ok=True)
    
    # Check if the number of filenames matches the number of samples in the tensor
    if len(filenames) != tensor.shape[0]:
        raise ValueError("The number of filenames does not match the number of samples in the tensor.")
    
    # Iterate over each sample and its corresponding filename in the tensor
    for i, (sample, filename) in enumerate(zip(tensor, filenames)):
        # Convert TensorFlow tensor to a NumPy array if necessary
        if isinstance(sample, tf.Tensor):
            sample = sample.numpy()
        
        # Extract the base name without extension and append .wav
        base_filename = os.path.splitext(os.path.basename(filename))[0] + ".wav"
        full_path = os.path.join(directory, base_filename)
        
        # Transpose the sample (channels last to channels first for soundfile compatibility) and save it
        sf.write(full_path, sample.T, samplerate=sample_rate)

# Ensure to adjust other parts of your code to handle TensorFlow tensors correctly.
        

def split_into_batches(filepaths, n_batches):
    """Split the filepaths into n_batches roughly equal parts."""
    for i in range(n_batches):
        yield filepaths[i::n_batches]

def process_and_safe_batches(batches, directory, filepaths, sample_rate, no_samples):
    """
    Processes and saves audio files in batches without creating new folders for each batch.
    
    Parameters:
    - batches: List of lists, each sublist contains file paths for one batch.
    - directory: The directory to save the processed audio files.
    - filepaths: The full list of original file paths for all audio files.
    - sample_rate: The sample rate of the audio files.
    - no_samples: Number of samples for each audio file after processing.
    """
    total_batches = len(batches)
    processed_files_count = 0  # Keep track of the number of processed files

    for batch_idx, batch in enumerate(batches):
        print(f"Processing batch no: {batch_idx + 1} of {total_batches}")
        
        # Load and apply random patches to the batch
        processed_tensor = load_and_apply_random_patch(batch, sample_rate, no_samples)
        
        # Determine the filenames for the current batch based on the processed files count
        batch_filenames = filepaths[processed_files_count:processed_files_count + len(batch)]
        processed_files_count += len(batch)  # Update the count of processed files
        # Save the processed tensor slices as audio files, directly in the specified directory
        save_tensor_as_audio(processed_tensor, batch_filenames, directory, sample_rate)

# Assuming the setup is as follows:
# - `batches` is a list of file path batches for processing.
# - `directory` is the target directory where all processed files will be saved.
# - `filepaths` is a list of all original file paths corresponding to the audio files.
# - `sample_rate` and `no_samples` are parameters for audio processing.

# You can call `process_and_safe_batches` with your specific parameters.

def normalize_audio_tensors(audio_tensors):
    """
    Normalizes a batch of audio files represented as a TensorFlow tensor.
    Each file is normalized based on its own maximum value across its channels and samples.
    
    Parameters:
    audio_tensors (tf.Tensor): A 3D tensor of shape (num_files, num_channels, num_samples).
    
    Returns:
    tf.Tensor: The normalized audio tensors.
    """
    # Calculate the maximum absolute value for each file
    max_vals = tf.reduce_max(tf.abs(audio_tensors), axis=[1, 2], keepdims=True)
    
    # Normalize each audio file by its own maximum value
    normalized_audio_tensors = audio_tensors / max_vals
    
    return normalized_audio_tensors