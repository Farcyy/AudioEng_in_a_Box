import matplotlib.pyplot as plt
import os
import tensorflow as tf
import numpy as np

from DSP import parametric_eq
import pdb
import re
import own_config as config

from helpers.process import load_and_preprocess
import librosa
import librosa.display


np.random.seed(config.config['random_seed'])
tf.random.set_seed(config.config['random_seed'])

def get_file_paths(directory):
    # List all files in the given directory, but exclude .DS_Store files and potentially other non-audio file types
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.wav') and not f.startswith('.')]

def extract_core_identifier(filepath):
    # Extract the identifier, which can be numeric or a song name, ignoring 'IR', style, and file extension
    # This regex captures a sequence of characters that ends with '_IR.wav', ignoring any style-related parts
    match = re.search(r'/([^/_]+?)(?:_[\w-]+)?_IR\.wav$', filepath)
    if match:
        return match.group(1).lower()  # Return the identifier in lowercase to ensure case-insensitive matching
    return None

def find_file_pairs(dir1_files, dir2_files):
    # Create dictionaries to store files with their core identifiers as keys
    dir1_dict = {extract_core_identifier(filepath): filepath for filepath in dir1_files}
    dir2_dict = {extract_core_identifier(filepath): filepath for filepath in dir2_files}

    # Lists to hold matching file paths
    dir1_matched_files = []
    dir2_matched_files = []

    # Find matching pairs based on core identifiers
    for identifier, filepath in dir1_dict.items():
        if identifier in dir2_dict:
            dir1_matched_files.append(filepath)
            dir2_matched_files.append(dir2_dict[identifier])
    
    return dir1_matched_files, dir2_matched_files

def find_unique_file_paths(dir1_files, dir2_files, no_of_files=20000):
    # Create dictionaries to store files with their core identifiers as keys
    dir1_identifiers = {extract_core_identifier(filepath) for filepath in dir1_files}
    
    # List to hold unique file paths from dir2 that are not in dir1
    unique_dir2_files = []
    
    # Iterate through dir2 files to find unique core identifiers not in dir1
    for filepath in dir2_files:
        identifier = extract_core_identifier(filepath)
        if identifier not in dir1_identifiers:
            unique_dir2_files.append(filepath)
            # If we've reached the requested number of unique files, stop the search
            if len(unique_dir2_files) == no_of_files:
                break
    
    return unique_dir2_files

def shuffle_batch_elements(batch, seed):
    """
    Shuffle the elements within a batch using the same permutation across different datasets.
    
    Parameters:
    - batch: A batch from the dataset.
    - seed: Random seed for the shuffle operation.
    
    Returns:
    - Shuffled batch with the same permutation applied.
    """
    # Generate a permutation of indices based on the batch size
    batch_size = tf.shape(batch)[0]
    indices = tf.random.shuffle(tf.range(batch_size), seed=seed)
    
    # Apply the same permutation to the batch
    return tf.gather(batch, indices, axis=0)

def shuffle_batches(x_dataset, y_dataset, target_dataset, seed):
    """
    Shuffle each batch within the datasets while keeping the pairing between them.
    
    Parameters:
    - x_dataset: The first dataset.
    - y_dataset: The second dataset, paired with the first dataset.
    - target_dataset: The target dataset, also paired with the first dataset.
    - seed: Random seed for the shuffle operation.
    
    Returns:
    - x_dataset_shuffled: Shuffled version of the first dataset.
    - y_dataset_shuffled: Shuffled version of the second dataset, maintaining pairing with the first.
    - target_dataset_shuffled: Shuffled version of the target dataset, maintaining pairing with the first.
    """
    # Shuffle each batch within the datasets
    x_dataset_shuffled = x_dataset.map(lambda batch: shuffle_batch_elements(batch, seed))
    y_dataset_shuffled = y_dataset.map(lambda batch: shuffle_batch_elements(batch, seed))
    target_dataset_shuffled = target_dataset.map(lambda batch: shuffle_batch_elements(batch, seed))
    
    return x_dataset_shuffled, y_dataset_shuffled, target_dataset_shuffled


def zip_train_datasets(x_dataset, y_dataset, targets):
    # Assuming `x_dataset` corresponds to `audio_input_in` and `y_dataset` to `audio_input_ref`
    # This function should return a dataset that yields ((input1, input2), target) tuples
    return tf.data.Dataset.zip(((x_dataset, y_dataset), targets))

def zip_train_datasets_new(x_data, y_data=None, targets=None):
    # If x_data is not already a tf.data.Dataset, convert it to one
    if not isinstance(x_data, tf.data.Dataset):
        x_data = tf.data.Dataset.from_tensor_slices([x_data])

    # Do the same for y_data and targets if they are provided
    if y_data is not None and not isinstance(y_data, tf.data.Dataset):
        y_data = tf.data.Dataset.from_tensor_slices([y_data])
    
    if targets is not None and not isinstance(targets, tf.data.Dataset):
        targets = tf.data.Dataset.from_tensor_slices([targets])

    # Now zip the datasets
    if y_data is not None and targets is not None:
        return tf.data.Dataset.zip(((x_data, y_data), targets))
    elif y_data is not None:
        return tf.data.Dataset.zip((x_data, y_data))
    else:
        return x_data

styles = ['neutral', 'broadcast', 'telephone', 'bright', 'warm']
def strip_style_suffix(filename):
    # Remove the style suffix from the filename if present
    for style in styles:
        if f"_{style}" in filename:
            return filename.split(f"_{style}")[0]  # Return the part before the style
    return filename  # Return the original filename if no style suffix is found

def prepare_test_dataset(x_dir, y_dir, no_of_testfiles=None):
    AUTOTUNE = tf.data.AUTOTUNE  # Use tf.data.AUTOTUNE
    # Get file paths

    #assert len(x_dir) == len(y_dir), "Datasets have different sizes"
    # Assuming 'get_file_paths' returns a list of file paths from a directory
    x_files = get_file_paths(x_dir)
    y_files = get_file_paths(y_dir)

    # If 'no_of_testfiles' is not provided, set it to the minimum of the lengths of x_files and y_files
    if no_of_testfiles is None:
        no_of_testfiles = min(len(x_files), len(y_files))
    x_files, y_files = x_files[:no_of_testfiles], y_files[:no_of_testfiles]  

    x_files, y_files = find_file_pairs(x_files, y_files)

    for x_path, y_path in zip(x_files, y_files):
        x_file_name = x_path.split('/')[-1].split('_IR.wav')[0]
        y_file_name = y_path.split('/')[-1].split('_IR.wav')[0]

        # Strip the style suffix from filenames before comparison
        x_file_name_base = strip_style_suffix(x_file_name)
        y_file_name_base = strip_style_suffix(y_file_name)
        assert x_file_name_base == y_file_name_base, f"Mismatch found (not checking suffixes): {x_file_name} != {y_file_name}"

    # Create datasets and apply preprocessing
    x_dataset = tf.data.Dataset.from_tensor_slices(x_files).map(load_and_preprocess, num_parallel_calls=AUTOTUNE)
    y_dataset = tf.data.Dataset.from_tensor_slices(y_files).map(load_and_preprocess, num_parallel_calls=AUTOTUNE)

    x_dataset = x_dataset.map(TFpeak_normalize, num_parallel_calls=AUTOTUNE)
    y_dataset = y_dataset.map(TFpeak_normalize, num_parallel_calls=AUTOTUNE)

    #x_dataset, y_dataset = shuffle_paired_datasets(x_dataset, y_dataset, buffer_size=10000) #### ACHTUNG BUFFERSIZE BELIEBEIG GEWÄHLT

    # Splitting and rejoining 
    xi_a, xi_b = split_dataset(x_dataset)
    xr_a, xr_b = split_dataset(y_dataset)

    x_dataset = xi_a #input
    y_dataset = xr_b #referenz
    target_dataset = xr_a #target

    # Zip the datasets together
    test_dataset = zip_train_datasets(x_dataset, y_dataset, target_dataset)

    return test_dataset.batch(1)  # assuming you want to run inference file by file

# Function to compute and plot STFT
# Adjust the plotting function to accept an axis parameter
def plot_stft_spectra(signal, sr=24000, title="STFT-Spectrogram", ax=None):
    # Compute the STFT
    stft = librosa.stft(signal.flatten(), n_fft=2048, hop_length=512)
    # Convert to dB
    db_stft = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    
    if ax is None:
        ax = plt.gca()  # Get current axis if none is provided
    
    img = librosa.display.specshow(db_stft, sr=sr, x_axis='time', y_axis='log', hop_length=512, ax=ax)
    ax.set_title(title)
    # Use ax.figure to access the figure associated with the ax
    ax.figure.colorbar(img, ax=ax, format='%+2.0f dB')
    return stft

def split_dataset(x_dataset):
    # Define a function to split each audio file tensor in the dataset
    def split_audio(audio):
        # Determine the midpoint of the audio file tensor
        midpoint = tf.shape(audio)[0] // 2
        
        # Split the audio tensor into two halves
        audio_a = audio[:midpoint]
        audio_b = audio[midpoint:]
        
        # Return the two halves
        return audio_a, audio_b

    # Apply the split function to each element in the dataset
    # Then, use flat_map to flatten the dataset structure
    split_dataset = x_dataset.map(split_audio, num_parallel_calls=tf.data.AUTOTUNE)

    # Extract xi_a and xi_b by separately mapping each half
    xi_a = split_dataset.map(lambda x, _: x)
    xi_b = split_dataset.map(lambda _, y: y)
    
    return xi_a, xi_b

def peak_normalize(audio_signals, target_level_db=0):
    normalized_signals = []
    for signal in audio_signals:
        max_amplitude = np.max(np.abs(signal))
        target_amplitude = 10 ** (target_level_db / 20)  # Convert dB to amplitude
        normalization_factor = target_amplitude / max_amplitude
        normalized_signal = signal * normalization_factor
        normalized_signals.append(normalized_signal)
    return np.array(normalized_signals)

def TFpeak_normalize(audio):
    max_val = tf.reduce_max(tf.abs(audio))
    # Avoid division by zero
    max_val = tf.maximum(max_val, 1e-9)
    normalized_audio = audio / max_val
    return normalized_audio




def plot_waveform(signal, sr, title, ax):
    """Plots the audio signal over time.

    Args:
        signal: The audio signal to plot.
        sr: The sampling rate of the signal.
        title: The title of the plot.
        ax: The Matplotlib axis to plot on.
    """
    # Plot audio waveform
    librosa.display.waveshow(signal, sr=sr, ax=ax, color='gray')
    ax.set_title(title + ' Waveform')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylim(-2, 2)  # Setting the y-axis range to -2 to 2






def plot_stft_librosa(signal, sr, title, ax, hop_length=512):
    """Plots the Short-Time Fourier Transform (STFT) of an audio signal using Librosa.

    Args:
        signal: The audio signal.
        sr: The sampling rate of the signal.
        title: The title of the plot.
        ax: The Matplotlib axis object to plot on.
        hop_length: The hop length parameter for the STFT.

    Returns:
        The image object representing the plot.
    """
    # Compute the STFT of the signal
    D = librosa.stft(signal, hop_length=hop_length)
    # Convert the STFT to decibels
    dB = librosa.amplitude_to_db(np.abs(D), ref=1)

    # Use librosa's specshow function to plot the STFT in decibels with a logarithmic frequency axis
    img = librosa.display.specshow(dB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log', ax=ax)
    ax.set_title(title)
    return img
