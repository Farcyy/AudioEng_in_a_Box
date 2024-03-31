
import sounddevice as sd
import time
import os
import numpy as np
import pdb
import librosa
from scipy.signal import stft
import librosa.display
import tensorflow as tf
from helpers.utils import get_file_paths, zip_train_datasets, split_dataset, TFpeak_normalize, strip_style_suffix, find_file_pairs, shuffle_batches
from helpers.process import load_and_preprocess
from make_network.callbacks import LRTensorBoard, get_run_logdir, SaveMelSpectrogramCallback
#from make_network.callbacks import PrintControllerOutputCallback

# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)



AUTOTUNE = tf.data.AUTOTUNE  # Use tf.data.AUTOTUNE



def get_train_val_Data(x_dir, y_dir, config):

    np.random.seed(config['random_seed'])
    tf.random.set_seed(config['random_seed'])

    x_files = get_file_paths(x_dir)
    y_files = get_file_paths(y_dir)


    # limit number of tracks if set true 
    if config['limit_tracks']:
        x_files = x_files[:config['max_tracks']]
        y_files = y_files[:config['max_tracks']]

    x_files, y_files = find_file_pairs(x_files, y_files)

    for x_path, y_path in zip(x_files, y_files):
        x_file_name = x_path.split('/')[-1].split('_IR.wav')[0]
        y_file_name = y_path.split('/')[-1].split('_IR.wav')[0]

        # Strip the style suffix from filenames before comparison
        x_file_name_base = strip_style_suffix(x_file_name)
        y_file_name_base = strip_style_suffix(y_file_name)
        assert x_file_name_base == y_file_name_base, f"Mismatch found (not checking suffixes): {x_file_name} != {y_file_name}"

    # Assuming x_files is a list of file paths
    print('<--- x_files --->')
    # Assuming x_files is a list with more than 10 elements
    print(x_files[:10])

    # Create datasets and apply preprocessing
    x_dataset = tf.data.Dataset.from_tensor_slices(x_files).map(load_and_preprocess, num_parallel_calls=AUTOTUNE)
    y_dataset = tf.data.Dataset.from_tensor_slices(y_files).map(load_and_preprocess, num_parallel_calls=AUTOTUNE)


    # Normalize Datapackages
    x_dataset = x_dataset.map(TFpeak_normalize, num_parallel_calls=AUTOTUNE)
    y_dataset = y_dataset.map(TFpeak_normalize, num_parallel_calls=AUTOTUNE)

    # Splitting and rejoining
    x_a, x_b = split_dataset(x_dataset)
    y_a, y_b = split_dataset(y_dataset)

    x_dataset_split = x_a.concatenate(x_b) #input
    y_dataset_split = y_b.concatenate(y_a) #referenz
    target_dataset_split = y_a.concatenate(y_b) #target
    
    # Calculate dataset sizes for splitting and batches
    total_size = len(x_files*2)
    train_size = int(total_size * config['train_ratio'])
    test_size = int(total_size * config['test_ratio'])
    val_size = total_size - train_size - test_size



    # Prepare the training and validation datasets with caching and batching
    x_train_dataset = x_dataset_split.take(train_size).cache().batch(config['batch_size']).repeat().prefetch(AUTOTUNE)
    x_val_dataset = x_dataset_split.skip(train_size).take(val_size).cache().batch(config['batch_size']).repeat().prefetch(AUTOTUNE)

    y_train_dataset = y_dataset_split.take(train_size).cache().batch(config['batch_size']).repeat().prefetch(AUTOTUNE)
    y_val_dataset = y_dataset_split.skip(train_size).take(val_size).cache().batch(config['batch_size']).repeat().prefetch(AUTOTUNE)

    target_train_dataset = target_dataset_split.take(train_size).cache().batch(config['batch_size']).repeat().prefetch(AUTOTUNE)
    target_val_dataset = target_dataset_split.skip(train_size).take(val_size).cache().batch(config['batch_size']).repeat().prefetch(AUTOTUNE)

    # Shuffle datasets while keeping paired structure
    seed = config['random_seed']
    x_train_dataset_shuffled, y_train_dataset_shuffled, target_train_dataset_shuffled = shuffle_batches(x_train_dataset, y_train_dataset, target_train_dataset, seed)
    x_val_dataset_shuffled, y_val_dataset_shuffled, target_val_dataset_shuffled = shuffle_batches(x_val_dataset, y_val_dataset, target_val_dataset, seed)


    # Configure Datasets 
    train_dataset = zip_train_datasets(x_train_dataset_shuffled, y_train_dataset_shuffled, target_train_dataset_shuffled) 
    val_dataset = zip_train_datasets(x_val_dataset_shuffled, y_val_dataset_shuffled, target_val_dataset_shuffled) 

    # Calculate steps per epoch for training and validation
    # Recalculate steps_per_epoch and validation_steps
    steps_per_epoch = max(1, train_size // config['batch_size'])
    validation_steps = max(1, val_size // config['batch_size'])
    print(f'Steps per epoch: {steps_per_epoch}, Validation steps: {validation_steps}')
    print(f'Total size: {total_size}, Train size: {train_size}, Validation size: {val_size}')


    return train_dataset, val_dataset, steps_per_epoch, validation_steps

def get_input_shape(train_dataset): 
    ##### input shape:
    for audio_tensor in train_dataset.take(1):
        input_shape = audio_tensor[0][0].shape

    for inputs, targets in train_dataset.take(1):
        # Assuming inputs is a tuple of (x_train, y_train) and targets is y_train_targets
        x_train_shape = inputs[0].shape
        y_train_shape = inputs[1].shape
        y_train_targets_shape = targets.shape
        
        print(f"x_train shape: {x_train_shape}")
        print(f"y_train shape: {y_train_shape}")
        print(f"y_train_targets shape: {y_train_targets_shape}")


    #Printing info
    print('<-> printing input shape here <->')
    print(input_shape)

    # Exclude the batch size from the input_shape
    # We do this by taking all dimensions except the first one
    return input_shape[1:]



