
print('   ')
print('                                                  ####################################################################')
print('                                                  #######-------->>>> ***++++  START  +++++*** <<<<-----------########')
print('                                                  ####################################################################')
print('   ')


import time
import os
import numpy as np
import pdb
import librosa
from scipy.signal import stft
import librosa.display
#from metrics import mse
import tensorflow as tf
from keras.optimizers import Adam
from make_network.system import get_model
from make_network.losses import MEL_MR_L1_Loss
from make_network.load_training_data import get_train_val_Data, get_input_shape
from make_network.callbacks import LRTensorBoard, get_run_logdir, SaveMelSpectrogramCallback
from helpers.utils import get_file_paths
#from make_network.callbacks import PrintControllerOutputCallback
import own_config as config
import keras_tuner as kt


np.random.seed(config.config['random_seed'])

# Load file paths 
x_dir = './datasets/test/dirty'
y_dir = './datasets/test/clean'

print(x_dir)
# Get Train Data

train_dataset, val_dataset, steps_per_epoch, validation_steps = get_train_val_Data(x_dir, y_dir, config.config)

total_size = len(get_file_paths(x_dir)*2)
train_size = int(total_size * config.config['train_ratio'])
test_size = int(total_size * config.config['test_ratio'])
val_size = total_size - train_size - test_size

input_shape = get_input_shape(train_dataset)


#with strategy.scope():
# Setting up the learning rate scheduler
initial_learning_rate = config.config['lr']
lr=initial_learning_rate
if (config.config['lr_scheduler'] == "True"):
    lr = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=steps_per_epoch/10,
    decay_rate=config.config['lr_decay'],
    staircase=True)

strategy = tf.distribute.MultiWorkerMirroredStrategy()
with strategy.scope(): 
    # Model training setup
    model, _ = get_model(config.config, input_shape)  

    # Applying the learning rate scheduler to the Adam optimizer 
    if(config.config['system'] == "Apple_Silicon"): 
        if(config.config['optimizer'] == "adam"):optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=lr)
        if(config.config['optimizer'] == "sdg"):optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=lr)
    else:
        adam_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    #Setting up Loss
    mel_mr_l1 = MEL_MR_L1_Loss(l1_ratio=config.config["l1_loss_ratio"]) 


    #Compiling mit MR-Losses
    model.compile(optimizer=optimizer,loss=mel_mr_l1) 

    #Setting up callbacks and tensorboard
    run_logdir = get_run_logdir()
    tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir, profile_batch='5,10')
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(run_logdir, "cp-{epoch:04d}.weights.h5"), save_weights_only=True, verbose=1)

    lr_filepath = os.path.join(run_logdir, "lr-{epoch:04d}.json")
    lr_saving_cb = LRTensorBoard(log_dir=run_logdir)

    #Early Stopping callback
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor='accuracy', 
                                                        min_delta=0, 
                                                        patience=2, 
                                                        verbose=0, 
                                                        mode='auto', 
                                                        baseline=None, 
                                                        restore_best_weights=True)


    #Load Weigth if needed 
    #weights_path = 'my_logs/run_2024_03_22_21_57_05/cp-0001.weights.h5'
    #model.load_weights(weights_path)

    history = model.fit(
    train_dataset,
    epochs=config.config['no_epochs'],
    steps_per_epoch=steps_per_epoch,
    validation_data=val_dataset,
    validation_steps=validation_steps,
    callbacks=[tensorboard_cb, cp_callback, lr_saving_cb]  # Your custom callbacks
    )