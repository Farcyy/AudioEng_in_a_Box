# Defining your configuration for Trainings
config = {
    'frame_length': 512,   
    'batch_size': 64,      
    'random_seed': 42,
    'train_ratio': 0.8,
    'test_ratio' : 0.1,
    'val_ratio' : 0.1,
    'buffer_size' : 1500,   ## for now
    'std_sample_rate' : 24000,
    'fft_length' : 512,  # Increase for better frequency resolution
    'hop_length' : 256, # Common choice for 16kHz sample rate, adjust as needed
    'n_mels' : 256, # Decrease if you get empty filter warnings, or to reduce computation
    'lr' : 0.0019004,
    'lr_decay': 0.97,
    'lr_scheduler': False,
    'optimizer': "sdg",
    'limit_tracks' : False,
    'max_tracks' : 250,
    'no_epochs' : 150,
    'system' : "Apple_Silicon", # Intel/AMD
    'l1_loss_ratio' : 2.7916,
    'hidden_dim': 32, 
    'num_control_params': 24
    }
