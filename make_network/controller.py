import tensorflow as tf

#from keras.applications import EfficientNetV2L



def make_controller_layer(con_in):

    mlp_in_dim = (int(con_in['n_mels']/4-2), int(con_in['n_mels']/8))
    hidden_dim = int(con_in['hidden_dim'])
    num_control_params = con_in['num_control_params']
    #num_control_params = 6 # for now!

    controller = tf.keras.models.Sequential([
        tf.keras.layers.Dense(hidden_dim, input_shape=(mlp_in_dim), activation=None),
        tf.keras.layers.Flatten(),
        tf.keras.layers.LeakyReLU(0.01),
        tf.keras.layers.Dense(hidden_dim, activation=None),
        tf.keras.layers.LeakyReLU(0.01),
        tf.keras.layers.Dropout(.1),
        tf.keras.layers.Dense(num_control_params, activation='sigmoid', name='Controller')  # Output layer with sigmoid activation
    ])

    return controller


