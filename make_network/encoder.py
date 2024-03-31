import tensorflow as tf
from make_network import mel_layer  # If mel_layer is a class or function

##### achtung!
#from make_network import mel_layer_with_plots as mel_layer


# Assuming MelSpectrogramLayer is already defined somewhere in your code.
# from somewhere import MelSpectrogramLayer

def build_encoders(config,input_shape_in, input_shape_ref):
   
    # Define the input layers
    encoder_input_in = tf.keras.Input(input_shape_in, name="input_data_channel")
    encoder_input_ref = tf.keras.Input(input_shape_ref, name="referenz_data_channel")
    
    in_mel = mel_layer.MelSpectrogramLayer(
        sample_rate=config['std_sample_rate'],
        n_fft=config['fft_length'],
        hop_length=config['hop_length'],
        window=config["frame_length"],
        n_mels=config['n_mels'],
        name='MelLayer_in'
    )

    ref_mel = mel_layer.MelSpectrogramLayer(
        sample_rate=config['std_sample_rate'],
        n_fft=config['fft_length'],
        hop_length=config['hop_length'],
        n_mels=config['n_mels'],
        window=config["frame_length"],
        name='MelLayer_ref'
    )

    # Input encoder
    mel_spec_x = in_mel(encoder_input_in)
    x = tf.keras.layers.Conv2D(16, kernel_size=2, activation="relu")(mel_spec_x)
    x = tf.keras.layers.Conv2D(32, kernel_size=2, activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = tf.keras.layers.Conv2D(32, kernel_size=2, activation="relu")(x)
    x = tf.keras.layers.Conv2D(16, kernel_size=2, activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    #x = tf.keras.layers.GlobalMaxPooling2D()(x)

    x = tf.keras.layers.BatchNormalization()(x)#Cons
    x = tf.keras.backend.mean(x, axis=1, keepdims=False)
    encoder_output_in = x

    #encoder_1 = tf.keras.Model(encoder_input_1, encoder_output_1, name="encoder_1")

    # Reference encoder
    mel_spec_y = ref_mel(encoder_input_ref)

    #y = tf.keras.layers.Dense(num_basis, activation='linear')(mel_spec_y) #Cons
    #y = tf.keras.layers.Reshape((mel_spec_y,1))(mel_spec_y,1) #Cons
    y = tf.keras.layers.Conv2D(16, kernel_size=2, activation="relu")(mel_spec_y)
    y = tf.keras.layers.Conv2D(32, kernel_size=2, activation="relu")(y)
    y = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(y)

    y = tf.keras.layers.Conv2D(32, kernel_size=2, activation="relu")(y)
    y = tf.keras.layers.Conv2D(16, kernel_size=2, activation="relu")(y)
    y = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(y)
    #y = tf.keras.layers.GlobalMaxPooling2D()(y)

    y = tf.keras.layers.BatchNormalization()(y)#Cons
    y = tf.keras.backend.mean(y, axis=1, keepdims=False)
    encoder_output_ref = y


    #encoder_2 = tf.keras.Model(encoder_input_2, encoder_output_2, name="encoder_2")

    # Merge the outputs from both encoders
    #merged_output = tf.keras.layers.Concatenate()([encoder_1(encoder_input_1), encoder_2(encoder_input_2)])
    merged_output = tf.keras.layers.Concatenate()([encoder_output_in, encoder_output_ref])

    encoder_output = tf.keras.layers.BatchNormalization()(merged_output)

    # Create the final encoder model
    encoder = tf.keras.Model(inputs=[encoder_input_in, encoder_input_ref], outputs=[encoder_output], name="encoder")

    encoder.summary()

    return encoder

# Usage example:
# Assuming 'x_test_dataset' and 'config' have been defined elsewhere in your code.
# encoder_model = build_encoders(x_test_dataset, config)
