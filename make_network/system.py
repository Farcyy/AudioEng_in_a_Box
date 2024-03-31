
import tensorflow as tf
from make_network import controller, encoder, spsa_layer

def get_model(config, input_shape):
    # Create a Keras Input layer with the specified input shape and data type => Funktioniert
    audio_input_in = tf.keras.Input(shape=input_shape, dtype=tf.float32, name='audio_input_in')
    audio_input_ref = tf.keras.Input(shape=input_shape, dtype=tf.float32, name='audio_input_ref')
    
    # Assume encoders and controller_layer are functions returning Keras layers or models configured based on `config` => Funktioniert
    encoder_model = encoder.build_encoders(config, input_shape, input_shape)

    #encoder_output = encoder_model([audio_input_in, audio_input_ref])
    encoder_output = encoder_model([audio_input_in, audio_input_ref]) ### see encoder.py -> for merged_outputs_mit_mel

    #Controller 
    controller_layer = controller.make_controller_layer(config)
    hidden = controller_layer(encoder_output)  # Pass the encoder output through the controller
    hidden = tf.keras.layers.Lambda(lambda x: tf.clip_by_value(x, clip_value_min=0, clip_value_max=1))(hidden)
    
    # Build analyzer model for params retrieve
    analyzer_model = tf.keras.models.Model(inputs=[audio_input_in, audio_input_ref], 
                                           outputs=[hidden], 
                                           name="analyzer_model")
    hidden_params = analyzer_model([audio_input_in, audio_input_ref])

    dafx_output = spsa_layer.DSPLayer()([audio_input_in, hidden_params])

    # Compute the model
    full_model = tf.keras.models.Model(inputs=[audio_input_in, audio_input_ref], outputs=dafx_output, name="full_model")

    full_model.summary()  # Print a summary to verify the model's architecture

    return full_model, analyzer_model

    
