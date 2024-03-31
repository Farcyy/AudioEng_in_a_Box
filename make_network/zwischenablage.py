 #x_processed = autodiff_layer(encoder_input_1,controller_output) ##<-

    inputs = [encoder_input_1, controller_output]

    # Call the instantiated layer with the inputs
    x_processed = peq_Layer(inputs)

# Stack the processed tensors into a single tensor
#x_processed = tf.stack(processed_tensors)

#output = autodiff_layer(encoder_input_1,controller_output)
    output = peq_Layer(inputs)


    model = keras.Model(inputs=[encoder_input_1,encoder_input_2], outputs=[x_processed,encoder_input_2], name="system_with_spsas")

        dafx_output = PEQLayer()([audio_time, params])

  
        flat = keras.layers.Flatten()(dafx_output)

    # Compute the model
model = keras.models.Model(inputs=[audio_time], outputs=flat, name="full_model")


def load_and_preprocess_dataOLD(data_dir, classes, target_shape=(128, 128)):
    data = []
    labels = []

    for i, class_name in enumerate(classes):
       # class_dir = os.path.join(data_dir, class_name) Klassen in x_dir und y_dir oben definiert

        for filename in os.listdir(data_dir):
            if filename.endswith('.wav'):
                file_path = os.path.join(data_dir, filename)
                audio_data, sample_rate = librosa.load(file_path, sr=None)
                # Perform preprocessing (e.g., convert to Mel spectrogram and resize)
                mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
                mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
                data.append(mel_spectrogram)
                labels.append(i)

    return np.array(data), np.array(labels)


    #NEW!



# This function now takes a list of file paths and a single class label
def load_and_preprocess_data(file_paths, class_label, target_shape=(128, 128)):
    data = []
    labels = []
    waveforms =[]
    # Process files for the given class
    for file_path in file_paths:
        if file_path.endswith('.wav'):
            audio_data, sample_rate = librosa.load(file_path, sr=None)
            mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
            mel_spectrogram = resize(mel_spectrogram, target_shape)
            data.append(np.expand_dims(mel_spectrogram, axis=-1))  # Add channel dimension at the end
            labels.append(class_label)
            audio_data=tfio.audio.AudioIOTensor(file_path)
            waveforms.append(audio_data.to_tensor())  # Keep the audio data for later and typecast to tensor for keras

    return np.array(data), np.array(labels), waveforms