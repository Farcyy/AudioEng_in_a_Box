import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
import librosa

# New MelSpectrogram layer => Higher Efficiency 
class MelSpectrogramLayer(tf.keras.layers.Layer):
    def __init__(self, sample_rate=24000, n_fft=256, hop_length=128, n_mels=128, window=None, target_shape=None, **kwargs):
        super().__init__(**kwargs)
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.window = window
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.target_shape = (n_mels, n_mels)

    def call(self, audio, training=False):

        # Convert to mono if the waveform is stereo
        if audio.ndim == 2 and audio.shape[1] == 2:
            audio = np.mean(audio, axis=1)

        spectrogram = tfio.audio.spectrogram(audio, nfft=self.n_fft, window=256, stride=self.hop_length)
        mel_spectrogram = tfio.audio.melscale(spectrogram, rate=24000, mels=self.n_mels, fmin=0, fmax=12000)
        dbscale_mel_spectrogram = tfio.audio.dbscale(mel_spectrogram, top_db=80)

        # Add a channel dimension
        dbscale_mel_spectrogram = tf.expand_dims(dbscale_mel_spectrogram, -1)

        # Resize if target shape is provided
        if self.target_shape is not None:
            dbscale_mel_spectrogram = tf.image.resize(dbscale_mel_spectrogram, self.target_shape)

        return  dbscale_mel_spectrogram
    
    
# Assuming mel_layer is a module containing MelSpectrogramLayer

class SaveMelSpectrogramLayer(tf.keras.layers.Layer):
    def __init__(self, mel_layer_instance, save_path, name=None, **kwargs):
        super(SaveMelSpectrogramLayer, self).__init__(name=name, **kwargs)
        self.mel_layer_instance = mel_layer_instance
        self.save_path = save_path


    def call(self, inputs, training=False):
        # Compute the mel spectrogram
        mel_output = self.mel_layer_instance(inputs)

        # Assuming mel_output shape is [batch_size, n_mels, n_frames, channels]
        for i in range(tf.shape(mel_output)[0]):
            # Generate a filename for each item in the batch
            filename = tf.strings.join([self.save_path, "/item_", tf.strings.as_string(i), "_", tf.strings.as_string(tf.timestamp()), ".tfrecord"])
            # Serialize and save each item individually
            tf.io.write_file(filename, tf.io.serialize_tensor(mel_output[i]))

        return mel_output