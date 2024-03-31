

import tensorflow as tf
from pathlib import Path
from time import strftime
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import numpy as np

def get_run_logdir(root_logdir="my_logs"):
    return Path(root_logdir) / strftime("run_%Y_%m_%d_%H_%M_%S")



class SaveMelSpectrogramCallback(tf.keras.callbacks.Callback):
    def __init__(self, save_path, encoder, dataset, sample_rate=24000, num_samples=1, target_shape=(128, 128)):
        super().__init__()
        self.save_path = save_path
        self.encoder = encoder
        self.dataset = dataset
        self.sample_rate = sample_rate
        self.num_samples = num_samples
        self.target_shape = target_shape
        # Calculate hop_length to achieve the target shape, assuming n_fft=2048 and duration matches 120000 samples
        self.hop_length = 120000 // (target_shape[1] - 1)  # Adjust based on the specific requirements

    def on_epoch_end(self, epoch, logs=None):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        for inputs, _ in self.dataset.take(self.num_samples):
            outputs = self.encoder.predict(inputs)

            if len(outputs) < 3:
                print("Unexpected number of outputs from model. Expected at least 3, got:", len(outputs))
                continue

            x_raw_audio, y_raw_audio = outputs[-2], outputs[-1]

            # You might need to adjust the length of x_raw_audio and y_raw_audio to match 120000 samples if it's not consistent
            try:
                x_mel_output = librosa.feature.melspectrogram(y=x_raw_audio[:120000], sr=self.sample_rate, n_fft=2048, hop_length=self.hop_length, n_mels=self.target_shape[0])
                y_mel_output = librosa.feature.melspectrogram(y=y_raw_audio[:120000], sr=self.sample_rate, n_fft=2048, hop_length=self.hop_length, n_mels=self.target_shape[0])

                '''print(f"Shape of x_raw_audio: {x_raw_audio.shape}")
                print(f"Shape of y_raw_audio: {y_raw_audio.shape}")
                print(f"Shape of x_mel_output: {x_mel_output.shape}")
                print(f"Shape of y_mel_output: {y_mel_output.shape}")'''

                if x_mel_output.shape != self.target_shape or y_mel_output.shape != self.target_shape:
                    print(f"Warning: Mel output shape does not match target shape {self.target_shape}.")
                for idx, mel_output in enumerate([x_mel_output, y_mel_output], start=1):
                    mel_spectrogram = np.squeeze(mel_output)
                    if mel_spectrogram.ndim != 2:
                        print(f"Unexpected tensor shape for Mel output {idx}: {mel_spectrogram.shape}, expected 2D. Skipping.")
                        continue

                    plt.figure(figsize=(10, 4))
                    S_dB = librosa.power_to_db(mel_spectrogram, ref=np.max)
                    librosa.display.specshow(S_dB, sr=self.sample_rate, hop_length=self.hop_length, y_axis='mel', fmax=self.sample_rate / 2)
                    plt.colorbar(format='%+2.0f dB')
                    plt.title(f'Mel Spectrogram (epoch {epoch+1}, Mel output {idx})')
                    plt.tight_layout()
                    print('plots here; shape of mel:')
                    print(mel_spectrogram.shape)
                    filename = os.path.join(self.save_path, f"epoch_{epoch+1}_mel_output_{idx}.png")
                    plt.savefig(filename)
                    plt.close()
                    print(f"Saved Mel spectrogram to {filename}")

            except Exception as e:
                print(f"Error during Mel spectrogram generation or saving: {e}")
                # Continue with your code to process and save the Mel spectrograms...
            
class PrintControllerOutputCallback(tf.keras.callbacks.Callback):
    def on_batch_end(self, batch, logs=None):
        # Get the input tensors to the controller layer
        controller_inputs = self.model.controller_layer.input

        # Run the controller layer to get the outputs
        controller_output = tf.keras.backend.function(controller_inputs, self.model.controller_layer.output)

        # Fetch input data from the batch
        x_batch = self.model.input
        # Use the input data to compute the controller output
        output_values = controller_output(x_batch)

        print("Controller output:", output_values)

class LRTensorBoard(tf.keras.callbacks.TensorBoard):
    def __init__(self, **kwargs):
        super(LRTensorBoard, self).__init__(**kwargs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        lr = self.model.optimizer.learning_rate
        if callable(lr):
            lr = lr(epoch)
        logs.update({'lr': tf.keras.backend.get_value(lr)})
        super().on_epoch_end(epoch, logs)

