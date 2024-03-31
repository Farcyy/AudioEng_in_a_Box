import numpy as np
import tensorflow as tf 
import own_config as config
import scipy

#Multi-Resolutional STFT Loss
#Orientiert an https://github.com/TensorSpeech/TensorflowTTS/blob/master/tensorflow_tts/losses/stft.py

def l1_loss(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))

def l2_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

class TFSpectralConvergence(tf.keras.layers.Layer):
    """Spectral convergence loss."""

    def __init__(self):
        """Initialize."""
        super().__init__()

    def call(self, y_mag, x_mag):
        """Calculate forward propagation.
        Args:
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Spectral convergence loss value.
        """
        return tf.norm(y_mag - x_mag, ord="fro", axis=(-2, -1)) / tf.norm(
            y_mag, ord="fro", axis=(-2, -1)
        )


class TFLogSTFTMagnitude(tf.keras.layers.Layer):
    """Log STFT magnitude loss module."""

    def __init__(self):
        """Initialize."""
        super().__init__()

    def call(self, y_mag, x_mag):
        """Calculate forward propagation.
        Args:
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Spectral convergence loss value.
        """
        return tf.abs(tf.math.log(y_mag) - tf.math.log(x_mag))


class TFSTFT(tf.keras.layers.Layer):
    """STFT loss module."""

    def __init__(self, frame_length=config.config["frame_length"], frame_step=config.config["hop_length"], fft_length=config.config["fft_length"]):
        """Initialize."""
        super().__init__()
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.fft_length = fft_length
        self.spectral_convergenge_loss = TFSpectralConvergence()
        self.log_stft_magnitude_loss = TFLogSTFTMagnitude()

    def call(self, y, x):
        """Calculate forward propagation.
        Args:
            y (Tensor): Groundtruth signal (B, T).
            x (Tensor): Predicted signal (B, T).
        Returns:
            Tensor: Spectral convergence loss value (pre-reduce).
            Tensor: Log STFT magnitude loss value (pre-reduce).
        """
        x_mag = tf.abs(
            tf.signal.stft(
                signals=x,
                frame_length=self.frame_length,
                frame_step=self.frame_step,
                fft_length=self.fft_length,
            )
        )
        y_mag = tf.abs(
            tf.signal.stft(
                signals=y,
                frame_length=self.frame_length,
                frame_step=self.frame_step,
                fft_length=self.fft_length,
            )
        )

        # add small number to prevent nan value.
        # compatible with pytorch version.
        x_mag = tf.clip_by_value(tf.math.sqrt(x_mag ** 2 + 1e-7), 1e-7, 1e3)
        y_mag = tf.clip_by_value(tf.math.sqrt(y_mag ** 2 + 1e-7), 1e-7, 1e3)

        sc_loss = self.spectral_convergenge_loss(y_mag, x_mag)
        mag_loss = self.log_stft_magnitude_loss(y_mag, x_mag)

        return sc_loss, mag_loss
    
class TFMultiResolutionSTFT(tf.keras.layers.Layer):
    """Multi resolution STFT loss module."""

    def __init__(
        self,
        fft_lengths=[1024, 2048, 512],
        frame_lengths=[600, 1200, 240],
        frame_steps=[120, 240, 50],
    ):
        """Initialize Multi resolution STFT loss module.
        Args:
            frame_lengths (list): List of FFT sizes.
            frame_steps (list): List of hop sizes.
            fft_lengths (list): List of window lengths.
        """
        super().__init__()
        assert len(frame_lengths) == len(frame_steps) == len(fft_lengths)
        self.stft_losses = []
        for frame_length, frame_step, fft_length in zip(
            frame_lengths, frame_steps, fft_lengths
        ):
            self.stft_losses.append(TFSTFT(frame_length, frame_step, fft_length))

    def call(self, y, x):
        """Calculate forward propagation.
        Args:
            y (Tensor): Groundtruth signal (B, T).
            x (Tensor): Predicted signal (B, T).
        Returns:
            Tensor: Multi resolution spectral convergence loss value.
            Tensor: Multi resolution log STFT magnitude loss value.
        """
        sc_loss = 0.0
        mag_loss = 0.0
        for f in self.stft_losses:
            sc_l, mag_l = f(y, x)
            sc_loss += tf.reduce_mean(sc_l, axis=list(range(1, len(sc_l.shape))))
            mag_loss += tf.reduce_mean(mag_l, axis=list(range(1, len(mag_l.shape))))

        sc_loss /= len(self.stft_losses)
        mag_loss /= len(self.stft_losses)

        return sc_loss, mag_loss
    
#Multi-Resolutional Mel Loss

class TFMelSpectralConvergence(tf.keras.layers.Layer):
    """Mel spectral convergence loss."""

    def __init__(self):
        """Initialize."""
        super().__init__()

    def call(self, y, x):
        """Calculate forward propagation.
        Args:
            y (Tensor): Magnitude mel spectrogram of groundtruth signal (B, #frames, #mel_bins).
            x (Tensor): Magnitude mel spectrogram of predicted signal (B, #frames, #mel_bins).
        Returns:
            Tensor: Mel spectral convergence loss value.
        """
        return tf.norm(y - x, ord="fro", axis=(-2, -1)) / tf.norm(y, ord="fro", axis=(-2, -1))

class TFLogMelMagnitude(tf.keras.layers.Layer):
    """Log mel magnitude loss module."""

    def __init__(self):
        """Initialize."""
        super().__init__()

    def call(self, y, x):
        """Calculate forward propagation.
        Args:
            y (Tensor): Magnitude mel spectrogram of groundtruth signal (B, #frames, #mel_bins).
            x (Tensor): Magnitude mel spectrogram of predicted signal (B, #frames, #mel_bins).
        Returns:
            Tensor: Log mel magnitude loss value.
        """
        return tf.abs(tf.math.log(y + 1e-7) - tf.math.log(x + 1e-7))   

class TFMelSTFT(tf.keras.layers.Layer):
    """Mel STFT loss module."""

    def __init__(self, num_mel_bins=config.config["n_mels"], sample_rate=24000, fmin=20, fmax=12000,
                 frame_length=config.config["frame_length"], frame_step=config.config["hop_length"], fft_length=config.config["fft_length"]):
        """Initialize."""
        super().__init__()
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.fft_length = fft_length
        self.num_mel_bins = num_mel_bins
        self.sample_rate = sample_rate
        self.fmin = fmin
        self.fmax = fmax
        self.spectral_convergenge_loss = TFMelSpectralConvergence()
        self.log_mel_magnitude_loss = TFLogMelMagnitude()

    def call(self, y, x):
        """Calculate forward propagation.
        Args:
            y (Tensor): Groundtruth signal (B, T).
            x (Tensor): Predicted signal (B, T).
        Returns:
            Tensor: Mel spectral convergence loss value (pre-reduce).
            Tensor: Log mel magnitude loss value (pre-reduce).
        """
        # Compute STFT
        y_stft = tf.signal.stft(signals=y, frame_length=self.frame_length, frame_step=self.frame_step, fft_length=self.fft_length)
        x_stft = tf.signal.stft(signals=x, frame_length=self.frame_length, frame_step=self.frame_step, fft_length=self.fft_length)

        # Convert to magnitude
        y_mag = tf.abs(y_stft)
        x_mag = tf.abs(x_stft)

        # Compute mel spectrograms
        mel_matrix = tf.signal.linear_to_mel_weight_matrix(num_mel_bins=self.num_mel_bins, num_spectrogram_bins=self.fft_length // 2 + 1, sample_rate=self.sample_rate, lower_edge_hertz=self.fmin, upper_edge_hertz=self.fmax)
        y_mel = tf.tensordot(y_mag, mel_matrix, 1)
        x_mel = tf.tensordot(x_mag, mel_matrix, 1)

        # Normalize for numerical stability
        y_mel = tf.clip_by_value(tf.math.sqrt(y_mel ** 2 + 1e-7), 1e-7, 1e3)
        x_mel = tf.clip_by_value(tf.math.sqrt(x_mel ** 2 + 1e-7), 1e-7, 1e3)

        # Calculate losses
        sc_loss = self.spectral_convergenge_loss(y_mel, x_mel)
        mag_loss = self.log_mel_magnitude_loss(y_mel, x_mel)

        return sc_loss, mag_loss

class TFMultiResolutionMel(tf.keras.layers.Layer):
    """Multi resolution Mel spectrogram loss module."""

    def __init__(
        self,
        num_mel_bins=config.config["n_mels"],
        sample_rate=24000,
        fmin=20,
        fmax=12000,
        fft_lengths=[1024, 2048, 512],
        frame_lengths=[600, 1200, 240],
        frame_steps=[120, 240, 50],
    ):
        """Initialize Multi resolution Mel spectrogram loss module.
        Args:
            num_mel_bins (int): Number of Mel bins.
            sample_rate (int): Sample rate of the audio signal.
            fmin (float): Minimum frequency to include in the Mel scale.
            fmax (float): Maximum frequency to include in the Mel scale.
            frame_lengths (list): List of frame lengths.
            frame_steps (list): List of hop sizes.
            fft_lengths (list): List of FFT sizes.
        """
        super().__init__()
        assert len(frame_lengths) == len(frame_steps) == len(fft_lengths)
        self.mel_losses = []
        for frame_length, frame_step, fft_length in zip(frame_lengths, frame_steps, fft_lengths):
            self.mel_losses.append(
                TFMelSTFT(num_mel_bins=num_mel_bins, sample_rate=sample_rate, fmin=fmin, fmax=fmax,
                          frame_length=frame_length, frame_step=frame_step, fft_length=fft_length)
            )

    def call(self, y, x):
        """Calculate forward propagation.
        Args:
            y (Tensor): Groundtruth signal (B, T).
            x (Tensor): Predicted signal (B, T).
        Returns:
            Tensor: Multi resolution Mel spectral convergence loss value.
            Tensor: Multi resolution log Mel magnitude loss value.
        """
        sc_loss = 0.0
        mag_loss = 0.0
        for mel_loss in self.mel_losses:
            sc_l, mag_l = mel_loss(y, x)
            sc_loss += tf.reduce_mean(sc_l, axis=list(range(1, len(sc_l.shape))))
            mag_loss += tf.reduce_mean(mag_l, axis=list(range(1, len(mag_l.shape))))

        sc_loss /= len(self.mel_losses)
        mag_loss /= len(self.mel_losses)

        return sc_loss, mag_loss
    
class MEL_MR_L1_Loss(tf.keras.losses.Loss):
    def __init__(self, l1_ratio=1.0):
        super().__init__()
        self.l1_ratio = l1_ratio

    def call(self, y_true, y_pred):
        TF_MR_STFT = TFMultiResolutionSTFT()
        TF_MR_MEL = TFMultiResolutionMel()
        _, mel_loss = TF_MR_MEL(y_true, y_pred)

        # Compute L1 loss
        l1_loss = tf.reduce_mean(tf.abs(y_true - y_pred))

        # Combine the losses using the hyperparameter
        total_loss = mel_loss + self.l1_ratio * l1_loss
        return total_loss