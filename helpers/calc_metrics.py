###### größtenteils übernommen aus deepafx/metrics.py


import torch
import auraloss
import resampy
import torchaudio
import numpy as np
from pesq import pesq
import pyloudnorm as pyln
import matplotlib.pyplot as plt
import os
import shutil

def crest_factor(x):
    """Compute the crest factor of waveform."""

    peak, _ = x.abs().max(dim=-1)
    rms = torch.sqrt((x ** 2).mean(dim=-1))

    return 20 * torch.log(peak / rms.clamp(1e-8))


def rms_energy(x):

    rms = torch.sqrt((x ** 2).mean(dim=-1))

    return 20 * torch.log(rms.clamp(1e-8))


def spectral_centroid(x):
    """Compute the crest factor of waveform.

    See: https://gist.github.com/endolith/359724

    """

    spectrum = torch.fft.rfft(x).abs()
    normalized_spectrum = spectrum / spectrum.sum()
    normalized_frequencies = torch.linspace(0, 1, spectrum.shape[-1])
    spectral_centroid = torch.sum(normalized_frequencies * normalized_spectrum)

    return spectral_centroid


def loudness(x, sample_rate):
    """Compute the loudness in dB LUFS of waveform."""
    meter = pyln.Meter(sample_rate)

    # add stereo dim if needed
    if x.shape[0] < 2:
        x = x.repeat(2, 1)

    return torch.tensor(meter.integrated_loudness(x.permute(1, 0).numpy()))


class MelSpectralDistance(torch.nn.Module):
    def __init__(self, sample_rate, length=65536):
        super().__init__()
        self.error = auraloss.freq.MelSTFTLoss(
            sample_rate,
            fft_size=length,
            hop_size=length,
            win_length=length,
            w_sc=0,
            w_log_mag=1,
            w_lin_mag=1,
            n_mels=128,
            scale_invariance=False,
        )

        # I think scale invariance may not work well,
        # since aspects of the phase may be considered?

    def forward(self, input, target):
        return self.error(input, target)


class PESQ(torch.nn.Module):
    def __init__(self, sample_rate):
        super().__init__()
        self.sample_rate = sample_rate

    def forward(self, input, target):
        if self.sample_rate != 16000:
            target = resampy.resample(
                target.view(-1).numpy(),
                self.sample_rate,
                16000,
            )
            input = resampy.resample(
                input.view(-1).numpy(),
                self.sample_rate,
                16000,
            )

        return pesq(
            16000,
            target,
            input,
            "wb",
        )


class CrestFactorError(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        return torch.nn.functional.l1_loss(
            crest_factor(input),
            crest_factor(target),
        ).item()


class RMSEnergyError(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        return torch.nn.functional.l1_loss(
            rms_energy(input),
            rms_energy(target),
        ).item()


class SpectralCentroidError(torch.nn.Module):
    def __init__(self, sample_rate, n_fft=2048, hop_length=512):
        super().__init__()

        self.spectral_centroid = torchaudio.transforms.SpectralCentroid(
            sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
        )

    def forward(self, input, target):
        return torch.nn.functional.l1_loss(
            self.spectral_centroid(input + 1e-16).mean(),
            self.spectral_centroid(target + 1e-16).mean(),
        ).item()


class LoudnessError(torch.nn.Module):
    def __init__(self, sample_rate: int, peak_normalize: bool = False):
        super().__init__()
        self.sample_rate = sample_rate
        self.peak_normalize = peak_normalize

    def forward(self, input, target):

        if self.peak_normalize:
            # peak normalize
            x = input / input.abs().max()
            y = target / target.abs().max()
        else:
            x = input
            y = target

        return torch.nn.functional.l1_loss(
            loudness(x.view(1, -1), self.sample_rate),
            loudness(y.view(1, -1), self.sample_rate),
        ).item()


def calculate_mel_spectral_distance(input, target, sample_rate):
    msd = MelSpectralDistance(sample_rate)
    return msd(input, target).item()


def calculate_pesq(input, target, sample_rate):
    pesq_error = PESQ(sample_rate)
    return pesq_error(input, target)


def calculate_crest_factor_error(input, target):
    cf_error = CrestFactorError()
    return cf_error(input, target)


def calculate_rms_energy_error(input, target):
    rms_error = RMSEnergyError()
    return rms_error(input, target)


def calculate_spectral_centroid_error(input, target, sample_rate):
    sc_error = SpectralCentroidError(sample_rate)
    return sc_error(input, target)


def calculate_loudness_error(input, target, sample_rate, peak_normalize=False):
    loudness_error = LoudnessError(sample_rate, peak_normalize)
    return loudness_error(input, target)




def calculate_all_errors(input, target, sample_rate):
    # Convert TensorFlow tensors to NumPy arrays
    input_np = input.numpy()
    target_np = target.numpy()

    # Convert NumPy arrays to PyTorch tensors
    input_tensor = torch.from_numpy(input_np)
    target_tensor = torch.from_numpy(target_np)

    # Expand dimensions if necessary (assuming input and target are single-channel signals)
    if len(input_tensor.shape) == 1:  # If input is mono (1D tensor), add a batch and channel dimension
        input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
        target_tensor = target_tensor.unsqueeze(0).unsqueeze(0)
    elif len(input_tensor.shape) == 2:  # If input has only batch dimension, add a channel dimension
        input_tensor = input_tensor.unsqueeze(1)
        target_tensor = target_tensor.unsqueeze(1)

    # Calculate errors
    errors = {}
    errors['Mel Spectral Distance'] = calculate_mel_spectral_distance(input_tensor, target_tensor, sample_rate)
    #errors['PESQ'] = calculate_pesq(input_tensor, target_tensor, sample_rate)
    errors['Crest Factor Error'] = calculate_crest_factor_error(input_tensor, target_tensor)
    errors['RMS Energy Error'] = calculate_rms_energy_error(input_tensor, target_tensor)
    errors['Spectral Centroid Error'] = calculate_spectral_centroid_error(input_tensor, target_tensor, sample_rate)
    errors['Loudness Error'] = calculate_loudness_error(input_tensor, target_tensor, sample_rate)
    
    return errors


def transform_dsp_parameters(dsp_arg):
    dsp_transformed = {
        "low_shelf_gain_dB": dsp_arg[0]*48-24,
        "low_shelf_cutoff_freq": 16 + (np.exp(dsp_arg[1]) - 1) / (np.e - 1) * (128-16),
        "low_shelf_q_factor": dsp_arg[2]*8+0.707,
        "first_band_gain_dB": dsp_arg[3]*48-24,
        "first_band_freq": 128 + (np.exp(dsp_arg[4]) - 1) / (np.e - 1) * (512 - 128),
        "first_band_q_factor": dsp_arg[5]*8+0.707,
        "second_band_gain_dB": dsp_arg[6]*48-24,
        "second_band_freq": 512 + (np.exp(dsp_arg[7]) - 1) / (np.e - 1) * (1024 - 512),
        "second_band_q_factor": dsp_arg[8]*8+0.707,
        "third_band_gain_dB": dsp_arg[9]*48-24,
        "third_band_freq": 1024 + (np.exp(dsp_arg[10]) - 1) / (np.e - 1) * (2048 - 1024),
        "third_band_q_factor": dsp_arg[11]*8+0.707,
        "fourth_band_gain_dB": dsp_arg[12]*48-24,  # It seems there's a mistake in the original code for the index; it should likely be dsp_arg[12] not dsp_arg[11]
        "fourth_band_freq": 2048 + (np.exp(dsp_arg[13]) - 1) / (np.e - 1) * (4096 - 2048),
        "fourth_band_q_factor": abs(dsp_arg[14])*8+0.707,  # Using abs() as in original, assuming a typo in the original instructions
        "high_shelf_gain_dB": dsp_arg[15]*48-24,
        "high_shelf_cutoff_freq": 4096 + (np.exp(dsp_arg[16]) - 1) / (np.e - 1) * (10240 - 4096),
        "high_shelf_q_factor": dsp_arg[17]*8+0.707,
        # Extracting DRC parameters
        "drc_threshold": 0 - dsp_arg[18] * 60,  # Adjusted according to your new formula
        "drc_ratio": dsp_arg[19] * 10,  # Updated multiplication factor
        "drc_attack_time": max(dsp_arg[20] / 10, 0.0001),  # Updated to use max() for lower bound
        "drc_release_time": max(dsp_arg[21] * 3, 0.005),  # Updated multiplication and lower bound
        "drc_knee_dB": dsp_arg[22] * 24,  # As specified
        "drc_makeup_gain_dB": dsp_arg[23] * 20,  # Updated multiplication factor
    }
    return dsp_transformed

def transform_dsp_parameters_for_diagram(dsp_arg):
    """
    Transforms and renames DSP parameters according to specified formulas for diagram presentation.
    
    Args:
        dsp_arg (list): A list of DSP parameter values.
        
    Returns:
        dict: A dictionary with transformed and descriptively named DSP parameter values for diagrams.
    """
    dsp_transformed_for_diagram = {
        "Low Shelf Gain (dB)": dsp_arg[0]*48-24,
        "Low Shelf Cutoff Frequency (Hz)": 16 + (np.exp(dsp_arg[1]) - 1) / (np.e - 1) * (128-16),
        "Low Shelf Q Factor": dsp_arg[2]*8+0.707,
        "First Band Gain (dB)": dsp_arg[3]*48-24,
        "First Band Frequency (Hz)": 128 + (np.exp(dsp_arg[4]) - 1) / (np.e - 1) * (512 - 128),
        "First Band Q Factor": dsp_arg[5]*8+0.707,
        "Second Band Gain (dB)": dsp_arg[6]*48-24,
        "Second Band Frequency (Hz)": 512 + (np.exp(dsp_arg[7]) - 1) / (np.e - 1) * (1024 - 512),
        "Second Band Q Factor": dsp_arg[8]*8+0.707,
        "Third Band Gain (dB)": dsp_arg[9]*48-24,
        "Third Band Frequency (Hz)": 1024 + (np.exp(dsp_arg[10]) - 1) / (np.e - 1) * (2048 - 1024),
        "Third Band Q Factor": dsp_arg[11]*8+0.707,
        "Fourth Band Gain (dB)": dsp_arg[12]*48-24,
        "Fourth Band Frequency (Hz)": 2048 + (np.exp(dsp_arg[13]) - 1) / (np.e - 1) * (4096 - 2048),
        "Fourth Band Q Factor": abs(dsp_arg[14])*8+0.707,
        "High Shelf Gain (dB)": dsp_arg[15]*48-24,
        "High Shelf Cutoff Frequency (Hz)": 4096 + (np.exp(dsp_arg[16]) - 1) / (np.e - 1) * (10240 - 4096),
        "High Shelf Q Factor": dsp_arg[17]*8+0.707,
        "DRC Threshold (dB)": 0 - dsp_arg[18] * 60,
        "DRC Ratio": dsp_arg[19] * 10,
        "DRC Attack Time (s)": max(dsp_arg[20] / 10, 0.0001),
        "DRC Release Time (s)": max(dsp_arg[21] * 3, 0.005),
        "DRC Knee (dB)": dsp_arg[22] * 24,
        "DRC Makeup Gain (dB)": dsp_arg[23] * 20,
    }
    return dsp_transformed_for_diagram

def get_dsp_parameter(dsp_arg):
    dsp_params = {
        "low_shelf_gain_dB": dsp_arg[0],
        "low_shelf_cutoff_freq": dsp_arg[1],
        "low_shelf_q_factor": dsp_arg[2],
        "first_band_gain_dB": dsp_arg[3],
        "first_band_freq": dsp_arg[4],
        "first_band_q_factor": dsp_arg[5],
        "second_band_gain_dB": dsp_arg[6],
        "second_band_freq": dsp_arg[7],
        "second_band_q_factor": dsp_arg[8],
        "third_band_gain_dB": dsp_arg[9],
        "third_band_freq": dsp_arg[10],
        "third_band_q_factor": dsp_arg[11],
        "fourth_band_gain_dB": dsp_arg[12],
        "fourth_band_freq": dsp_arg[13],
        "fourth_band_q_factor": dsp_arg[14],
        "high_shelf_gain_dB": dsp_arg[15],
        "high_shelf_cutoff_freq": dsp_arg[16],
        "high_shelf_q_factor": dsp_arg[17],
        "drc_threshold": dsp_arg[18],
        "drc_ratio": dsp_arg[19],
        "drc_attack_time": dsp_arg[20],
        "drc_release_time": dsp_arg[21],
        "drc_knee_dB": dsp_arg[22],
        "drc_makeup_gain_dB": dsp_arg[23]
    }
    return dsp_params

def get_dsp_parameter_for_diagram(dsp_arg):
    dsp_params_diagram = {
        "Low Shelf Gain (dB)": dsp_arg[0],
        "Low Shelf Cutoff Frequency (Hz)": dsp_arg[1],
        "Low Shelf Q Factor": dsp_arg[2],
        "First Band Gain (dB)": dsp_arg[3],
        "First Band Frequency (Hz)": dsp_arg[4],
        "First Band Q Factor": dsp_arg[5],
        "Second Band Gain (dB)": dsp_arg[6],
        "Second Band Frequency (Hz)": dsp_arg[7],
        "Second Band Q Factor": dsp_arg[8],
        "Third Band Gain (dB)": dsp_arg[9],
        "Third Band Frequency (Hz)": dsp_arg[10],
        "Third Band Q Factor": dsp_arg[11],
        "Fourth Band Gain (dB)": dsp_arg[12],
        "Fourth Band Frequency (Hz)": dsp_arg[13],
        "Fourth Band Q Factor": dsp_arg[14],
        "High Shelf Gain (dB)": dsp_arg[15],
        "High Shelf Cutoff Frequency (Hz)": dsp_arg[16],
        "High Shelf Q Factor": dsp_arg[17],
        "DRC Threshold (dB)": dsp_arg[18],
        "DRC Ratio": dsp_arg[19],
        "DRC Attack Time (s)": dsp_arg[20],
        "DRC Release Time (s)": dsp_arg[21],
        "DRC Knee (dB)": dsp_arg[22],
        "DRC Makeup Gain (dB)": dsp_arg[23]
    }
    return dsp_params_diagram



def plot_dsp_table(dsp_transformed, dsp_values, ax):
    # Prepare data for the table
    column_labels = ["DSP Parameter", "Controller Output", "DSP Value"]
    table_data = []
    for key, transformed_value in dsp_transformed.items():
        # Get the original (not transformed) value
        original_value = dsp_values[key]
        # Check and round both original and transformed values if they are floats
        # Round the numerical values to two decimal places and convert them to strings
        if isinstance(original_value, float):
            original_value = f"{original_value:.2f}"
        else:
            original_value = str(original_value)
            
        if isinstance(transformed_value, float):
            transformed_value = f"{transformed_value:.2f}"
        else:
            transformed_value = str(transformed_value)
        
        # Append the row with the name, original, and transformed values
        table_data.append([key, original_value, transformed_value])

    # Create the table and specify the cell alignment
    table = ax.table(cellText=table_data, colLabels=column_labels, loc='center', cellLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.2)
    # Align the column headers to the left
    for (i, _), cell in table.get_celld().items():
        if i == 0:
            cell.set_text_props(horizontalalignment='left')
    # Turn off the axis
    ax.axis('off')


def plot_error_table(errors, ax):
    ax.axis('off')  # Turn off axis for the table
    # Round the values to two decimal places and convert to string
    cell_text = [[k, f"{v:.2f}"] for k, v in errors.items()]
    # Determine the maximum length of text in each column to set column width
    col_widths = [max(len(str(item)) for item in col) / 80 for col in zip(*cell_text)]  # Further decreased size factor
    # Create the table with cell sizes fit to the text
    table = ax.table(cellText=cell_text,
                     loc='center',
                     colLabels=['Error Metric', 'Value'],
                     cellLoc='left',
                     colWidths=col_widths)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    return table

def organize_files_by_style(input_files, reference_files, styles):
    # Initialize dictionaries to hold the input and reference files organized by style
    inputByStyles_paths = {style: [] for style in styles}
    refByStyles_paths = {style: [] for style in styles}

    # Create a mapping from identifiers to reference files for easy lookup
    reference_mapping = {ref.split('/')[-1]: ref for ref in reference_files}

    # Process each input file to extract its style and match it with the corresponding reference file
    for input_file in input_files:
        file_name = input_file.split('/')[-1]  # Extract the filename
        file_id = file_name.split('_')[0]  # Assuming the ID is the first part before '_'
        file_style = next((style for style in styles if style in file_name), None)  # Find the style in the filename

        # If a style was found and there's a matching reference file, add them to their respective dictionaries
        if file_style and file_id + '_' + file_style + '_IR.wav' in reference_mapping:
            matched_reference = reference_mapping[file_id + '_' + file_style + '_IR.wav']
            inputByStyles_paths[file_style].append(input_file)
            refByStyles_paths[file_style].append(matched_reference)

    return inputByStyles_paths, refByStyles_paths

def get_paths_for_style(inputByStyles_paths, refByStyles_paths, style):
    """
    Extracts and returns the paths for a specific style.

    Args:
        inputByStyles_paths (dict): A dictionary with styles as keys and lists of input file paths as values.
        refByStyles_paths (dict): A dictionary with styles as keys and lists of reference file paths as values.
        style (str): The style for which paths are to be returned.

    Returns:
        tuple: A tuple containing two lists, the first with input file paths and the second with reference file paths for the specified style.
    """
    inputForStyle_paths = inputByStyles_paths.get(style, [])
    refForStyle_paths = refByStyles_paths.get(style, [])
    
    return inputForStyle_paths, refForStyle_paths


def get_and_save_paths_for_style(inputByStyles_paths, refByStyles_paths, style, dir_path, no_of_files, save=False):
    """
    Extracts and optionally saves a specified number of files for a specific style into designated directories.

    Args:
        inputByStyles_paths (dict): Dictionary with styles as keys and lists of input file paths as values.
        refByStyles_paths (dict): Dictionary with styles as keys and lists of reference file paths as values.
        style (str): The style for which paths are to be processed.
        dir_path (str): Base directory path where files should be saved.
        no_of_files (int): Number of files to process and optionally save.
        save (bool): If True, saves files to new directory structure.

    Returns:
        tuple: Two lists, the first with processed input file paths and the second with processed reference file paths.
    """
    # Retrieve paths
    input_paths = inputByStyles_paths.get(style, [])
    ref_paths = refByStyles_paths.get(style, [])

    # Limit the number of files to process
    input_paths = input_paths[:no_of_files]
    ref_paths = ref_paths[:no_of_files]
    
    if save:
        # Define new directories
        input_dir = os.path.join(dir_path, 'dirty', style)
        ref_dir = os.path.join(dir_path, 'clean', style)

        # Create the directories if they don't already exist
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(ref_dir, exist_ok=True)

        # Save files to the new directories
        for input_path in input_paths:
            shutil.copy(input_path, input_dir)
        for ref_path in ref_paths:
            shutil.copy(ref_path, ref_dir)
    
    # Generate new paths if saved, else return original paths
    saved_input_paths = [os.path.join(input_dir, os.path.basename(p)) for p in input_paths] if save else input_paths
    saved_ref_paths = [os.path.join(ref_dir, os.path.basename(p)) for p in ref_paths] if save else ref_paths

    return saved_input_paths, saved_ref_paths


def plot_boxplot_metrics(accumulated_metrics,plots_saving_on=False, y_lim=(0, 100)):
    """
    Creates a box plot for the accumulated metrics, excluding "Spectral Centroid Error" from the plot,
    but includes it in the summary table showing mean and std, positioned inside the box plot at the top right.
    Adjusts the table for reduced row heights, increased column widths, and overall smaller size while keeping font size unchanged.
    
    Args:
    accumulated_metrics (dict): A dictionary where keys are metric names and values are lists of metric values.
    """
    # Filter out metrics not intended for the plot but keep them for the table
    plot_data = {metric: values for metric, values in accumulated_metrics.items() if metric != "Spectral Centroid Error"}
    data_to_plot = [values for values in plot_data.values()]
    labels = [metric for metric in plot_data.keys()]

    # Create the figure and axis for the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create the box plot for filtered metrics
    if data_to_plot:  # Check if there is data to plot
        bplot = ax.boxplot(data_to_plot, vert=True, patch_artist=True, labels=labels)

        # Customize the appearance for a classical look
        for patch, color in zip(bplot['boxes'], ['lightblue'] * len(labels)):
            patch.set_facecolor(color)
        for whisker in bplot['whiskers']:
            whisker.set(color='black', linewidth=1.5, linestyle=":")
        for cap in bplot['caps']:
            cap.set(color='black', linewidth=2)
        for median in bplot['medians']:
            median.set(color='red', linewidth=2)
        for flier in bplot['fliers']:
            flier.set(marker='D', color='#e7298a', alpha=0.5)

    # Calculate mean and std for all metrics for the table
    table_data = [["Metric", "Mean", "Std"]]
    for metric, values in accumulated_metrics.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        table_data.append([metric, f"{mean_val:.2f}", f"{std_val:.2f}"])

    # Adjust plot to make room for the table
    plt.subplots_adjust(right=0.7)

    # Add a table inside the plot at the top right for all metrics
    # Adjustments for smaller table with wider columns and reduced row height
    table = ax.table(cellText=table_data, cellLoc='center', loc='center right',
                     bbox=[0.6, 0.73, 0.36, 0.22], colWidths=[1.6, 0.4, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(0.9, 0.9)  # Slightly reduce the scale to make the table overall smaller

    # Adding titles and labels
    ax.set_title('Metrics Distribution: Mean and Standarddeviation (Small Model)')
    ax.set_ylabel('Error Value')
    ax.grid(True, linestyle=':', linewidth='0.5', color='gray')

    plt.xticks(rotation=45)
    
    plt.ylim(y_lim)

    if plots_saving_on:
        plt.savefig('./plots/metrics_small_model.jpg', bbox_inches='tight')
    plt.show()