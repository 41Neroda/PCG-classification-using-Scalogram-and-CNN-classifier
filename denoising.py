import os
import numpy as np
from scipy.io import wavfile
from scipy import signal
import pywt
from scipy.signal import morlet2, spectrogram

def bayesian_soft_thresholding(x, sigma_sq):
    threshold = np.sqrt(3 * sigma_sq)
    return np.sign(x) * np.maximum(0, np.abs(x) - threshold)

def sym20_wavelet_bayesian_soft_thresholding(input_file, output_folder, num_levels):
    # Load the PCG signal
    sample_rate, pcg_signal = wavfile.read(input_file)
    coeffs = pywt.wavedec(pcg_signal, 'sym20', level=num_levels)
    thresholded_coeffs = [bayesian_soft_thresholding(c, np.var(c)) for c in coeffs]
    reconstructed_data = pywt.waverec(thresholded_coeffs, 'sym20')
    # Save the filtered PCG signal
    output_file = os.path.join(output_folder, os.path.basename(input_file))
    wavfile.write(output_file, sample_rate, reconstructed_data.astype(np.int16))


# Get the current directory
current_directory = os.getcwd()

# List of folders containing PCG signal files
input_folders = ['training-a', 'training-b', 'training-c', 'training-d', 'training-e', 'training-f']

# Folder to save filtered PCG signal files
output_folder = os.path.join(current_directory, 'denoised_pcg_signals')

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Iterate over input folders
for folder in input_folders:
    input_folder = os.path.join(current_directory, folder)
    # Check if the input folder exists
    if os.path.exists(input_folder):
        # Iterate over PCG signal files in the input folder
        for filename in os.listdir(input_folder):
            if filename.endswith('.wav'):
                input_file = os.path.join(input_folder, filename)
                # Apply Symlet wavelet with 10 decomposition levels and Bayesian soft thresholding
                reconstructed_data = sym20_wavelet_bayesian_soft_thresholding(input_file, output_folder, num_levels=10)
    else:
        print(f"Folder '{folder}' not found.")

print("Denoising completed.")