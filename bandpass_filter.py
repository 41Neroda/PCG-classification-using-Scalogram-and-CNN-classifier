import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal

# Define function to filter PCG signals
def filter_pcg_signal(pcg_signal, sample_rate):

    # Resample the PCG signal to 1000 Hz
    resampled_rate = 1000
    resampled_pcg_signal = signal.resample(pcg_signal, int(len(pcg_signal) * (resampled_rate / sample_rate)))

    # Define the band-pass filter parameters
    nyquist_freq = resampled_rate / 2
    lowcut = 20
    highcut = 400
    lowcut_normalized = lowcut / nyquist_freq
    highcut_normalized = highcut / nyquist_freq

    # Create the band-pass filter
    b, a = signal.butter(4, [lowcut_normalized, highcut_normalized], btype='band')

    # Apply the band-pass filter to the resampled PCG signal
    filtered_pcg_signal = signal.filtfilt(b, a, resampled_pcg_signal)
    return filtered_pcg_signal


# Function to process PCG WAV file
def process_pcg_wav_file(input_filename, output_folder):
    # Read PCG signal from WAV file
    fs, pcg_signal = wavfile.read(input_filename)

    # Normalize the signal
    normalised_signal = filter_pcg_signal(pcg_signal, fs)

    # Get the base filename without extension
    basename = os.path.splitext(os.path.basename(input_filename))[0]

    # Specify the output filename
    """output_filename = os.path.join(output_folder, f"{basename}_new_filtered.wav")"""
    output_filename = os.path.join(output_folder, f"{basename}.wav")

    # Save the normalized signal as WAV file
    wavfile.write(output_filename, fs, normalised_signal.astype(np.int16))


# Main function
if __name__ == "__main__":
    # Specify the input folder containing WAV files
    input_folder = "denoised_pcg_signals"

    # Specify the output folder for saving normalized signals
    output_folder = "final_filtered_pcg_signals"

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Check if the input folder exists
    if os.path.isdir(input_folder):
        # Iterate over files in the input folder
        for filename in os.listdir(input_folder):
            if filename.endswith('.wav'):
                input_file_path = os.path.join(input_folder, filename)
                process_pcg_wav_file(input_file_path, output_folder)
    else:
        print(f"Error: Folder '{input_folder}' not found.")
print("Bandpass Filtering completed.")