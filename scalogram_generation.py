import os
import numpy as np
from scipy.io import wavfile
import pywt
import matplotlib.pyplot as plt

def generate_scalogram(signal, wavelet='morl', scales=None):
    if scales is None:
        scales = range(1, 128)  # Default scales

    try:
        coefficients, frequencies = pywt.cwt(signal, scales, wavelet)
        power = (abs(coefficients)) ** 2
        return power, frequencies
    except Exception as e:
        print(f"Error generating scalogram for signal: {e}")
        return None, None

def plot_scalogram(power, frequencies):
    plt.imshow(power, extent=[0, len(power[0]), frequencies[-1], frequencies[0]], aspect='auto', cmap='jet', vmax=np.max(power)*0.01) # Adjust the scaling factor as needed
    plt.axis('off')

def generate_and_save_scalogram(input_folder, output_folder, wavelet='morl', scales=None):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over files in the input folder
    for filename in sorted(os.listdir(input_folder)):
        if filename.endswith('.wav'):
            input_file_path = os.path.join(input_folder, filename)
            output_file_path = os.path.join(output_folder, os.path.splitext(filename)[0] + '_scalogram.png')

            try:
                # Load PCG signal from WAV file
                fs, pcg_signal = wavfile.read(input_file_path)

                # Only consider one channel if it's a stereo recording
                if pcg_signal.ndim > 1:
                    pcg_signal = pcg_signal[:, 0]

                # Normalize signal based on maximum absolute value
                max_abs = np.max(np.abs(pcg_signal))
                pcg_signal = pcg_signal.astype(np.float32)
                pcg_signal /= max_abs

                # Generate scalogram
                power, frequencies = generate_scalogram(pcg_signal, wavelet=wavelet, scales=scales)

                if power is not None and frequencies is not None:
                    # Plot and save scalogram
                    plot_scalogram(power, frequencies)
                    plt.savefig(output_file_path, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    print(f"Scalogram saved to: {output_file_path}")

                else:
                    print(f"Skipping processing of {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

# Main function
if __name__ == "__main__":
    # Specify the input folder containing normalized PCG signals
    input_folder = "final_filtered_pcg_signals"

    # Specify the output folder for saving scalogram images
    output_folder = "scalogram"

    # Generate and save scalograms for signals in the input folder
    generate_and_save_scalogram(input_folder, output_folder)