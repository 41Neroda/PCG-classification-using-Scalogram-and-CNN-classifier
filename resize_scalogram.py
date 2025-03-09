import os
from PIL import Image

def resize_images(input_folder, output_folder, size=(224, 224)):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            try:
                # Open an image file
                with Image.open(os.path.join(input_folder, filename)) as img:
                    # Resize image
                    img = img.resize(size, Image.LANCZOS)
                    # Save it to the output folder
                    img.save(os.path.join(output_folder, filename))
                    print(f"Resized and saved {filename} to {output_folder}")
            except Exception as e:
                print(f"Could not process {filename}: {e}")

# Define the input and output folder paths
input_folder = 'scalogram'
output_folder = 'last_scalogram'

# Call the resize function
resize_images(input_folder, output_folder)