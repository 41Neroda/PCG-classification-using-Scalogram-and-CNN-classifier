# PCG-classification-using-Scalogram-and-CNN-classifier
This is a project where we try to detect the abnomalities present in the PCG (phonocardiogram) signal of heart, using the pre-trained CNN model.

# Main Goal
This project classifies Phonocardiogram (PCG) heart sounds using scalogram and a Convolutional Neural Network (CNN). The goal is to classify whether a person' heart is normal or abnormal from the PCG recordings.

# Dataset
Dataset is taken from the PhysioNet/CinC Challenge 2016. It comprises of 3240 PCG recordings sourced from various patients. These recordings have been divided into 6 distinct training datasets, labeled as training-a, training-b, training-c, training-d, training-e and training-f.

These datasets contains different number of recordings, with training-e containing the highest number of recording with 2141 (normal-1958 and abnormal-183) and training-c having the lowest with 31 (normal-7 and abnormal-24). Each recording is provided in the .wav file format and is uniquely identified by a record name, such as "a0001.wav". Here, the letter 'a' denotes the training dataset name, while the numeric portion '0001' denotes the specific record number. This stardardized file naming convention facilitates organization and retrieval of the PCG recordings for analysis and processing.

# Hardware Requirements
Processor Core i5

Hard Disk Drive 512 GB

16 GB RAM

GPU

# Software Requirements
Python modules: numpy, pandas, matplotlib, scikit-learn

Google colaboratory

# Preprocessing Steps
1. Denoising PCG signal using Discrete Wavelet Transform (DWT):
Denoising with DWT gives better result as it offers both time and frequency information, particularly for nonstationary signals like PCGs. As auscultation is prone to noise interference, this step is very crucial.

2. Butterworth Bandpass Filter:
To improve the quality of PCG signals, we apply a 4th-order Butterworth bandpass filter that focuses on the 20-400 Hz frequency range. This range is chosen because it contains the most important heart sound components. Using butterworth filter helps in effectively removing the noise without distorting the signal.

3. Scalogram Generation:
The PCG signals are converted into 2D scalogram images using Continuous Wavelet Transform (CWT). These images provide a time-frequency representation of the signals which makes it easier to analyze complex patterns in heart sounds.

# Pre-trained CNN Models
Some of the well-established pre-trained CNN models have been trained and evaluated to analyze their performance effectively.

1. VGG16 Model: It consists of 13 convolutional layers and 3 fully connected layers, renowned for its simplicity and effectiveness for image classification and object recognition. Here, the accuracy for training and testing are 86% and 83% respectively.

2. VGG19 Model: It consists of 19 layers, including 16 convolutional layers and 3 fully connected layers. It is widely used for computer vision tasks and takes 224x224 RGB images as input. In this implementation, it achieved the training and testing accuracies of 89% and 87%, demontrating its effectiveness in classifying PCG signals. 

3. DenseNet121 Model: The DenseNet121 model, with densely connected layers, enhances information flow by linking each layer to every other layer in a feedforward fashion. It consists of five pooling and convolution layers, three transition levels (6, 12, and 24 layers), one classification layer (16 units), and two DenseBlocks (1x1 and 3x3 convolutions). The model achieves 96% training accuracy and 83% testing accuracy, but its 92% testing loss indicates overfitting and poor generalization.

4. DenseNet169 Model: The DenseNet169 model, with 169 layers, achieves 99% training accuracy and 84% testing accuracy, but its 64% testing loss indicates overfitting. After applying L2 regularization, the testing loss drops to 44%, showing improved generalization and reduced overfitting.

5. DenseNet201 Model: The DenseNet201 model, the deepest variant with 201 layers, requires more parameters and computational resources for training and inference. It achieves 98% training accuracy and 83% testing accuracy, but its 92% testing loss indicates overfitting and poor generalization.

# Conclusion:
The project uses a CNN-based approach to classify Phonocardiogram (PCG) signals, distinguishing between normal and abnormal cardiac states. Several pretrained models were tested, and VGG19 delivered the best performance, achieving 89% training accuracy and 87% testing accuracy. Its superior feature extraction capabilities make it highly effective for automated cardiac diagnostics, aiding healthcare professionals in early detection and treatment of cardiovascular diseases. Future improvements could involve refined data preprocessing and model fine-tuning to further enhance accuracy.

Published paper: https://www.taylorfrancis.com/chapters/edit/10.1201/9781003513445-10/pcg-classification-using-scalogram-cnn-classifier-maibam-mangalleibi-chanu-ksh-merina-devi-khumbongmayum-belona-devi-preceela-chanu-irengbam-shougaijam-neroda-devi 
