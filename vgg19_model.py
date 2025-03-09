import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.applications import VGG19
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Function to read scalogram images from folder
def read_images_from_folder(folder):
    images = []
    for filename in sorted(os.listdir(folder)):
        if filename.endswith(".png"):
            img_path = os.path.join(folder, filename)
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=(img_height, img_width))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            images.append(img_array)
    return np.array(images)

def read_labels_from_csv(csv_file):
    df = pd.read_csv(csv_file)
    labels = df['Abnormality'].values
    return labels

# Load scalogram images and labels
folder_path = "resized_final_scalogram"
csv_file = "output.csv"
img_height, img_width = 224, 224  # Define your desired image dimensions

images = read_images_from_folder(folder_path)
labels = read_labels_from_csv(csv_file)

# Preprocess the data (e.g., normalization)
images = images / 255.0  # Normalize pixel values to [0, 1]

# Convert labels to numerical format
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, stratify=labels, random_state=42)

# Load the pre-trained VGG19 model
base_model = VGG19(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# Add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)

# Add a fully-connected layer
x = Dense(1024, activation='relu')(x)

# Add dropout
x = Dropout(0.3)(x)

# Add a logistic layer for binary classification
predictions = Dense(1, activation='sigmoid')(x)

# This is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the base model
for layer in base_model.layers:
    layer.trainable = False

# Instantiate the Adam optimizer with the desired learning rate
optimizer = Adam(learning_rate=0.001)

# Compile the model with the Adam optimizer
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=200, batch_size=16, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print('\nTest accuracy:', test_acc)

# Make predictions
predictions = model.predict(X_test)

# Convert predictions to binary classes
binary_predictions = np.round(predictions).flatten()

# Generate classification report
print('\nClassification Report:')
print(classification_report(y_test, binary_predictions))

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, binary_predictions)
print('\nConfusion Matrix:')
print(conf_matrix)

# Plot accuracy and loss graphs
plt.figure(figsize=(12, 6))

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

plt.tight_layout()
plt.show()

# Save or serialize the model if desired
model.save("cvd_vgg19_200_model.h5")