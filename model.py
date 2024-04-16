import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, GlobalMaxPooling1D, LSTM
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns


paths = []
labels = []
for dirname, _, filenames in os.walk(r"C:\Users\shrey\OneDrive\Desktop\ML_Projects\EmotionDetector\Audio_Video\audio_dataset\TESS Toronto emotional speech set data"):
    for filename in filenames:
        paths.append(os.path.join(dirname, filename))
        label = filename.split('_')[-1].split('.')[0].lower()
        labels.append(label)

df = pd.DataFrame({'speech': paths, 'label': labels})

def extract_features(filename):
    y, sr = librosa.load(filename, duration=3, offset=0.5)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    rms = librosa.feature.rms(y=y)[0]
    spectral_center = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
    
    # Set the desired maximum length of the feature vectors
    max_length = 128
    
    # Pad the feature vectors to have the same length
    mfcc_pad = np.pad(mfcc, ((0, 0), (0, max_length - mfcc.shape[1])), mode='constant')
    rms_pad = np.pad(rms, (0, max_length - len(rms)), mode='constant')
    spectral_center_pad = np.pad(spectral_center, (0, max_length - len(spectral_center)), mode='constant')
    zero_crossing_rate_pad = np.pad(zero_crossing_rate, (0, max_length - len(zero_crossing_rate)), mode='constant')
    
    # Concatenate the feature vectors
    features = np.vstack([mfcc_pad, rms_pad, spectral_center_pad, zero_crossing_rate_pad])
    return features

X = np.array([extract_features(path) for path in df['speech']])
labels = df['label'].unique()
label_to_index = {label: idx for idx, label in enumerate(labels)}
y = np.array([label_to_index[label] for label in df['label']])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and compile the model
model = Sequential([
    Conv1D(64, 3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(123, return_sequences=True),  # LSTM layer with return_sequences=True to output sequences
    LSTM(64),  # Second LSTM layer without return_sequences to process sequences
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(len(labels), activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=512, shuffle=True)

model.save('model_1.keras')

# Optionally, you can also save the model architecture as JSON
model_json = model.to_json()
with open("model_1.json", "w") as json_file:
    json_file.write(model_json)