import tkinter as tk
from tkinter import Button, Label
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import librosa
import pyaudio
import time  # Import the time module

# Define the emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Load the trained model
model = load_model('model_1.keras')

# Define a function to extract features from audio data
def extract_features(audio_data, sr):
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=40)
    rms = librosa.feature.rms(y=audio_data)
    spectral_center = librosa.feature.spectral_centroid(y=audio_data, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio_data)
    
    # Pad the feature vectors to have the same length
    max_length = 128
    mfcc_pad = np.zeros((40, max_length))
    mfcc_pad[:, :mfcc.shape[1]] = mfcc
    rms_pad = np.zeros((1, max_length))
    rms_pad[:, :rms.shape[1]] = rms
    spectral_center_pad = np.zeros((1, max_length))
    spectral_center_pad[:, :spectral_center.shape[1]] = spectral_center
    zero_crossing_rate_pad = np.zeros((1, max_length))
    zero_crossing_rate_pad[:, :zero_crossing_rate.shape[1]] = zero_crossing_rate

    # Concatenate the feature vectors
    features = np.vstack([mfcc_pad, rms_pad, spectral_center_pad, zero_crossing_rate_pad])
    return features

def open_video():
    # Open the video capture device (webcam)
    cap = cv2.VideoCapture(0)

    # Get the sample rate of the audio
    sr = 44100

    # Set up PyAudio for audio recording
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=sr, input=True, frames_per_buffer=1024)

    last_emotion_time = time.time()  # Initialize last emotion time

    while cap.isOpened():
        # Read a frame from the video
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame from the video capture device.")
            break

        # Read audio from the microphone
        audio_data = stream.read(1024)
        audio_data = np.frombuffer(audio_data, dtype=np.int16)

        # Convert audio data to floating-point format
        audio_data = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max

        # Extract features from audio data
        features = extract_features(audio_data, sr)

        # Make a prediction using the loaded model
        prediction = model.predict(np.expand_dims(features, axis=0))
        emotion = emotion_labels[np.argmax(prediction)]

        # Display the detected emotion on the frame
        current_time = time.time()
        if current_time - last_emotion_time >= 1:  # Check if 1 second has passed
            # cv2.putText(frame, f"Emotion: {emotion}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Emotion: {emotion}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)  # Change text color to dark blue
            last_emotion_time = current_time  # Update last emotion time

        # Convert the frame to grayscale for face detection (if necessary)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the video frame
        cv2.imshow('Emotion Detection', frame)

        # Check for user input to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture device and close all windows
    cap.release()
    cv2.destroyAllWindows()
    stream.stop_stream()
    stream.close()
    p.terminate()


# GUI Setup
top = tk.Tk()
top.geometry('800x600')
top.title('Emotion Detection')
top.configure(background='#CDCDCD')

heading = Label(top, text='Emotion Detection', pady=20, font=('arial', 25, 'bold'))
heading.configure(background="#CDCDCD", foreground="#364156")
heading.pack()

open_video_button = tk.Button(top, text="Open Video", command=open_video, padx=10, pady=5)
open_video_button.configure(background="#364156", foreground="white", font=('arial', 10, 'bold'))
open_video_button.place(relx=0.4, rely=0.5)

top.mainloop()
