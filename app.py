import streamlit as st
import cv2
import time
from ultralytics import YOLO
from gtts import gTTS
import os
import pygame

# Load the YOLOv8 model
model_weights_path = '/Users/cliq-tech/Desktop/ES/nnn/yolov8/runs/detect/train/weights/best.pt'  # Replace with your model weights path
model = YOLO(model_weights_path)

# Initialize Pygame for audio playback
pygame.mixer.init()

# Define class names and indices
class_names = ['Angry', 'Hey', 'Honour', 'No', 'Ok', 'One', 'Peace', 'Promise', 'Three', 'Two']

# Function to play the detected word
def speak(word):
    tts = gTTS(text=word, lang='en')
    tts.save('word.mp3')
    pygame.mixer.music.load('word.mp3')
    pygame.mixer.music.play()

# Streamlit app
st.title('Sign Language Detection with YOLOv8')

# Sidebar options
option = st.sidebar.selectbox('Choose an option:', ['Live Detection without Voice', 'Live Detection with Voice'])

if option == 'Live Detection with Voice':
    st.write("Starting live detection with voice. Press 'Stop' to quit.")
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.write("Error: Could not open video stream")
    else:
        # Create a placeholder to display the frames
        frame_placeholder = st.empty()
        
        # Create a stop button
        stop_button_key = 'stop_button_' + str(time.time())
        stop_button = st.button('Stop Live Detection', key=stop_button_key)

        last_spoken_word = None
        last_speak_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.write("Error: Failed to capture image")
                break
            
            # Run YOLOv8 model on the frame
            results = model(frame)
            
            # Draw the predictions on the frame
            detected_labels = set()  # To keep track of detected words in the frame
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get the bounding box coordinates
                conf = box.conf[0]  # Confidence score
                cls = int(box.cls[0])  # Class index
                label = class_names[cls]  # Only display the class name
                
                detected_labels.add(label)  # Add detected label to the set
                
                # Draw the bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw the label at the top left corner
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Speak out the detected words
            current_time = time.time()
            if detected_labels and (last_spoken_word is None or current_time - last_speak_time >= 3):
                for label in detected_labels:
                    if label != last_spoken_word:
                        speak(label)
                        last_spoken_word = label
                        last_speak_time = current_time
                        break  # Only speak one word per frame
            
            # Display the frame
            frame_placeholder.image(frame, channels="BGR")

            # Check if the stop button was pressed
            if stop_button:
                break

        cap.release()
        cv2.destroyAllWindows()

elif option == 'Live Detection without Voice':
    st.write("Starting live detection without voice. Press 'Stop' to quit.")
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.write("Error: Could not open video stream")
    else:
        # Create a placeholder to display the frames
        frame_placeholder = st.empty()
        
        # Create a stop button
        stop_button_key = 'stop_button_' + str(time.time())
        stop_button = st.button('Stop Live Detection', key=stop_button_key)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.write("Error: Failed to capture image")
                break
            
            # Run YOLOv8 model on the frame
            results = model(frame)
            
            # Draw the predictions on the frame
            detected_labels = set()  # To keep track of detected words in the frame
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get the bounding box coordinates
                conf = box.conf[0]  # Confidence score
                cls = int(box.cls[0])  # Class index
                label = class_names[cls]  # Only display the class name
                
                detected_labels.add(label)  # Add detected label to the set
                
                # Draw the bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw the label at the top left corner
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Display the frame
            frame_placeholder.image(frame, channels="BGR")

            # Display all detected labels
            for i, label in enumerate(detected_labels):
                st.write(f"Detected: {label}")

            # Check if the stop button was pressed
            if stop_button:
                break

        cap.release()
        cv2.destroyAllWindows()
