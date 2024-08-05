# Sign Language Recognition: Live Detection to Text and Speech

## Authors: Elsie and Sedem

---

## Table of Contents


1. [Introduction](#introduction)
2. [link to the video](#link-to-the-video)
3. [Dataset](#dataset)
4. [Prerequisites](#prerequisites)
5. [Installation](#installation)
6. [Data Preparation](#data-preparation)
7. [Model Training](#model-training)
8. [Live Detection](#live-detection)
9. [Conclusion](#conclusion)

---

## Introduction

This project aims to create a system that recognizes sign language in real time and converts the recognized signs into text and speech. The primary goal is to assist communication for individuals who use sign language by providing a tool that translates sign language gestures into understandable text and spoken words.

## link to the video
https://youtu.be/0HBzqiFTleQ?feature=shared


## Dataset

The dataset used in this project consists of images of hands performing various sign language gestures. Each gesture corresponds to a specific word. The dataset was annotated with bounding boxes around the hands and labeled with the corresponding word.

*Link to Dataset: [[Dataset Link](https://drive.google.com/drive/folders/14vjb1WniH9Xc1E1bw7T7aiDPePYr0Yzx?usp=sharing)](#)*

## Prerequisites

Before running the project, ensure you have the necessary software and libraries installed. This includes Python, OpenCV, MediaPipe, gtts,pygame 

## Data Preparation
Resizing Images: All images are resized to 200x200 pixels for consistency.
Data Augmentation: The dataset is augmented by applying random rotations, horizontal flips, and scaling.
Annotating Images: Using Mediapipe, landmarks on the hands are detected, and bounding boxes are drawn around them for training the YOLOv8 model.
Data Splitting: The data is split into training, validation, and test sets

## Model Training
Download YOLOv8 Model: The YOLOv8 model is downloaded and prepared for training.
Training: The model is trained on the dataset with the initial parameters.
Hyperparameter Tuning: The model undergoes hyperparameter tuning to optimize its performance.
Saving the Model: The trained model is saved for future use.

## Live Detection
The model is tested live using a webcam to ensure it accurately recognizes and translates sign language gestures into text and voice.

 
## Conclusion
This project demonstrates the potential of using YOLOv8 for real-time sign language recognition and translation, and this can improve  communication for individuals using sign language as their only means of communicating. Thank you.


---

