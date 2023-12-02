# Face Recognition Web Application

This repository contains the source code for a real-time face recognition web application developed using Python, Flask, OpenCV, and Socket.io.

## Project Overview

The project involves the creation of a web application that captures live video from the user's webcam, performs real-time face recognition, and provides additional features for data capture. The application utilizes the Flask web framework, OpenCV for face recognition, and Socket.io for real-time communication.

## Key Components

### 1. FaceRecognition Class (`model/faceRecognition.py`)

- Implements face recognition using OpenCV.
- Utilizes a pre-trained face detection model and LBPH face recognizer.
- Loads images from a specified dataset, trains the face recognizer model, and stores necessary information for recognition.

### 2. Capture HTML Page (`templates/capture.html`)

- Provides a user-friendly interface for capturing face recognition data.
- Allows users to input a person's name, start/stop the webcam, and captures multiple images for training.
- Real-time updates on progress and notifications enhance the user experience.

### 3. Index HTML Page (`templates/index.html`)

- Displays the live face recognition feed with an option to capture data.
- Integrates seamlessly with the Flask application, providing a clean and visually appealing layout.

### 4. Flask Application (`app.py`)

- Serves as the backend for the web application.
- Implements routes for the main index page, video feed, and data capture page.
- Uses Flask-SocketIO for real-time communication with the frontend.
- Integrates the FaceRecognition class for real-time face recognition.

### 5. Socket.io Integration (`templates/capture.html`)

- Enables real-time updates on data capture progress.
- Notifies users upon completion and handles potential errors during image capture.

## Usage

1. Clone the repository: `git clone https://github.com/your-username/face-recognition-web-app.git`
2. Navigate to the project directory: `cd face-recognition-web-app`
3. Install dependencies: `pip install -r requirements.txt`
4. Run the Flask application: `python app.py`
5. Access the web application in your browser: `http://127.0.0.1:5000/`

## Technologies Used

- Python
- Flask (Web Framework)
- OpenCV (Computer Vision Library)
- Socket.io (Real-time Communication)
- HTML, CSS (Frontend)

## Future Enhancements

- Implement user authentication and secure data storage.
- Expand the dataset and improve face recognition accuracy.
- Explore additional features, such as facial expression recognition.
- Optimize for deployment in a production environment.

This project provides a foundation for real-time face recognition applications, demonstrating the integration of various technologies to create a responsive and interactive user experience.
