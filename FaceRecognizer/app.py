import os
import base64
import cv2
from flask import Flask, Response, render_template, request, jsonify
import numpy as np
from flask_cors import CORS
from flask_socketio import SocketIO
from model.faceRecognition import FaceRecognition  # Import your FaceRecognition class

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
CORS(app) 
socketio = SocketIO(app)
datasets_path = 'datasets'
haar_file = 'haarcascade_frontalface_default.xml'
face_recognition = FaceRecognition(datasets_path, haar_file)
face_cascade = cv2.CascadeClassifier(haar_file)

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        recognized_faces = face_recognition.recognize_faces(frame)
        # for face_name in recognized_faces:
        #     cv2.putText(frame, face_name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Encode the frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Use multipart response with a unique boundary string
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture')
def capture():
    return render_template('capture.html')

@app.route('/capture_data', methods=['POST'])
def capture_data():
    data = request.get_json()
    person_name = data.get('personName')
    num_images = 100
    images_captured = 0

    person_directory = os.path.join(datasets_path, person_name)
    os.makedirs(person_directory, exist_ok=True)

    webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use CAP_DSHOW flag

    def capture_image():
        nonlocal images_captured

        if images_captured >= num_images:
            socketio.emit('completion', {'message': 'Image capture completed for ' + person_name}, namespace='/test')
            return

        _, frame = webcam.read()
        if frame is None:
           print("Error: Couldn't read frame from the camera.")
           return
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 4)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            face = gray[y:y + h, x:x + w]
            face_resize = cv2.resize(face, (130, 100))
            image_filename = os.path.join(person_directory, f'{person_name}_{images_captured}.png')
            cv2.imwrite(image_filename, face_resize)
            images_captured += 1

        if len(faces) > 0:
            progress = int(images_captured / num_images * 100)
            socketio.emit('progress', {'progress': progress}, namespace='/test')

        socketio.start_background_task(target=capture_image)

    capture_image()

    return jsonify({'message': 'Image capture in progress for ' + person_name, 'face_detected': True})


if __name__ == '__main__':
    socketio.run(app, debug=True)
