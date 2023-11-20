import cv2
import os
import numpy as np

class FaceRecognition:
    def __init__(self, datasets_path, haar_file):
        self.face_cascade = cv2.CascadeClassifier(haar_file)
        self.datasets_path = datasets_path
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.images, self.labels, self.names = self.load_dataset(datasets_path)
        self.train_model()

    def load_dataset(self, datasets_path):
        images, labels, names, id = [], [], {}, 0
        for subdir in os.listdir(datasets_path):
            names[id] = subdir
            subject_path = os.path.join(datasets_path, subdir)
            for filename in os.listdir(subject_path):
                path = os.path.join(subject_path, filename)
                label = id
                images.append(cv2.imread(path, 0))
                labels.append(int(label))
            id += 1
        return np.array(images), np.array(labels), names

    def train_model(self):
        self.face_recognizer.train(self.images, self.labels)

    def recognize_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        recognized_faces = []
        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]
            label, confidence = self.face_recognizer.predict(face_roi)
            if confidence < 800:  # Adjust the confidence threshold as needed
                recognized_name = self.names[label]
                recognized_faces.append(recognized_name)

                # Draw rectangle around the face
                color = (0, 255, 0)  # Yellow color in BGR
                stroke = 2  # Line thickness
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, stroke)

                # Put recognized name below the rectangle
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                font_color = (0, 255, 0)  # Red color in BGR
                font_thickness = 2
                text_size = cv2.getTextSize(recognized_name, font, font_scale, font_thickness)[0]
                text_x = x + (w - text_size[0]) // 2
                text_y = y + h + 20  # 20 pixels below the rectangle
                cv2.putText(frame, recognized_name, (text_x, text_y), font, font_scale, font_color, font_thickness, cv2.LINE_AA)

        return recognized_faces



