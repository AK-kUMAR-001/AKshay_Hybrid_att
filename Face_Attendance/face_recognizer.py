import cv2
import face_recognition
import pickle
import numpy as np

import os

# Resolve paths relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENCODINGS_FILE = os.path.join(BASE_DIR, "encodings.pkl")
THRESHOLD = 0.55

class FaceRecognizer:
    def __init__(self):
        self.known_encodings = []
        self.known_names = []
        self.known_ids = []
        self.load_encodings()

    def load_encodings(self):
        try:
            with open(ENCODINGS_FILE, "rb") as f:
                data = pickle.load(f)
                self.known_encodings = data["encodings"]
                self.known_names = data["names"]
                self.known_ids = data["ids"]
            print("Encodings loaded successfully.")
        except FileNotFoundError:
            print("Error: encodings.pkl not found. Run face_encoder.py first.")

    def recognize_face(self, frame_encoding):
        if not self.known_encodings:
             return "Unknown", None

        matches = face_recognition.compare_faces(self.known_encodings, frame_encoding, tolerance=THRESHOLD)
        name = "Unknown"
        student_id = None
        
        face_distances = face_recognition.face_distance(self.known_encodings, frame_encoding)
        
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_names[best_match_index]
                student_id = self.known_ids[best_match_index]
                
        return name, student_id
