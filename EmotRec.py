import cv2 as cv
import numpy as np
import face_recognition

from tensorflow.keras.models import load_model


class EmotRec:
    """Facial emotion recognition class.

    Attributes:
        SIZE_SHAPE (tuple): Shape for resizing images before predictions.
        CLASSES (list): Possible emotions that can be detected.
        FACE_DIFF_THRESHOLD (int): Threshold for detecting changes in facial landmarks.
    """

    SIZE_SHAPE = (224, 224)
    CLASSES = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise', 'uncertain']
    FACE_DIFF_THRESHOLD = 10

    def __init__(self, id_webcam, fer_path, face_detector_path):
        """Initializes the emotion recognition model and face detector.

        Args:
            id_webcam (int): ID of the webcam.
            fer_path (str): Path to the facial emotion recognition model.
            face_detector_path (str): Path to the face detector model.
        """

        self.prev_landmarks = {}
        self.capture = cv.VideoCapture(id_webcam)
        self.FER = load_model(fer_path)
        self.face_detector = cv.CascadeClassifier(face_detector_path)

        self.start_stream()

    def start_stream(self):
        """Starts the webcam stream and displays emotion labels on detected faces."""

        if not self.capture.isOpened():
            raise IOError("Cannot open webcam")

        label_dict = {}
        while True:
            ret, frame = self.capture.read()
            if not ret:
                print("Failed to retrieve frame")
                continue

            gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            faces = self.face_detector.detectMultiScale(gray_frame, 1.3, 5)

            new_landmarks = {}
            for i, (x, y, w, h) in enumerate(faces):
                current_face_img = frame[y:y + h, x:x + w]
                face_landmarks = face_recognition.face_landmarks(current_face_img)

                should_predict = True

                if i in self.prev_landmarks and face_landmarks:
                    current_all_landmarks, prev_all_landmarks = [], []

                    for feature in face_landmarks[0]:
                        current_all_landmarks.extend(face_landmarks[0][feature])
                        prev_all_landmarks.extend(self.prev_landmarks[i][feature])

                    diff = np.linalg.norm(np.array(current_all_landmarks) - np.array(prev_all_landmarks), axis=1).mean()

                    if diff <= self.FACE_DIFF_THRESHOLD:
                        should_predict = False

                if should_predict:
                    label = self.predict(current_face_img)
                    label_dict[i] = label

                if face_landmarks:
                    new_landmarks[i] = face_landmarks[0]

                frame = cv.putText(frame, label_dict.get(i, ""), (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0),
                                   2)
                frame = cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

            self.prev_landmarks = new_landmarks

            cv.imshow('Input', frame)

            if cv.waitKey(1) == 27 or cv.getWindowProperty('Input', cv.WND_PROP_VISIBLE) < 1:
                break

        self.capture.release()
        cv.destroyAllWindows()

    def predict(self, frame):
        """Predicts the emotion on the provided frame.

        Args:
            frame (array-like): Image frame to predict emotion.

        Returns:
            str: Predicted emotion.
        """

        predictions = self.FER.predict(
            np.expand_dims(
                cv.cvtColor(
                    cv.resize(frame, self.SIZE_SHAPE),
                    cv.COLOR_BGR2RGB),
                axis=0)
        )

        return self.CLASSES[np.argmax(predictions)]
