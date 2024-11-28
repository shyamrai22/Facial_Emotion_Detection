from flask import Flask, render_template, Response, request
import cv2
from transformers import pipeline
from PIL import Image
import numpy as np
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load Hugging Face emotion detection model (image-classification pipeline)
emotion_model = pipeline('image-classification', model='dima806/facial_emotions_image_detection')

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

camera = None  # Initialize the camera variable

def generate_frames():
    global camera
    while camera and camera.isOpened():
        success, frame = camera.read()  # Read frame from webcam
        if not success:
            logging.error("Failed to read frame from camera")
            break
        else:
            # Convert the frame to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces using the Haar Cascade
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            for (x, y, w, h) in faces:
                # Draw a rectangle around the faces
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # Crop the detected face from the frame
                face = frame[y:y + h, x:x + w]

                # Convert the face to RGB and then to PIL image
                face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(face_rgb)

                # Pass the PIL image to the Hugging Face emotion detection model
                try:
                    result = emotion_model(pil_image)
                    label = result[0]['label']
                    confidence = result[0]['score']
                except Exception as e:
                    logging.error(f"Error during emotion detection: {e}")
                    label = "Unknown"
                    confidence = 0.0

                # Display the emotion prediction result on the frame
                cv2.putText(frame, f'{label}: {confidence:.2f}', (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Convert the processed frame back to JPEG format for streaming
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                logging.error("Failed to encode frame as JPEG")
                break
            frame = buffer.tobytes()

            # Stream the frame as a response
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_webcam', methods=['POST'])
def start_webcam():
    global camera
    if camera is not None and camera.isOpened():
        return "Webcam is already running", 200
    try:
        if camera is not None:
            camera.release()  # Release previous camera instance if it exists
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            logging.error("Failed to open webcam")
            return "Failed to open webcam", 500
        return "Webcam started", 200
    except Exception as e:
        logging.error(f"Error starting webcam: {e}")
        return "Failed to start webcam", 500

@app.route('/stop_webcam', methods=['POST'])
def stop_webcam():
    global camera
    try:
        if camera is not None and camera.isOpened():
            camera.release()
            camera = None
            return "Webcam stopped", 200
        else:
            return "Webcam is not running", 200
    except Exception as e:
        
        logging.error(f"Error stopping webcam: {e}")
        return "Failed to stop webcam", 500

@app.route('/')
def index():
    return render_template('index.html')


