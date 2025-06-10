import cv2
import numpy as np
from PIL import Image
import os
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load the model
model = MobileNetV2(weights='imagenet')

# Load OpenCV face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def model_predict(frame, model):
    # Convert frame to PIL Image
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img = img.resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x, mode='tf')
    preds = model.predict(x)
    return preds

def detect_faces(frame):
    try:
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # Draw rectangles around faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        if len(faces) > 0:
            return f"Faces detected: {len(faces)}"
        else:
            return "No faces detected"
    except Exception as e:
        return "Face detection error"

# Open the default camera
cam = cv2.VideoCapture(0)

# Initialize prediction variables
frame_count = 0
prediction_text = "Loading..."
face_text = "Detecting faces..."

while True:
    ret, frame = cam.read()
    
    if not ret:
        break
    
    # Make prediction every 30 frames to reduce computation
    frame_count += 1
    if frame_count % 30 == 0:
        try:
            # Object detection
            preds = model_predict(frame, model)
            decoded_preds = decode_predictions(preds, top=1)[0]
            prediction_text = f"Object: {decoded_preds[0][1]}: {decoded_preds[0][2]:.2f}"
        except:
            prediction_text = "Processing..."
    
    # Face detection on every frame to prevent blinking
    try:
        face_text = detect_faces(frame)
    except:
        face_text = "Face detection error"

    # Display predictions on frame with bigger text
    cv2.putText(frame, prediction_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.putText(frame, face_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)

    # Display the captured frame
    cv2.imshow('Camera', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and writer objects
cam.release()
cv2.destroyAllWindows()