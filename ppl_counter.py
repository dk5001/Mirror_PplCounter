import cv2
import mediapipe as mp
import numpy as np
import time

# MediaPipe Face Detection setup
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Fixed parameter - count up to 5 people
MAX_PEOPLE = 5

# Stabilization variables
stable_count = 0  # The confirmed stable people count
pending_count = 0  # The currently detected count that's not yet confirmed
last_change_time = time.time()  # When the count last changed
STABILITY_THRESHOLD = 1.0  # Time in seconds for count to be stable

# OpenCV webcam capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB (MediaPipe uses RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the image and detect faces
    results = face_detection.process(rgb_frame)

    # Count detected people
    current_count = 0
    
    # Count detected people
    people_count = 0
    
    if results.detections:
        # Count detections (up to MAX_PEOPLE)
        current_count = min(len(results.detections), MAX_PEOPLE)

        # Draw face detections (can be removed for better performance)
        # for detection in results.detections[:MAX_PEOPLE]:
        #     mp_drawing.draw_detection(frame, detection)

    # Check if the current count differs from pending count
    if current_count != pending_count:
        pending_count = current_count
        last_change_time = time.time()
    
    # Check if the pending count has been stable for the threshold time
    if time.time() - last_change_time >= STABILITY_THRESHOLD:
        if stable_count != pending_count:
            # Update the stable count and log only when it actually changes
            stable_count = pending_count
            print(f"People detected (stable): {stable_count}")

        # people_count = min(len(results.detections), MAX_PEOPLE)
        # print(f"People detected: {people_count}")
        
    cv2.imshow("People Counter", frame)

    # Break loop with ESC
    if cv2.waitKey(1) & 0xFF == 27:  
        break

cap.release()
cv2.destroyAllWindows()