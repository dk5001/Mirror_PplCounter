import cv2
import mediapipe as mp

# MediaPipe Face Detection setup - this can find multiple faces
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Set maximum number of people to track
MAX_PEOPLE = 5

# OpenCV webcam capture
cap = cv2.VideoCapture(0)  # 0 is usually the default camera

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB (MediaPipe uses RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the image and detect faces
    results = face_detection.process(rgb_frame)
    
    # Count detected people
    people_count = 0
    if results.detections:
        # Count detections (up to MAX_PEOPLE)
        people_count = min(len(results.detections), MAX_PEOPLE)
        
        # Draw face detections
        for detection in results.detections[:MAX_PEOPLE]:
            mp_drawing.draw_detection(frame, detection)
            
            # Get bounding box coordinates
            box = detection.location_data.relative_bounding_box
            h, w, c = frame.shape
            x, y = int(box.xmin * w), int(box.ymin * h)
            
            # Draw person number above each detection
            person_id = results.detections.index(detection) + 1
            cv2.putText(frame, f"Person {person_id}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display frame and count
    cv2.putText(frame, f"People Detected: {people_count}/{MAX_PEOPLE}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("People Detection", frame)

    # Break loop with ESC
    if cv2.waitKey(1) & 0xFF == 27:  # 27 is the ASCII code for the ESC key
        break

cap.release()
cv2.destroyAllWindows()