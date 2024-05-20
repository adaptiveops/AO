import cv2

# Specify the path to the Haar Cascade XML file for face detection
face_cascade_path = '/path/to/your/opencv/data/haarcascades/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(face_cascade_path)

# Flag to track if facial recognition is enabled
facial_recognition_enabled = False

# Function to toggle facial recognition
def toggle_facial_recognition():
    global facial_recognition_enabled
    facial_recognition_enabled = not facial_recognition_enabled

# Function to perform face detection
def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    return frame

# Open the Logitech 4K Pro webcam
cap = cv2.VideoCapture(0)  # Use 0 as the camera index for the default camera

# Initial window size parameters
window_width = 640
window_height = 480

while True:
    ret, frame = cap.read()

    # Check if facial recognition is enabled
    if facial_recognition_enabled:
        frame = detect_faces(frame)

    # Rotate the frame by 180 degrees
    rotated_frame = cv2.rotate(frame, cv2.ROTATE_180)

    # Resize the frame according to window size
    resized_frame = cv2.resize(rotated_frame, (window_width, window_height))

    # Display the resized frame
    cv2.imshow('Logitech 4K Pro Webcam', resized_frame)

    # Check for key press events
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('b'):  # Press 'b' to make the window larger
        window_width += 20
        window_height += 20
    elif key == ord('s'):  # Press 's' to make the window smaller
        window_width = max(20, window_width - 20)
        window_height = max(20, window_height - 20)
    elif key == ord('f'):  # Press 'f' to toggle facial recognition
        toggle_facial_recognition()

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()