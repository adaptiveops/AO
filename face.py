import cv2

# Open the Logitech 4K Pro webcam
cap = cv2.VideoCapture(0)  # Use 0 as the camera index for the default camera

# Check if the webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Initial window size parameters
window_width = 640
window_height = 480

while True:
    ret, frame = cap.read()

    # Check if frame is read correctly
    if not ret:
        print("Error: Could not read frame.")
        break

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

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
