import sys
import os

# Set the library path
os.environ['LD_LIBRARY_PATH'] = '/usr/local/lib' + ':' + os.environ.get('LD_LIBRARY_PATH', '')

# Ensure the virtual environment site-packages are in the path
sys.path.append('/home/aounit1/venv/lib/python3.12/site-packages')
sys.path.append('/home/aounit1/MIPI_Camera/RPI/python')  # Ensure this path is added

from arducam_mipicamera import ArducamCamera
import cv2

def main():
    camera0 = ArducamCamera(0)
    camera1 = ArducamCamera(1)

    camera0.init_camera()
    camera1.init_camera()

    try:
        while True:
            frame0 = camera0.capture()
            frame1 = camera1.capture()

            cv2.imshow("Camera 0", frame0)
            cv2.imshow("Camera 1", frame1)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        camera0.close_camera()
        camera1.close_camera()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
