import cv2
import time
from picamera2 import Picamera2

def main():
    picam2 = Picamera2()
    # Use a preview configuration for a 640x480 output.
    config = picam2.create_preview_configuration(main={"size": (640, 480)})
    picam2.configure(config)
    
    # Start the camera; it should now run in the binned mode (2028x1520 sensor mode) but output a 640x480 preview.
    picam2.start()
    
    try:
        while True:
            # Capture a frame from the preview stream.
            frame = picam2.capture_array()
            if frame is None:
                print("No frame captured.")
                break
            
            # Display the frame.
            cv2.imshow("Raspberry Pi AI Camera - Preview", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Small delay to reduce CPU usage.
            time.sleep(0.03)
    except KeyboardInterrupt:
        print("Interrupted by user, shutting down.")
    finally:
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
