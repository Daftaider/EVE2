import cv2
import time
from picamera2 import Picamera2

def main():
    # Initialise the camera
    picam2 = Picamera2()
    # Configure the camera for still captures (which, for the AI Camera, enables the onboard AI mode)
    config = picam2.create_still_configuration(main={"size": (640, 480)})
    picam2.configure(config)
    
    # Start the camera; onboard AI inference is automatically enabled for the IMX500 sensor.
    picam2.start()
    
    try:
        while True:
            # Capture metadata which includes the image and (if available) detection results.
            metadata = picam2.capture_metadata()
            # The captured image is stored under the 'main' key.
            frame = metadata.get("main")
            # The detections should be provided in the metadata under 'detections'.
            # (Check your docs if they use a different key such as 'ai_detections'.)
            detections = metadata.get("detections", [])
            
            if frame is None:
                print("No frame captured.")
                break
            
            # Process each detection and overlay bounding boxes and labels.
            for detection in detections:
                # Expecting each detection as a dict with keys: 'bbox', 'label', and 'confidence'
                x, y, w, h = detection.get("bbox", (0, 0, 0, 0))
                label = detection.get("label", "object")
                confidence = detection.get("confidence", 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Display the result.
            cv2.imshow("Raspberry Pi AI Camera - Object Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(0.03)
            
    except KeyboardInterrupt:
        print("Interrupted by user, exiting...")
    finally:
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
