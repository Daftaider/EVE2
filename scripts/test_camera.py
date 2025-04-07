import cv2
import time
from picamera2 import Picamera2

def main():
    # Initialize the camera
    picam2 = Picamera2()
    
    # Configure the camera in AI mode.
    # The create_ai_configuration() method is assumed to set up the onboard AI inference.
    # The 'main' configuration sets the resolution – adjust as needed.
    ai_config = picam2.create_ai_configuration(main={"size": (640, 480)})
    picam2.configure(ai_config)
    
    # Start the camera; onboard AI inference is now active.
    picam2.start()
    
    try:
        while True:
            # Capture a frame along with the onboard AI detections.
            # This method is assumed to return a tuple (frame, detections)
            # where detections is a list of dicts containing keys: 'bbox', 'label', and 'confidence'.
            frame, detections = picam2.capture_frame_with_detections()
            
            # If there are any detections, draw them on the frame.
            if detections:
                for detection in detections:
                    # Extract bounding box and detection details.
                    x, y, w, h = detection['bbox']
                    label = detection.get('label', 'object')
                    confidence = detection.get('confidence', 0)
                    
                    # Draw a rectangle and label.
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Show the processed frame.
            cv2.imshow("Raspberry Pi AI Camera - Object Detection", frame)
            
            # Quit on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Small delay for smooth frame updates.
            time.sleep(0.03)
            
    except KeyboardInterrupt:
        print("Interrupted by user, exiting...")
        
    finally:
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
