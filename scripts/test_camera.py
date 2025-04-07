import cv2
import time
from picamera2 import Picamera2

def main():
    picam2 = Picamera2()
    
    # Create a still configuration for a 640x480 image.
    # (You can adjust the resolution as needed.)
    config = picam2.create_still_configuration(main={"size": (640, 480)})
    
    # Note: Do not add "post_process_file" here—Picamera2 doesn't allow it.
    picam2.configure(config)
    picam2.start()
    
    try:
        while True:
            # Capture a full request (which includes both the image and its metadata)
            request = picam2.capture_request()
            
            # Get the main image as a NumPy array
            frame = request.make_array("main")
            
            # Attempt to extract AI detections from metadata (if the onboard AI is active)
            metadata = request.get_metadata()
            detections = metadata.get("detections", [])
            
            # Draw detections if any exist
            for detection in detections:
                x, y, w, h = detection.get("bbox", (0, 0, 0, 0))
                label = detection.get("label", "object")
                confidence = detection.get("confidence", 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            cv2.imshow("Raspberry Pi AI Camera - Object Detection", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Release the request so buffers are recycled
            request.release()
            time.sleep(0.03)
            
    except KeyboardInterrupt:
        print("Interrupted by user, shutting down.")
    finally:
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
