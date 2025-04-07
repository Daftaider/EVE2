import cv2
import time
from picamera2 import Picamera2

def main():
    picam2 = Picamera2()
    
    # Create a still configuration for a 640x480 preview image.
    # Adjust this size as needed.
    config = picam2.create_still_configuration(main={"size": (640, 480)})
    
    # Add the post-process file to enable onboard object detection.
    # This file is provided by Raspberry Pi (for example, MobileNet SSD for the IMX500).
    config["post_process_file"] = "/usr/share/rpi-camera-assets/imx500_mobilenet_ssd.json"
    
    picam2.configure(config)
    picam2.start()
    
    try:
        while True:
            # Capture a full request (which includes the image and its metadata)
            request = picam2.capture_request()
            
            # Extract the main image as a NumPy array.
            frame = request.make_array("main")
            
            # Extract detections from the metadata.
            # The key "detections" should contain a list of detection dicts if the onboard AI is active.
            detections = request.get_metadata().get("detections", [])
            
            if frame is None:
                print("No frame captured.")
                break
            
            # Process detections: draw bounding boxes and labels.
            for detection in detections:
                # Expect each detection to include a bounding box (bbox), a label, and a confidence score.
                x, y, w, h = detection.get("bbox", (0, 0, 0, 0))
                label = detection.get("label", "object")
                confidence = detection.get("confidence", 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Display the frame.
            cv2.imshow("Raspberry Pi AI Camera - Object Detection", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Release the request so the buffers are recycled.
            request.release()
            time.sleep(0.03)
            
    except KeyboardInterrupt:
        print("Interrupted by user, shutting down.")
    finally:
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
