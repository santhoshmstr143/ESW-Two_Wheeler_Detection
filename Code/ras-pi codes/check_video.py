from picamera2 import Picamera2
import cv2

# Initialize camera
picam2 = Picamera2()

config = picam2.create_preview_configuration(
    main={"format": "RGB888", "size": (1280, 720)}
)
picam2.configure(config)
picam2.start()

while True:
    frame = picam2.capture_array()
    cv2.imshow("Raspberry Pi Camera 3 - Live", frame)

    # quit window using q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

picam2.stop()
cv2.destroyAllWindows()
