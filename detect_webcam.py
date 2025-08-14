from ultralytics import YOLO
import cv2

# Load trained model
model = YOLO("runs/detect/human_detection_yolo/weights/best.pt")

cap = cv2.VideoCapture(0)  # 0 = default webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model(frame, conf=0.5)
    annotated_frame = results[0].plot()

    cv2.imshow("Live Human Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
