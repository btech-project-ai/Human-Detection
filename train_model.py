from ultralytics import YOLO

# Load YOLOv8n (nano version for speed; use YOLOv8m or YOLOv8l for accuracy)
model = YOLO("yolov8n.pt")

# Train model
model.train(
    data="datasets.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    name="human_detection_yolo"
)

# Save final model
model.export(format="pt")
