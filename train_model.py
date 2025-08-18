import ultralytics
import torch

def train():
    # Check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")

    # Load YOLOv8n (nano model)
    model = ultralytics.YOLO("yolov8n.pt")

    # Train
    model.train(
        data="datasets.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        device=device,
        workers=0,   # ✅ avoid multiprocessing issues on Windows
        name="human_detection_yolov8n"
    )

    # Export final model
    best_model_path = "runs/detect/human_detection_yolov8n/weights/best.pt"
    best_model = YOLO(best_model_path)
    best_model.export(format="pt")

if __name__ == "__main__":   # ✅ required on Windows for torch DataLoader
    train()
