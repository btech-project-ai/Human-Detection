from ultralytics import YOLO
import torch

def evaluate_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Evaluating on: {device}")

    # Load your trained model
    model = YOLO("runs/detect/human_detection_yolov8n/weights/best.pt")

    # Validate on your dataset
    metrics = model.val(data="datasets.yaml", device=device)

    # Correct attributes
    precision = metrics.mp       # mean precision
    recall = metrics.mr          # mean recall
    map50 = metrics.map50        # mAP@0.5
    map95 = metrics.map          # mAP@0.5:0.95

    print("\n📊 Evaluation Results:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"mAP@0.5:   {map50:.4f}")
    print(f"mAP@0.5:0.95: {map95:.4f}")

if __name__ == "__main__":
    evaluate_model()
