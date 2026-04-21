from ultralytics import YOLO
import os

"""
YOLOv8 Classification Training
Task: Binary classification -> Healthy vs Diseased
"""

DATASET_DIR = r"C:\Users\vashn\OneDrive\Desktop\pcl_new_working\New folder\shrimp_detection\dataset"

MODEL_SIZE = "n"   # n = nano, s = small, m = medium, l = large
EPOCHS = 50        # increase if needed
BATCH = 32
IMGSZ = 224        # 224 is standard for classification

# Pick a classification checkpoint (not detection!)
model = YOLO(f"yolov8{MODEL_SIZE}-cls.pt")

# Train
results = model.train(
    data=DATASET_DIR,   # classification dataset root
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=BATCH,
    project="runs/classify",
    name="shrimp_cls",
    patience=20
)

# Best weights will be saved here:
print("Training complete. Best weights:",
      os.path.join("runs", "classify", "shrimp_cls", "weights", "best.pt"))
