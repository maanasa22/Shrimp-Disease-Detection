import os, cv2, numpy as np, seaborn as sns, matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import argparse, torch, shutil
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score

def main():
    # Setup argparse with minimal but effective parameters
    parser = argparse.ArgumentParser(description="Optimized YOLO Shrimp Disease Detection")
    parser.add_argument('--dataset_path', type=str, default='./dataset', help='Dataset directory path')
    parser.add_argument('--epochs', type=int, default=2, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--img_size', type=int, default=480, help='Image size')
    parser.add_argument('--confidence', type=float, default=0.25, help='Confidence threshold')
    args = parser.parse_args()

    # Paths setup
    dataset_path = Path(args.dataset_path).resolve()
    output_path = dataset_path
    
    if not dataset_path.exists() or not all((dataset_path / d).exists() for d in ['train', 'validation', 'test']):
        print(f"Dataset path {dataset_path} missing or incomplete!")
        return
    
    for subdir in ['results']:
        (output_path / subdir).mkdir(exist_ok=True)

    # Auto-detect classes
    class_names = []
    for item in os.listdir(dataset_path / 'train'):
        if os.path.isdir(dataset_path / 'train' / item) and (dataset_path / 'train' / item / 'images').exists():
            class_names.append(item)
    
    if not class_names:
        class_names = ['healthy', 'wssv', 'ems', 'ihhnv', 'fungal']
    
    print(f"Detected {len(class_names)} classes: {', '.join(class_names)}")
    num_classes = len(class_names)

    # Create YAML config
    yaml_content = f"""train: {dataset_path}/train
val: {dataset_path}/validation
test: {dataset_path}/test

nc: {num_classes}
names: {str(class_names).replace("'", '"')}

# Optimized augmentation for speed
mosaic: 0.7
mixup: 0.2
hsv_h: 0.015
hsv_s: 0.5
hsv_v: 0.7
degrees: 10.0
translate: 0.1
scale: 0.4
shear: 0.0
flipud: 0.2
fliplr: 0.5
"""
    yaml_path = output_path / "data.yaml"
    with open(yaml_path, "w") as file:
        file.write(yaml_content)

    # Check for CUDA and optimize memory usage
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Memory optimization for 7GB RAM, 4GB GPU
    if device.startswith('cuda'):
        torch.cuda.empty_cache()
        # Set lower precision to reduce memory usage
        torch.backends.cudnn.benchmark = True
        # Lower memory usage with deterministic algorithms 
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    # Load smaller YOLOv8 model for 4GB GPU
    try:
        #model = YOLO('yolov8n.pt')
        model = YOLO("C:/Users/vashn/OneDrive/Desktop/pcl_new_working/New folder/shrimp_detection/dataset/best_model.pt")

        
        # nano model for lower memory usage
        print(f"Base model loaded: {model.model.names}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Training with memory-efficient settings
    try:
        print(f"Starting training for {num_classes} classes...")
        results = model.train(
            data=str(yaml_path),
            epochs=args.epochs,
            imgsz=args.img_size,  # Smaller image size (480 instead of 640)
            batch=args.batch_size,  # Smaller batch size (8 instead of 16)
            workers=2,  # Fewer workers to save RAM
            verbose=True,
            patience=5,  # Lower patience for faster convergence
            plots=True,
            save=True,
            project=str(output_path / "runs"),
            name="disease_detection",
            exist_ok=True,
            amp=device.startswith('cuda'),  # Mixed precision for speed
            cos_lr=True,
            lr0=0.01,
            lrf=0.001,
            optimizer='Adam',
            pretrained=True,
            resume=False,
            cache=False,
            device=device,
            close_mosaic=5,  # Close mosaic in last epochs
            max_det=100,  # Fewer detections
            nms=True,  # Enable NMS for faster inference
        )

        # Save the model - simplified with better fallbacks
        best_model_path = output_path / "best_model.pt"
        try:
            # Try direct copy from runs directory first (most reliable)
            run_model_path = output_path / "runs/disease_detection/weights/best.pt"
            if run_model_path.exists():
                shutil.copy(run_model_path, best_model_path)
                print(f"Model saved to {best_model_path}")
            else:
                # Fallback to other saving methods
                try:
                    torch.save(model.model.state_dict(), best_model_path)
                except:
                    if hasattr(model, 'export'):
                        model.export(format="pt", file=best_model_path)
        except Exception as e:
            print(f"Warning: Could not save model: {e}")

        # Load best model for evaluation
        try:
            best_model = YOLO(run_model_path if run_model_path.exists() else best_model_path)
        except:
            best_model = model

        # Evaluate on validation set
        print("Evaluating model...")
        val_metrics = best_model.val(data=str(yaml_path), split='val')
        
        # Test set evaluation with batch processing for memory efficiency
        test_images = []
        for category in class_names:
            test_category_path = dataset_path / 'test' / category / 'images'
            if not test_category_path.exists():
                continue
                
            class_id = class_names.index(category)
            img_files = [f for f in os.listdir(test_category_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for img_name in img_files:
                test_images.append((str(test_category_path / img_name), class_id))
        
        if not test_images:
            print("No test images found.")
            return
            
        print(f"Processing {len(test_images)} test images...")
        
        # Smaller batch size for GPU memory
        batch_size = min(args.batch_size, len(test_images))
        y_true, y_pred = [], []
        confidences = []
        
        # Use tqdm for progress tracking
        for i in tqdm(range(0, len(test_images), batch_size)):
            batch = test_images[i:i+batch_size]
            batch_paths = [item[0] for item in batch]
            batch_labels = [item[1] for item in batch]
            
            try:
                # Clear cache between batches
                if device.startswith('cuda'):
                    torch.cuda.empty_cache()
                    
                batch_results = best_model(batch_paths, conf=args.confidence)
                
                for j, result in enumerate(batch_results):
                    true_class = batch_labels[j]
                    y_true.append(true_class)
                    
                    if len(result.boxes) > 0:
                        confidences_arr = result.boxes.conf.cpu().numpy()
                        classes_arr = result.boxes.cls.cpu().numpy().astype(int)
                        
                        if len(classes_arr) > 0:
                            max_conf_idx = np.argmax(confidences_arr)
                            pred_class = classes_arr[max_conf_idx]
                            confidence = confidences_arr[max_conf_idx]
                        else:
                            pred_class = 0
                            confidence = 0.0
                    else:
                        pred_class = 0
                        confidence = 0.0
                    
                    y_pred.append(pred_class)
                    confidences.append(confidence)
            except Exception as e:
                print(f"Error in batch {i}: {e}")
                continue
        
        if not y_true or not y_pred:
            print("No predictions made.")
            return
            
        # Calculate metrics
        # Generate confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Assuming background is class index 0, remove it
        # Remove row and column corresponding to background (class 0)
        cm = cm[1:, 1:]
        filtered_class_names = class_names[1:]  # remove the background label from the names too

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", 
                xticklabels=filtered_class_names, yticklabels=filtered_class_names)
        plt.title("Confusion Matrix (No Background)")
        plt.tight_layout()
        plt.savefig(str(output_path / "results/confusion_matrix.png"), dpi=200)
        plt.close()

        
        # Classification report
        clf_report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=class_names))
        
        # Save results to CSV
        import pandas as pd
        results_df = pd.DataFrame({
            'Image': [os.path.basename(img[0]) for img in test_images],
            'True_Class': [class_names[y] for y in y_true],
            'Predicted_Class': [class_names[y] for y in y_pred],
            'Confidence': confidences
        })
        results_df.to_csv(str(output_path / "results/prediction_results.csv"), index=False)
        
        # Save performance summary in a compact Markdown file
        with open(str(output_path / "results/performance.md"), "w") as f:
            f.write("# Disease Detection Performance\n\n")
            f.write(f"- Accuracy: {accuracy:.4f}\n")
            f.write(f"- F1 Score: {f1:.4f}\n")
            f.write(f"- Validation mAP@0.5: {val_metrics.box.map50:.4f}\n\n")
            
            f.write("## Class Performance\n")
            f.write("| Class | Precision | Recall | F1-Score |\n")
            f.write("|-------|-----------|--------|----------|\n")
            for class_name in class_names:
                if class_name in clf_report:
                    f.write(f"| {class_name} | {clf_report[class_name]['precision']:.4f} | " 
                            f"{clf_report[class_name]['recall']:.4f} | {clf_report[class_name]['f1-score']:.4f} |\n")
        
        print(f"\n✅ Results saved to {output_path / 'results'}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()