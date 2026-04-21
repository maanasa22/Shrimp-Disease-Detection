# app.py
from flask import Flask, request, jsonify, render_template, send_from_directory
import os
from pathlib import Path
from ultralytics import YOLO

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit uploads to 16MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained classification model
model = None
try:
    model_path = Path("runs/classify/shrimp_cls/weights/best.pt")
    if model_path.exists():
        model = YOLO(model_path)
        print(f"✅ Loaded trained classification model from {model_path}")
    else:
        raise FileNotFoundError("Trained classification model not found. Please train first.")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# Define class names (order must match your dataset folders: diseased=0, healthy=1)
class_names = ['diseased', 'healthy']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    # Save the uploaded file
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    try:
        # Run classification
        results = model(file_path)
        res = results[0]

        top1 = res.probs.top1           # index of top class
        confidence = float(res.probs.top1conf)
        class_name = class_names[top1]

        # Get description and recommendations
        description = get_description(class_name)
        recommendations = get_recommendations(class_name)

        return jsonify({
            'class': class_name,
            'confidence': round(confidence * 100, 2),
            'description': description,
            'recommendations': recommendations,
            'image_path': f"/uploads/{file.filename}"
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Recommendations for classification
def get_recommendations(class_name):
    recommendations = {
        'healthy': [
            'Continue regular health monitoring',
            'Maintain optimal water quality',
            'Ensure proper nutrition'
        ],
        'diseased': [
            'Isolate affected individuals',
            'Improve water quality immediately',
            'Consult an aquaculture specialist'
        ]
    }
    return recommendations.get(class_name, ['Consult with an aquaculture specialist'])

# Descriptions for classification
def get_description(class_name):
    descriptions = {
        'healthy': 'This shrimp appears healthy with no visible signs of disease.',
        'diseased': 'This shrimp shows signs of disease. Please take necessary actions immediately.'
    }
    return descriptions.get(class_name, 'Unknown condition detected. Please consult with a specialist.')

if __name__ == '__main__':
    app.run(debug=True)
