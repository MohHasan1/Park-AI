from Model import ParkAI
from FileDB import FileDB
from ImageProcessor import ImageProcessor
from utils.utils import base64_to_cv2

import os
from flask import Flask, request, jsonify


# Define paths
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, '..', 'inference', 'park_ai.pt')
# image_path = os.path.join(script_dir, '..', 'shared', 'images', 'park_4.jpg')

model = ParkAI()
processor = ImageProcessor()
db = FileDB()

app = Flask(__name__)

@app.route('/process', methods=['POST'])
def process_image():
    data = request.json

    if not data or 'image_base64' not in data:
        print("❌ Missing 'image_base64' in request.")
        return jsonify({"error": "Missing 'image_base64' in request body."}), 400
    else:
        print("📥 Received POST request with base64 image.")

    try:
        print("🖼️ Decoding base64 image...")
        img = base64_to_cv2(data['image_base64'])

        print("🧠 Running YOLO prediction...")
        results = model.predict(img)

        print("🧰 Processing image with ImageProcessor...")
        processor.add_image_direct(img)
        processor.annotate_image(results)

        index = db.read("index") or 0
        updated_index = index + 1
        db.update("index", updated_index)

        output_path = f"./inference/output/output{updated_index}.jpg"
        print(f"💾 Saving result to: {output_path}")
        processor.save_and_show(output_path, show=False)

        occupied = processor.get_parking_summary()
        print(f"✅ Occupied spots: {occupied}")
            
        return jsonify(processor.get_parking_summary())

    except Exception as e:
        print(f"🚨 Error occurred: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)