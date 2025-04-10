from Model import ParkAI
from FileDB import FileDB
from ImageProcessor import ImageProcessor
from utils.utils import base64_to_cv2, errorJson, successJson

from flask import Flask, request, jsonify


model = ParkAI()
processor = ImageProcessor()
db = FileDB()

app = Flask(__name__)


@app.route('/process', methods=['POST'])
def process_image():
    data = request.json

    if not data or 'image_base64' not in data:
        print("❌ Missing 'image_base64' in request.")
        return jsonify(errorJson("Missing 'image_base64' in request body.")), 400
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

        output_path = f"./Park-AI/inference/outputs/output{updated_index}.jpg"
        print(f"💾 Saving result to: {output_path}")
        processor.save_and_show(output_path, show=False)

        summary = processor.get_parking_summary()
        print(f"✅ Final Result: {summary}")
        print(f"😃 Sending success json to ESP32-CAM....")

        return jsonify(successJson(processor.get_parking_summary()))
    except Exception as e:
        print(f"🚨 Error occurred: {e}")
        print(f"Sending Error json: {errorJson(str(e))}")
        return jsonify(errorJson(str(e))), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5050)
