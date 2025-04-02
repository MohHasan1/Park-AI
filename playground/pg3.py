import cv2
from ultralytics import YOLO
import os

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the absolute path to the image
model_path = os.path.join(script_dir, '..', 'shared', 'models', 'park_ai.pt')
label_path = os.path.join(script_dir, '..', 'shared', 'labels', 'park_ai.txt')
image_path = os.path.join(script_dir, '..', 'shared', 'images', 'park_4.jpg')
output_path = "./playground/output/pg3.jpg"

model = YOLO(model_path)

# Load class names
with open(label_path, "r") as my_file:
    class_list = my_file.read().split("\n")
# class_list = ["cars", "free"]

image = cv2.imread(image_path)

if image is None:
    print("Error: Could not read the image.")
    exit()

# Perform inference
results = model.predict(image)
print(results)

# Collect boxes
all_boxes = []

for result in results:
    boxes = result.boxes.xyxy
    confidences = result.boxes.conf
    class_ids = result.boxes.cls

    for box, confidence, class_id in zip(boxes, confidences, class_ids):
        x_min, y_min, x_max, y_max = map(int, box)
        all_boxes.append({
            "coords": (x_min, y_min, x_max, y_max),
            "class_id": int(class_id),
            "confidence": float(confidence)
        })

# Sort boxes visually: top-to-bottom, left-to-right
all_boxes = sorted(all_boxes, key=lambda b: (
    b["coords"][1] // 50, b["coords"][0]))

occupied_spots = []

# Draw and label
for i, box_info in enumerate(all_boxes, start=1):
    x_min, y_min, x_max, y_max = box_info["coords"]
    class_id = box_info["class_id"]

    label = str(i)
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    cv2.putText(image, label, (x_min, y_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 0), 2)

    # If the detected object is a vehicle
    if class_id == 0:
        occupied_spots.append(i)

# Print the occupied spots
print("Occupied spots:", occupied_spots)

# Resize and show image
resized_image = cv2.resize(image, (1000, 1000))
cv2.imwrite(output_path, image)
cv2.imshow('YOLOv8 Inference', resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
