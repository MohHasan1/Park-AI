import cv2
from ultralytics import YOLO

model = YOLO('./best.pt')

# Load class names
with open("test_coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

image = cv2.imread('./top6.webp')

if image is None:
    print("Error: Could not read the image.")
    exit()

# Perform inference
results = model.predict(image)
print(results)

# Collect boxes, confidences, class_ids
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

# Sort boxes top-to-bottom, left-to-right
all_boxes = sorted(all_boxes, key=lambda b: (b["coords"][1] // 50, b["coords"][0]))  # adjust 50 if needed

# Draw boxes with ordered numbers
for i, box_info in enumerate(all_boxes, start=1):
    x_min, y_min, x_max, y_max = box_info["coords"]
    label = str(i)

    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    cv2.putText(image, label, (x_min, y_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 0), 2)

# Resize and show image
resized_image = cv2.resize(image, (1000, 1000))
cv2.imshow('YOLOv8 Inference', resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
