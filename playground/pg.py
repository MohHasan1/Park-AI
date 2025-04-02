import cv2
from ultralytics import YOLO

model = YOLO('./best.pt')

my_file = open("test_coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

image = cv2.imread('./park_2.jpg')

if image is None:
    print("Error: Could not read the image.")

# Perform inference
results = model.predict(image)
print(results)

# Iterate over the results
for result in results:
    boxes = result.boxes.xyxy  # Bounding box coordinates (x_min, y_min, x_max, y_max)
    confidences = result.boxes.conf  # Confidence scores
    class_ids = result.boxes.cls  # Class IDs

    for box, confidence, class_id in zip(boxes, confidences, class_ids):
        x_min, y_min, x_max, y_max = map(int, box)
        label = f"{class_list[int(class_id)]}: {confidence:.2f}"

        if (class_id == 1):
            # Draw the bounding box and label on the image
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display the annotated image
resized_image = cv2.resize(image, (1000, 1000))
cv2.imshow('YOLOv8 Inference', resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
