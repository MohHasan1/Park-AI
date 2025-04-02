import os
import cv2

class ImageProcessor:
    def __init__(self, class_list=None):
        self.original_image = None
        self.last_annotated_image = None
        self.class_list = class_list if class_list else ["car", "free"]
        self.occupied_spots = []
        self.all_boxes = []

    def add_image(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image path does not exist: {image_path}")

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from: {image_path}")

        self.original_image = image.copy()
        return image
    
    def add_image_direct(self, image):
        if image is None or not hasattr(image, 'shape'):
            raise ValueError("Invalid OpenCV image provided.")
        
        self.original_image = image.copy()
    
    def annotate_image(self, results):
        self.all_boxes.clear()
        self.occupied_spots.clear()
        image = self.original_image.copy()

        for result in results:
            boxes = result.boxes.xyxy
            confidences = result.boxes.conf
            class_ids = result.boxes.cls

            for box, confidence, class_id in zip(boxes, confidences, class_ids):
                x_min, y_min, x_max, y_max = map(int, box)
                self.all_boxes.append({
                    "coords": (x_min, y_min, x_max, y_max),
                    "class_id": int(class_id),
                    "confidence": float(confidence)
                })

        # Sort boxes visually: top-to-bottom, left-to-right
        self.all_boxes = sorted(self.all_boxes, key=lambda b: (b["coords"][1] // 50, b["coords"][0]))

        for i, box_info in enumerate(self.all_boxes, start=1):
            x_min, y_min, x_max, y_max = box_info["coords"]
            class_id = box_info["class_id"]

            # Draw box and label
            label = str(i)
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 0), 2)

            # Append the parking spots
            if class_id == 0:
                self.occupied_spots.append(i)

        self.last_annotated_image = image
        return image

    def get_occupied_spots(self):
        return self.occupied_spots

    def get_total_spots(self):
        """Total detected parking spots (both occupied and free)."""
        return len(self.all_boxes)

    def get_parked_spots(self):
        """Total number of parked vehicles (class_id == 0)."""
        return len([b for b in self.all_boxes if b["class_id"] == 0])

    def get_empty_spots(self):
        """Total number of free spots (class_id == 1)."""
        return len([b for b in self.all_boxes if b["class_id"] == 1])
    
    def get_parking_summary(self):
        """Returns a summary of total, parked, and empty spots."""
        parked = self.get_parked_spots()
        empty = self.get_empty_spots()
        total = parked + empty  # or len(self.all_boxes)
        parked_indices = self.occupied_spots  # these are the box numbers

        return {
            "total_spots": total,
            "parked_count": parked,
            "empty_count": empty,
            "parked_indices": parked_indices
    }


    def save_and_show(self, output_path="../inference/output/output0.jpg", show=True, resize_dim=(1000, 1000)):
        if self.last_annotated_image is None:
            raise ValueError(
                "No annotated image found. Run annotate_image() first.")

        cv2.imwrite(output_path, self.last_annotated_image)

        if show:
            resized = cv2.resize(self.last_annotated_image, resize_dim)
            cv2.imshow('Parking Detection', resized)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
