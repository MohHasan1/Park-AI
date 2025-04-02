import base64
import cv2
import os
import numpy as np
def base64_to_cv2(base64_str):
    try:
        _, encoded = base64_str.split(",", 1) if "," in base64_str else ("", base64_str)
        img_bytes = base64.b64decode(encoded)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Decoded image is None.")
        return image
    except Exception as e:
        raise ValueError(f"Base64 decode error: {e}")


def image_to_base64(image_path, mime_type="image/webp"):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image path does not exist: {image_path}")

    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

    return f"data:{mime_type};base64,{encoded_string}"

# img = image_to_base64("./shared/images/park_4.jpg", mime_type="image/jpg")
# print(img)