import base64
import cv2
import os
import numpy as np


def base64_to_cv2(base64_str):
    try:
        _, encoded = base64_str.split(
            ",", 1) if "," in base64_str else ("", base64_str)
        img_bytes = base64.b64decode(encoded)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Decoded image is None.")
        return image
    except Exception as e:
        raise ValueError(f"Base64 decode error: {e}")


def errorJson(error_message):
    return {"data": "", "status": False, "error_msg": error_message}


def successJson(data):
    return {"data": data, "status": True, "error_msg": ""}
