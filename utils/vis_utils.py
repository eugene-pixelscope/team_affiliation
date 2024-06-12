import cv2
import numpy as np


def show_mask_on_image(img, mask, image_weight=0.5):
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    h, w, c = img.shape
    mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    heatmap = cv2.applyColorMap(np.uint8(255 * mask_resized), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255

    cam = (1 - image_weight) * heatmap + image_weight * img_bgr
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)
