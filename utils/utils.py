import cv2
import numpy as np
from PIL import Image


def load_img_to_array(img_p):
    img = Image.open(img_p)
    if img.mode == "RGBA":
        img = img.convert("RGB")
    return np.array(img)


def save_array_to_img(img_arr, img_p):
    Image.fromarray(img_arr.astype(np.uint8)).save(img_p)


def dilate_mask(mask, dilate_factor=15):
    mask = mask.astype(np.uint8)
    mask = cv2.dilate(
        mask,
        np.ones((dilate_factor, dilate_factor), np.uint8),
        iterations=1
    )
    return mask


def copy_paste(mask, src_img, dst_img):
    mask = np.expand_dims(mask, axis=-1)
    masked_img = src_img * mask
    # Image.fromarray(masked_img).save('masked_src_img.png')

    mask = np.broadcast_to(mask, src_img.shape)
    dst_img_ = dst_img.copy()
    dst_img_[mask != 0] = masked_img[mask != 0]
    
    return dst_img_


def ext_mask_from_img(mask, img):
    white_color=(255, 255, 255)
    white_image = np.ones_like(img) * white_color
    white_image[mask == 1] = img[mask == 1]
    
    return white_image.astype(np.uint8)


def get_canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)


def save_img(img, path):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    
    img.save(path)