import cv2
import numpy as np
import os


def _pre_process(image_name):
    image = cv2.imread(image_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)[1]
    kernel = np.array([[0, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 1], [0, 1, 1, 0]], np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    image = cv2.resize(image, (800, 800))
    image = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)[1]
    return image


def pre_process_images(src_dir, dst_dir):
    # get all images in src_dir
    images = []
    for file in os.listdir(src_dir):
        if file.endswith(('.png', '.jpg')):
            images.append(file)

    # pre-process images
    for image_name in images:
        image = _pre_process(os.path.join(src_dir, image_name))
        cv2.imwrite(os.path.join(dst_dir, image_name), image)

