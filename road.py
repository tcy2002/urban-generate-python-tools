import cv2
import numpy as np


if __name__ == '__main__':
    img = cv2.imread('road.png', cv2.IMREAD_GRAYSCALE)
    shape = img.shape
    points = np.where(img == 255)
    print(points[0].shape)

    with open('road.txt', 'w') as f:
        for x, y in zip(points[1], points[0]):
            f.write(f'{x}\n')
            f.write(f'{y}\n')
