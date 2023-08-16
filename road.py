import cv2
import numpy as np


def height_map_process(heightmap_name, processed_name, size):
    img = cv2.imread(heightmap_name, -1)
    n = round(np.log2(size / img.shape[0]))
    img = np.where(img > 32768, 32768 + ((img - 32768) >> n), 32768 - ((32768 - img) >> n))
    cv2.imwrite(processed_name, img)


def water_non_recursive(img, x, y, d):
    flag = img[y, x, 0]
    stack = [(x, y)]
    while len(stack) > 0:
        x, y = stack.pop()
        if x < 0 or x >= img.shape[1] or y < 0 or y >= img.shape[0]:
            continue
        if img[y, x, 0] == 255 or img[y, x, 0] > d + flag:
            continue
        img[y, x, :] = [255, 0, 0]
        stack.append((x - 1, y))
        stack.append((x + 1, y))
        stack.append((x, y - 1))
        stack.append((x, y + 1))


def mask_2_txt(mask_name, txt_name):
    mask = cv2.imread(mask_name, -1)
    mask = mask[:, :, 0]
    points = np.where(mask == 255)
    size = len(points[0])

    with open(txt_name, 'w') as f:
        for i in range(size):
            f.write(str(points[1][i]) + ',' + str(points[0][i]) + '\n')


if __name__ == '__main__':
    mask_2_txt(r'.\mask_rugged.png', r'.\mask_rugged.txt')
