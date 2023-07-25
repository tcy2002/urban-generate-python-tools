import cv2
import numpy as np


def process(x):
    if x > 32768:
        return 32768 + ((x - 32768) >> 2)
    else:
        return 32768 - ((32768 - x) >> 2)


def water_around(img, x, y, d, flag):
    if x < 0 or x >= img.shape[1] or y < 0 or y >= img.shape[0]:
        return
    if img[y, x, 0] == 255 or img[y, x, 0] > d + flag:
        return
    img[y, x, :] = [255, 0, 0]
    water_around(img, x - 1, y, d, flag)
    water_around(img, x + 1, y, d, flag)
    water_around(img, x, y - 1, d, flag)
    water_around(img, x, y + 1, d, flag)


def water(img, x, y, d):
    flag = img[y, x, 0]
    water_around(img, x, y, d, flag)


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


if __name__ == '__main__':
    # 生成landmark地图
    img = cv2.imread('landmark_993.png')
    points_road = np.where(img[:, :, 2] == 255)
    points_lake = np.where(img[:, :, 0] == 255)

    img = np.zeros(img.shape, np.uint8)
    with open('road.txt', 'w') as f:
        for x, y in zip(points_road[1], points_road[0]):
            f.write(f'{x},{y}\n')
            cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
        # for x, y in zip(points_lake[1], points_lake[0]):
        #     f.write(f'{x},{y}\n')
        #     cv2.circle(img, (x, y), 1, (0, 255, 0), -1)

    # img = cv2.resize(img, (1024, 1024))
    # cv2.imshow('img', img)
    # cv2.waitKey(0)


    # 调整大小
    # img = cv2.imread('landscape.png', -1)
    # for i in range(img.shape[0]):
    #     for j in range(img.shape[1]):
    #         img[i, j] = process(img[i, j])
    # img = cv2.resize(img, (993, 993))
    # cv2.imwrite('landscape_993.png', img)


    # 标记水域
    # img = cv2.imread('landmark_993.png')
    # lake_point = np.where(img[:, :, 0] == 255)
    # y, x = lake_point[0][0], lake_point[1][0]
    # print(x, y)
    # img[y, x, :] = img[y - 1, x, :]
    # water_non_recursive(img, x, y, 16)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # cv2.imwrite('landmark_993.png', img)
