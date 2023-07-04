import cv2
import numpy as np


def _find_nearest_road(image, seed, max_radius=25):
    x, y = seed

    # 生成一个圆形的kernel
    kernel = np.zeros((max_radius * 2 + 1, max_radius * 2 + 1), np.uint8)
    cv2.circle(kernel, (max_radius, max_radius), max_radius, 255, -1)

    # 寻找圆形范围内所有的道路像素点
    pixels = []
    xl = max(-x, -max_radius)
    xr = min(image.shape[1] - x, max_radius)
    yl = max(-y, -max_radius)
    yr = min(image.shape[0] - y, max_radius)
    for i in range(yl, yr):
        for j in range(xl, xr):
            if image[y + i, x + j] == 255 and kernel[max_radius + i, max_radius + j] == 255:
                pixels.append((x + j, y + i))

    # 如果没有找到道路像素点，返回None
    if len(pixels) == 0:
        return None, -1

    # 确定最近的道路像素点
    min_distance = 1e8
    nearest_index = -1
    for i, pixel in enumerate(pixels):
        distance = np.linalg.norm(np.array(seed) - np.array(pixel))
        if distance < min_distance:
            min_distance = distance
            nearest_index = i

    # 根据最近的道路像素点确定方向
    direction = np.array(pixels[nearest_index]) - np.array(seed)
    direction = direction / np.linalg.norm(direction)
    direction = np.array([-direction[1], direction[0]])

    return direction, min_distance


def _draw_rect(image, seed, direction, distance):
    # 在image上绘制一个以(seed[0], seed[1])为中心，方向为dire，边长为dist的正方形
    dist = min(10, distance)
    dist /= 2
    dire = direction
    dire_n = (direction[1], -direction[0])

    # 1. 计算正方形的四个顶点
    p1 = (int(seed[0] - dist * dire[0] - dist * dire_n[0]), int(seed[1] - dist * dire[1] - dist * dire_n[1]))
    p2 = (int(seed[0] - dist * dire[0] + dist * dire_n[0]), int(seed[1] - dist * dire[1] + dist * dire_n[1]))
    p3 = (int(seed[0] + dist * dire[0] + dist * dire_n[0]), int(seed[1] + dist * dire[1] + dist * dire_n[1]))
    p4 = (int(seed[0] + dist * dire[0] - dist * dire_n[0]), int(seed[1] + dist * dire[1] - dist * dire_n[1]))

    # 2. 绘制四条边
    cv2.line(image, p1, p2, 255, 1)
    cv2.line(image, p2, p3, 255, 1)
    cv2.line(image, p3, p4, 255, 1)
    cv2.line(image, p4, p1, 255, 1)

    return [p1, p2, p3, p4]


def generate_layout(image_name, seeds):
    image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
    rects = []
    for seed in seeds:
        direction, distance = _find_nearest_road(image, seed)
        if direction is not None:
            rects.append(_draw_rect(image, seed, direction, distance))
    return rects, image
